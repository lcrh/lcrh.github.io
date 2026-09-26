# Independent conceptual audit

Reviewed on 2026-09-26 by a separate agent, as requested by the user. This review
read the paper independently before inspecting the implementation. Production
code was not modified during the review.

## Verdict

**No loadbearing conceptual shortcut was found.** This is an actual small-scale
implementation of the paper's self-play training mechanism, not a simulation of
its expected results. Both models learn in the browser; generated programs really
execute; their output really supplies the learner's training data; and the
generator's reward really comes from the learner's parameter gradients and
historical movement.

This verdict concerns mechanism fidelity. It does not certify the paper's
empirical claims, guarantee interesting discoveries, or establish that this small
implementation reproduces its scaling, broad natural-data transfer, or in-context
learning results.

## Mechanisms inspected

- **Trainable autoregressive models.** `model.mjs` implements causal softmax
  self-attention, residual connections, normalization and a feed-forward layer.
  Reverse-mode gradients reach every parameter block. Teacher forcing shifts
  targets correctly; future targets cannot leak into current predictions.
  Incremental program sampling agrees with full sequence likelihood evaluation.
- **Executable program search.** `machine.mjs` includes all eight Brainfuck
  operations, the paper's ten macro expansions, and termination. Programs are
  sampled token by token, not selected from prewritten sequence families. The
  machine uses byte wraparound, a circular tape, bounded execution, random input,
  explicit emissions, zero padding and harmless unmatched brackets. The abstract
  language is universal; this actual finite-budget execution is necessarily
  bounded, as in the paper's experiments.
- **Learner objective.** Every output row trains a 256-way next-byte predictor
  with mean cross-entropy, including padding. Fresh, mutated and replayed programs
  all contribute. The learner is never given their source code.
- **Historical gradient reward.** The reward takes the absolute inner product
  of each row's full learner gradient and the historical parameter difference,
  multiplied coordinatewise by the bias-corrected Adam preconditioner and learning
  rate. It is measured before the learner update. It is not loss reduction,
  novelty, sequence difficulty, or a hand-authored complexity score.
- **Exact growing lookback.** Recursive deterministic trainers reconstruct the
  actual checkpoint at half the current round count. They reproduce the same
  training process, including its own historical reward. Retained-history tests
  compare complete weights and reward-bearing results exactly. Their state grows
  logarithmically in rounds; the sum of replayed training rounds is less than the
  main run's round count. This is a genuine alternative to retaining every
  historical weight snapshot.
- **Generator learning.** Policy-gradient weights use pool reward normalization,
  the uniform-token prior penalty and sequence-level replay importance ratios.
  Replay retains the original sampled log probability. Mutations, including
  replayed mutations lacking a proposal probability, are excluded from policy
  gradients. Reward-weighted expert iteration includes them. The zero-reward
  startup case remains finite.
- **Exploration and retention.** The pool contains fresh generation, local
  mutation and replay. Mutation parents come from a decaying quality-diversity
  archive indexed by dynamic loop depth and body length. Programs execute with
  fresh random inputs. Neither the bank nor the archive has preloaded programs.
- **Evaluation isolation.** Fixed probes and the editable program sandbox only
  execute and evaluate; they do not update either model or enter the program
  banks. The fixed-prior comparison starts with identical learner weights and
  trains on the same count of byte tokens. It is a comparison against fixed-prior
  pretraining, not a compute-matched ablation of only one generator objective.

## Independent checks

The existing suite passed with `node self-play/test.mjs`: interpreter cases,
39 finite differences covering all parameter blocks, genuine loss reduction,
sampling likelihood, exact historical reconstruction, checkpoint continuation,
evaluation isolation, and policy-objective edge cases.

Additional checks are retained in `audit-test.mjs`; run
`node self-play/audit-test.mjs`. These passed:

1. Changing current and future targets leaves the current and earlier prediction
   distributions exactly unchanged.
2. Forty independently seeded sampled sequences, including capped sequences,
   have likelihoods matching full teacher-forced evaluation.
3. Every learner parameter block changes during training.
4. An independently evaluated finite directional derivative matches the reward
   direction: analytic magnitude **0.026738**, numerical magnitude **0.026762**.
5. Direct reconstruction of the bias-corrected Adam/history direction agrees
   over all learner parameters.
6. Independently calculated policy-gradient/KL/replay weights match the code;
   mutation receives expert-iteration weight without policy-gradient weight.
7. Evaluating fixed probes and a manually supplied program leaves serialized
   training state unchanged; fixed-prior and self-play learners initialize
   identically.

## Reductions and interpretation risks

The single-layer transformer, narrow width, learned positions, ReLU layer,
restricted contexts, finite program lengths, small pools, bounded banks and
execution budgets are substantial quantitative reductions. The page discloses
them. They preserve the training mechanism but greatly restrict its capabilities.

Early learning can largely consist of predicting zero padding or constant
outputs. The separate emitted-byte/padding displays and fixed probes make this
failure mode observable. Improvements on a few fixed probes are suggestive,
not a statistically strong transfer result; different random seeds can behave
differently. The short English probe is particularly insufficient as evidence
of natural-language performance.

First-versus-last-token losses are descriptive within-sequence measurements:
position and target content differ along with available context. They should not
be described as a controlled demonstration that additional context caused an
improvement. **Resolved and independently verified:** the card now says
“Within-sequence prediction” and explicitly notes that position/content also
differ. A minor macro description was also corrected and verified: `W` now says
“increment right & clear current,” matching its behavior when the current cell
is nonzero. Neither finding required a training-mechanism change.

A subsequent display correction includes all 128 prediction positions in the
larger preset; this reads existing model probabilities and does not change the
training mechanism. The worker integration suite also passed independently with
`node self-play/test-worker.mjs`, exercising checkpoint roundtrips with the
fixed-prior control, sandbox isolation, resume, and larger-model training.

No hand-authored training curricula, trained weight assets, scripted learning
curves, proxy learner, or artificial discovery injection was found. The review
did not independently repeat long browser runs or verify hosting and interaction
behavior; those are separate operational checks.

## Reference

[Self-Play Pretraining with Zero Data, §2 and Appendices E–G](https://arxiv.org/html/2609.30063v1).
The checklist above is an assessment of this implementation against that source;
it is not an endorsement of the paper's empirical conclusions.


## Extension audit: companion, configurable transformers and WebGPU

The independent reviewer checked all companion articles against the paper and source. Corrections clarified the optimal expected noise loss (8 bits/byte, with biased predictions potentially worse), Adam startup behavior, probe composition, archive depth and entropy averaging. All source anchors and conceptual links were verified. The core historical reward, policy-gradient, KL, replay, mutation and expert-iteration explanations were found faithful.

The reviewer then checked actual multilayer and GPU tensor code. Added independent tests cover 87 finite differences in a three-layer/four-head model, causality, cached sampling, mixed-sign generator gradients, 12 asynchronous rounds compared with retained historical snapshots, complete optimizer/RNG/bank equality, checkpoint continuation and tensor disposal. These independent tensor-graph tests run on TensorFlow’s CPU test executor; the separate browser `test-gpu.html` supplies actual GPU-device evidence.

The audit identified one failure-state issue: an interrupted GPU round could advance randomness before an error and then be resumed. This has been fixed by marking partial training failures fatal, preventing continuation/export, and requiring a fresh run or completed checkpoint. Restore also rejects mismatched main/control round counts. GPU reproducibility claims are scoped to the same device/backend/runtime, not arbitrary devices.

## Follow-up: empty output and model depth

The paper explicitly zero-extends short output in [Appendix E](https://arxiv.org/html/2609.30063v1#A5).
Its [learner objective](https://arxiv.org/html/2609.30063v1#S2.SS2) averages output
content-token losses; no explicit padding mask was found in the paper. The
implementation trains on the full fixed-length byte row. Consequently `F` emits
nothing but supplies constant zeros, and `.F` supplies exactly the same training
values despite genuinely printing its first zero. This behavior is consistent
with the stated environment, rather than evidence that execution was bypassed.
Masking unprinted positions would be an explicit environment ablation, not a
correction justified by an author-specified mask. The authors' training code was
not independently inspected to resolve any ambiguity beyond the paper text.

In a real WebGPU check with Quick defaults except four layers, seed 7, immediate
termination probability remained approximately 5.0–5.4% through round 150. Mean
`F` reward in the 25-round windows ending at rounds 25/50/75/100/125/150 was
0.404/1.537/1.562/0.475/0.123/0.0476. Identical all-zero rows had exactly equal
rewards within each round. A separate CPU check of Quick defaults, seed 7, ended
at round 300 with first-token `F` probabilities 4.41% (one layer) and 2.94%
(four layers). The GPU harness is retained as `test-depth-gpu.html` and
`test-depth-gpu.mjs`; it records source-specific frequencies and reward windows.

These observations show transient high empty-output reward without immediate
termination collapse in these bounded runs. They do not establish stability for
other seeds, larger settings, or longer training. Reward-sorted rows mix fresh,
replayed and mutated programs and must not be read as policy probabilities.
Absolute gradient alignment can also reward a negative signed inner product;
it guarantees neither monotonic reward decay nor eventual curriculum complexity.

**Subsequent numerical failure:** continuing that four-layer GPU run showed finite
generator probabilities at round 375 (`p(F)` = 0.0514047), followed by NaN
probabilities by round 400 while learner rewards remained finite. The sampler's
fallback selected the final vocabulary entry, `F`, when NaN comparisons failed.
The earlier bounded observations therefore do not establish long-run numerical
stability. The first-failure harness now checks GPU outputs, parameter gradients
and Adam boundaries, and retains the preceding complete trainer checkpoint and
the offending operation. This failure must not be interpreted as learned
termination behavior.

The first invalid gradient occurred at update 378 with finite model weights
(maximum magnitude 1.11355) and finite objective coefficients (−0.1564 to 0.2944).
The exact pooled TensorFlow graph on its CPU executor matched the independent
manual derivative within 9.54e−7. Repeating the former pooled WebGPU graph
produced incorrect finite gradients as well as the original NaNs: observed
maximum errors included 2.4936 and 7.04e10. Each of the twelve individual GPU
rows instead matched the manual derivative within 4.17e−7. This localizes the
observed fault to pooled GPU graph execution; the specific underlying backend
kernel defect has not been identified.

The production workaround computes each unchanged signed weighted row gradient
on the GPU, sums checked values in double-precision CPU buffers, then validates
the Float32 result before copying any gradient into the model. There remains one
generator Adam update. No learning-progress formula, padding convention, policy
coefficient, prior or program filter changes. The summation schedule changes
floating-point arithmetic, so current exports use outer checkpoint format
version 2 and GPU revision `webgpu-rowwise-v2`. Older format-version-1 exports,
including CPU exports, are rejected with a fresh-start instruction. In
particular, the new arithmetic cannot safely replay old pooled-GPU histories.

The exact-input fixture `fixtures/gpu-gradient-round378.json` preserves weights,
program tokens and objective coefficients without optimizer history. The real
GPU regression passed three times with maximum CPU discrepancy 1.252e−6.
Tests also passed second-row NaN rejection with no partial gradient writes,
same-input repeatability, ten-round exact history, checkpoint continuation and
zero retained tensors. A fresh corrected four-layer Quick run, seed 7, then
completed 500 actual GPU rounds in 193.1 seconds, with finite checks throughout
and zero retained tensors. Final first-token `F` probability was 0.02453 and
entropy 4.1710 bits. In the final 25 rounds, fresh `F` frequency was 3.33%, mean
`F` reward was 0.02888, and mean reward for rows containing a nonzero byte was
0.37689; no top-three displayed row was `F`. This is bounded evidence for the
specific numerical fix, not an arbitrary-duration stability or transfer claim.
The actual browser interaction suite also passed GPU startup, sorted rows,
six-level nested math explanations, frozen diagnostics, units, custom evaluation,
stepped execution and both maps after the fix.
