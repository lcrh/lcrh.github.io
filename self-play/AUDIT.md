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
