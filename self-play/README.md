# Self-play · pretraining from zero data

[Open the playground](https://lcrh.github.io/self-play/) ·
[Paper](https://arxiv.org/abs/2609.30063) · [Independent fidelity audit](AUDIT.md)

Two randomly initialized, trainable causal transformers run entirely in a browser
worker. A generator writes executable programs, a learner predicts their byte
outputs, and an actual gradient-based reward changes the generator's curriculum.
There are no downloaded weights, model services, or prewritten training programs.
TensorFlow.js (4.22.0) and MathJax (3.2.2) are bundled locally with their licenses. The example in the editable sandbox is evaluation
only. The page never loads the verification results below.

Press **Start from zero**. The first seconds mostly teach the learner padding.
Over the next minutes, inspect programs and compare the fixed evaluation probes
against a learner trained on a fixed program prior. **Count** is useful once the
generator begins producing arithmetic sequences; **Repeat** can improve sooner.
Outcomes vary with seed, runtime and training duration.

## Implementation

This implements the mechanism of Cowsik et al., *Self-Play Pretraining with Zero
Data* (2026), at a much smaller scale. The details are inspectable without a build
step:

| File | Responsibility |
| --- | --- |
| `machine.mjs` | Seeded random source, executable Brainfuck and ten macro expansions, mutation |
| `model.mjs` | Causal transformer, explicit backpropagation through all weights, AdamW, cached sampling |
| `trainer.mjs` | Program pool, actual gradient reward, policy gradients, expert iteration, archives, evaluation |
| `worker.mjs` | Background training, matched-token control, save/restore, isolated sandbox |
| `gpu.mjs` | WebGPU transformer forward passes and full autodiff gradients |
| `config.mjs` | Validated presets and actual model/training configuration |
| `knowledge-data.mjs`, `knowledge.mjs`, `entrypoints.mjs` | Shared nested explanations, wiki and connected maps |
| `diagnostics.mjs`, `simulator.mjs` | Captured numerical diagnostics and exact stepped execution |
| `app.mjs` | Live data display, plots, controls, local persistence |

The presets use one pre-normalized layer, two attention heads and an MLP of
twice the model width. The settings foldout supports substantially larger custom
models: up to 128 layers, width 8,192, 128 heads, and 32,768-token contexts, subject
to valid integer shapes and the actual selected device's single-buffer limits.
Width must divide evenly into the number of heads; MLP multipliers, program pools,
execution budgets and replay capacity are independently configurable. These broad
technical bounds are not a recommendation to select every maximum simultaneously.
All configurations use RMS normalization, learned positions and ReLU activations. These
are small ordinary attention networks, not Llama replicas. Generator embeddings
and output head use 19 instruction tokens; the learner's output head uses all
256 bytes. Each has an additional learned beginning-of-sequence embedding.

Selecting **Custom** opens the settings and preserves their current values.
Editing any numeric field immediately labels the configuration as a custom draft;
it does not reset the current run. **Apply** creates the configured models.
Selecting Quick or Long run restores that preset's actual defaults and starts a
fresh run.

The foldout estimates CPU model/gradient/Adam state, the optional control,
recursive historical copies and forward/backward working storage. It does not
infer available RAM or promise GPU utilization. WebGPU validates each tensor
against the initialized device's `maxStorageBufferBindingSize` and `maxBufferSize`
before allocating model arrays; there is no universal 128 MiB GPU cap. The CPU
path retains a conservative single-array bound just below 2 GiB. Aggregate memory,
backend workspaces, checkpoint serialization and JavaScript program banks can
exhaust memory even when every individual allocation meets those limits.

| Setting | Quick | Long run |
| --- | ---: | ---: |
| Width | 24 | 48 |
| Output context | 64 | 128 |
| Maximum program tokens | 48 | 96 |
| Programs / round | 12 | 24 |
| Primitive execution steps / program | 4,096 | 16,384 |
| Tape cells | 256 | 256 |
| FIFO replay entries | 256 | 512 |
| Learner learning rate | 0.0015 | 0.001 |
| Generator learning rate | 0.0004 | 0.0002 |

Both use AdamW, β₁ = 0.9, β₂ = 0.999, ε = 1e−8, weight decay 0.01,
global gradient norm clipping at 1, uniform-prior KL coefficient 0.02, and
expert-iteration weight 1. No warmup, scheduling, ensembling or reward tuning uses
the displayed evaluation results. The fixed-prior comparison has the same learner
initialization and number of training output bytes, but less computation per round.
It is not a wall-clock-matched comparison or an ablation isolating policy
gradients from mutation/replay.

### The actual training step

The default pool is half fresh generator samples, one quarter local mutations,
and one quarter replay; empty banks fall back to fresh samples. A whole randomly
initialized neural policy chooses every fresh instruction. Length-capped programs
retain their sampled-prefix likelihood; a forced terminator is not counted as a
sampled action. Execution stops at the first `F`, program end, output limit or
primitive-step budget. Unmatched brackets are no-ops. Cells and tape positions
wrap; input reads fresh independent uniform bytes. Unemitted output is zero padded
and included in learner training, matching the paper's stated output semantics.

At round index `e` (the number of completed learner updates), every output row's
full cross-entropy gradient is calculated before either update. Its reward is
the absolute inner product with
`lr / (sqrt(v_hat) + epsilon) * (theta[floor(e/2)] - theta[e])`.
The Adam second moment is bias corrected. The direction covers all trainable
learner parameters. At initialization the historical difference is zero, so all
alignment rewards are zero; expert iteration safely contributes zero then.

The generator's sequence likelihood includes sampled termination. The policy
gradient uses rewards normalized over the entire pool, the uniform-token prior
penalty and replay importance ratios clipped in log space to ±20. The original
proposal log probability stays attached to a replay entry. Mutations have no
such proposal probability and remain excluded from the policy gradient even if
replayed later; they still participate in reward-weighted expert iteration and
learner training. Expert iteration uses summed sequence log probabilities,
weighted by each row's share of the total positive reward.

Mutation parents are selected uniformly across occupied quality-diversity
niches, then uniformly within a niche. The archive uses dynamic matched-loop depth
(including expanded macros), four length buckets, up to eight entries per niche,
and 0.97 reward decay per round. Replay entries are sampled without replacement
from a bounded FIFO bank, with fresh random execution inputs. These implementation
choices are explicit finite-resource versions of the paper's pool construction.

### Exact historical weights without an ever-growing checkpoint queue

Each trainer owns a lazily created, identically seeded copy of itself that advances
only to `floor(e/2)`. That copy owns a quarter-speed copy, and so on. Training
decisions use only deterministic model state and a private RNG; evaluation and UI
actions consume none of that RNG. Therefore the slower copy reconstructs exactly
the weights that the main trainer had at the requested round.

There are O(log rounds) trainer states, each with bounded program banks. Their
combined training work is less than one extra main run: `e/2 + e/4 + … < e`.
This retains the actual growing lookback horizon, without a one-step proxy,
approximate checkpoint selection, or linearly growing weight history. The
reference test stores every snapshot and compares complete weights and reward
vectors against this construction. Floating-point differences across JavaScript
engines can eventually change sampled trajectories, so bit-exact continuation
is guaranteed by the tests within the same runtime, not across all browsers.

### WebGPU execution

WebGPU is the default, with an explicit CPU compatibility option in Model & training settings. Forward passes and all learner/generator training gradients execute on the GPU; interpreter execution, cached token sampling, reward reductions, Adam updates and evaluation use the CPU. TensorFlow.js CPU forwarding is disabled for GPU tensor operations. The page reports its actual backend and does not silently fall back.

The full historical trainer hierarchy uses the same backend. GPU results differ slightly from CPU floating-point reductions and can diverge in sampled trajectories over time. Same-device/runtime checks reproduce historical weights and checkpoints exactly; cross-device bit identity is not promised. Checkpoints remember their backend. A failed operation inside a round blocks continuation/export until a new run or completed checkpoint is restored.

### Paper companion and interaction map

[Open the companion](https://lcrh.github.io/self-play/wiki.html). The 88 articles cover the method, equations, experiments, scaling argument and appendices, with linked source sections. All are reachable within three links of an interface entrypoint. [KNOWLEDGE-MAP.md](KNOWLEDGE-MAP.md) separately records the UI entrypoints and article graph; the interactive companion includes both maps, search, prerequisites, related concepts and backlinks.

Hover for a preview, click **Keep open**, and follow links inside to any depth. Nested panels retain the original program/graph-point data while training advances. Escape closes one level. Interactive equations use self-hosted MathJax. The inspector execution viewer uses the training interpreter and captured input RNG state: step/play/reset shows source and expanded instruction pointers, memory and emitted output.

## Reading the displays

- The curriculum contains the **current actual batch**, sorted by descending reward by default (sampling order is selectable). The
  inspector follows its highest-reward row until you select one. This selection
  only controls the display. In the larger preset the row tape previews the first
  64 of 128 output bytes; the inspector plots the full output.
- Dark underlined cells are padding, including padding after a step-budget halt.
  Emitted zeros are shown separately. The emitted fraction is for the current
  batch, not cumulative.
- Training loss is recorded before the update. The training curves separate
  emitted bytes and padding and average the last ten retained plot points. The
  scalar training loss includes both. Since the curriculum changes, this curve
  cannot by itself establish transfer.
- Probes are fixed, generated from a separate seed, and evaluated without updates
  at initialization, round 1, and every 20 rounds. Repetition, cycles, counting and
  random noise each use four sequences; text uses one short English passage.
  These are deliberately small demonstrations, not the paper's benchmarks.
- Custom evaluation accepts UTF-8 text or decimal/hex bytes. It scores without training, padding or archive insertion. Long sequences are split into nonoverlapping context-sized chunks, with boundaries explained in the result.
- Optional graphs show actual reward mean ± population standard deviation, output percentage, program-length mean ± standard deviation, and generator entropy. Hovered plot points use retained data rather than the latest batch.
- The first/last-16-byte comparison is descriptive: context, position and target
  content all differ. It is not a controlled test of in-context learning.
- Forecasts are greedy next-byte predictions conditioned on the true preceding
  output bytes. They are not free-running completions. The learner receives no
  source code. At long durations older plot points are downsampled to bound memory;
  model training and the historical reward are not downsampled.

**Difficulty** in Run options is an explicit ablation: it substitutes output
cross-entropy for the alignment reward. A new run is required to change objective,
model size, seed or control. CPU usage can change live without affecting the
training trajectory. Quick mode is the default because it offers more rounds per
minute. The larger model is an option for sustained exploration, not a promise
of improved results within a short visit.

## Persistence and privacy

Training uses only this device. IndexedDB keeps the latest local checkpoint,
approximately once per minute and on pause. Run options → **Resume saved run**
restores it, paused. **Download checkpoint** and **Load checkpoint** transfer the
full state: weights, optimizer moments, RNG, replay/archive entries, historical
trainers and chart history. No checkpoint is sent to a server. Closing the tab
stops computation; background browser throttling and device sleep can slow it.
Checkpoints from different runs overwrite the single local autosave slot. Download
a checkpoint to keep multiple experiments.

## Verification

No installation or build is required. Serve the repository root with an ordinary
static HTTP server and open `/self-play/`; browser module workers require HTTP(S).
From the repository root, on Node 24:

```sh
node self-play/test.mjs
node self-play/audit-test.mjs
node self-play/test-worker.mjs
node self-play/test-companion.mjs
node self-play/benchmark.mjs 2000 7
node self-play/benchmark.mjs 1500 7 prior
node self-play/benchmark.mjs 1000 23
```

The tests cover numerical gradients, causality, sampling likelihood, execution
semantics, the reward's full directional derivative and Adam preconditioner,
historical reconstruction, objective arithmetic, evaluation isolation, complete
checkpoint continuation, and the browser worker's message protocol. A separate
agent reviewed both the source paper and actual implementation. Its report and
additional tests are included in [AUDIT.md](AUDIT.md).

The original **CPU backend** verification on 2026-09-26 produced the following held-out **bits per byte**.
Each number is measured from real model probabilities. Full logs with emitted
program examples are in [verification-results.json](verification-results.json).

| Run | Rounds | Repeat | Cycle | Count | Noise | Text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial, seed 7 | 0 | 8.08 | 8.03 | 8.00 | 8.01 | 7.98 |
| Self-play, seed 7 | 300 | 6.44 | 8.35 | 8.17 | 8.29 | 8.20 |
| Self-play, seed 7 | 1,500 | 4.80 | 8.76 | 6.06 | 8.46 | 7.55 |
| Fixed prior, seed 7 | 1,500 | 6.79 | 8.48 | 8.27 | 8.53 | 8.41 |
| Self-play, seed 7 | 2,000 | 3.19 | 8.57 | 4.23 | 8.72 | 10.50 |
| Self-play, seed 23 | 1,000 | 3.17 | 8.12 | 8.17 | 8.56 | 8.42 |

The 2,000-round Node run took about 147 seconds on the development Mac. In a
separate actual browser run with the control enabled, 1,480 rounds took about
156 seconds; held-out counting loss was 6.08 versus 8.21 for the control. Browser
timings depend on device and load. The interface was also inspected at a 390-pixel
mobile viewport; the runtime remained responsive and reported no console errors.

These results show a short-run payoff on simple structure and also its limits:
counting improved in one seed while cycles did not; text performance could worsen
substantially; another seed had not learned counting by round 1,000. This is a
faithful miniature experiment, not evidence that leaving it running will reproduce
the paper's broader scaling, multimodal transfer, or mathematical discoveries.

## GPU and companion verification (2026-09-26)

Open `test-gpu.html` on a local server for actual WebGPU checks. Learner gradient comparisons (including two layers and 256 outputs) differed from the manual reference by at most 5.96e-8; mixed-sign generator gradients by at most 3.58e-7. Ten real GPU rounds matched a retained-history reference exactly. Same-device checkpoint continuation matched and zero tensors remained allocated. A default browser run completed 467 rounds with its prior control in 72 seconds of measured training time; this is a throughput observation, not a transfer claim.

`test-ui.html` exercises the actual page: GPU startup, sorted rows, nested mathematical explanations, frozen diagnostics, custom evaluation, stepped sandbox execution, and both maps. `test-companion.mjs` checks graph reachability, parameter counts, exact interpreter replay, reward decomposition, statistics and checkpoint continuation. Original preset CPU training weights, RNG and banks remain bit-identical after instrumentation.
