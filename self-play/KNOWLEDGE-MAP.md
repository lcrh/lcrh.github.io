# Paper companion map

88 articles; 60 entrypoints. All articles are reachable; maximum shortest path is 3 links.

## Interface entrypoints

| Location | Selector | First concept |
|---|---|---|
| Overview | `header h1` | [overview](wiki.html#article=overview) |
| Generator in introduction | `header [data-concept="generator"]` | [generator](wiki.html#article=generator) |
| Predictor in introduction | `header [data-concept="learner"]` | [learner](wiki.html#article=learner) |
| Reward in introduction | `header [data-concept="gradient-reward"]` | [gradient-reward](wiki.html#article=gradient-reward) |
| Model settings | `#model-settings summary` | [transformer](wiki.html#article=transformer) |
| Seed label | `label[for="seed"]` | [seed](wiki.html#article=seed) |
| Fixed-prior help | `.help-button` | [prior-control](wiki.html#article=prior-control) |
| Round counter | `#clock` | [round](wiki.html#article=round) |
| Reward objective | `.options label:first-child` | [reward-ablations](wiki.html#article=reward-ablations) |
| Checkpoint download | `#save` | [checkpoints](wiki.html#article=checkpoints) |
| Generator in flow | `.flow .green` | [generator](wiki.html#article=generator) |
| Execution in flow | `.flow>span:nth-of-type(2) b` | [machine](wiki.html#article=machine) |
| Learner in flow | `.flow .blue` | [learner](wiki.html#article=learner) |
| Program column | `.table-head>span:nth-child(1)` | [program](wiki.html#article=program) |
| Output column | `.table-head>span:nth-child(2)` | [emissions](wiki.html#article=emissions) |
| Reward column | `.table-head>span:nth-child(3)` | [gradient-reward](wiki.html#article=gradient-reward) |
| Occupied archive niches | `#batch-info` | [archive](wiki.html#article=archive) |
| Training loss | `#loss` | [cross-entropy](wiki.html#article=cross-entropy) |
| Emitted output percentage | `#emission` | [emissions](wiki.html#article=emissions) |
| Training-loss graph | `#training-chart` | [diagnostics](wiki.html#article=diagnostics) |
| Held-out test heading | `.probes h2` | [transfer](wiki.html#article=transfer) |
| Probe selector instructions | `.probe-instructions a` | [probes](wiki.html#article=probes) |
| Repeat / Cycle / Count / Text / Noise buttons | `#probe-tabs` | [probes](wiki.html#article=probes) |
| Transfer plot | `#probe-chart` | [transfer](wiki.html#article=transfer) |
| Within-sequence comparison | `.context-stat` | [icl](wiki.html#article=icl) |
| Selected program | `#selected-code` | [program](wiki.html#article=program) |
| Selected program reward | `#selected-reward` | [gradient-reward](wiki.html#article=gradient-reward) |
| Selected execution statistics | `#program-meta` | [budgets](wiki.html#article=budgets) |
| Selected byte values | `#byte-values` | [teacher-forcing](wiki.html#article=teacher-forcing) |
| Prediction plot | `#prediction-chart` | [teacher-forcing](wiki.html#article=teacher-forcing) |
| Execution viewer | `#execution summary` | [machine](wiki.html#article=machine) |
| Reward statistics | `#reward-chart` | [reward-statistics](wiki.html#article=reward-statistics) |
| Output percentage graph | `#emission-chart` | [emissions](wiki.html#article=emissions) |
| Program-length graph | `#length-chart` | [program-length](wiki.html#article=program-length) |
| Generator entropy graph | `#entropy-chart` | [exploration](wiki.html#article=exploration) |
| Archive coverage | `#archive-coverage` | [archive](wiki.html#article=archive) |
| Custom evaluation heading | `.custom-eval h2` | [probes](wiki.html#article=probes) |
| Custom sequence loss | `#custom-chart` | [bpb](wiki.html#article=bpb) |
| Program sandbox | `.sandbox summary` | [sandbox](wiki.html#article=sandbox) |
| Paper differences | `.under-lab>details:nth-child(2) summary` | [demo-differences](wiki.html#article=demo-differences) |
| Settings: Model width | `#cfg-dim` | [transformer](wiki.html#article=transformer) |
| Settings: Attention heads | `#cfg-heads` | [attention](wiki.html#article=attention) |
| Settings: Transformer layers | `#cfg-layers` | [transformer](wiki.html#article=transformer) |
| Settings: MLP width multiplier | `#cfg-ffMultiplier` | [transformer](wiki.html#article=transformer) |
| Settings: Output / learner context (bytes) | `#cfg-context` | [context](wiki.html#article=context) |
| Settings: Generator context (tokens) | `#cfg-programLength` | [program](wiki.html#article=program) |
| Settings: Programs per round | `#cfg-pool` | [pool](wiki.html#article=pool) |
| Settings: Primitive steps per program | `#cfg-steps` | [budgets](wiki.html#article=budgets) |
| Settings: Tape cells | `#cfg-memory` | [machine](wiki.html#article=machine) |
| Settings: Learner learning rate | `#cfg-lr` | [adam](wiki.html#article=adam) |
| Settings: Generator learning rate | `#cfg-generatorLR` | [policy-gradient](wiki.html#article=policy-gradient) |
| Settings: Prior KL strength β | `#cfg-beta` | [kl](wiki.html#article=kl) |
| Settings: Expert iteration weight | `#cfg-expert` | [expert-iteration](wiki.html#article=expert-iteration) |
| Settings: Replay bank capacity | `#cfg-bankSize` | [replay](wiki.html#article=replay) |
| Device training backend | `label[for="backend"]` | [gpu](wiki.html#article=gpu) |
| Each generated source program | `.program-code` | [program](wiki.html#article=program) |
| Each program’s reward | `.reward` | [gradient-reward](wiki.html#article=gradient-reward) |
| Each emitted byte | `.byte:not(.pad)` | [teacher-forcing](wiki.html#article=teacher-forcing) |
| Each padded byte | `.byte.pad` | [padding](wiki.html#article=padding) |
| Fresh / replay / mutation source | `.source-label` | [pool](wiki.html#article=pool) |

## Paper and article map

### 01 · The central idea

- **What is self-play pretraining?** (overview); paper S1; links: round, tabula-rasa, experiments, ansatz, limitations, generator, machine, learner, gradient-reward, random-tape, transfer, padding, gradients, preconditioner, history; shortest UI route: overview
- **What does “zero data” mean?** (tabula-rasa); paper S6.SS0.SSS0.Px2; links: overview, tuning, contingent, sandbox, program, macros, universal-structure, probes; shortest UI route: overview → tabula-rasa
- **One complete self-play round** (round); paper S2; links: pool, learner, gradient-reward, policy-gradient, zero-reward, random-tape, padding, expert-iteration, adam, probes, history; shortest UI route: round
- **The generator** (generator); paper S2; links: autoregressive, program, policy-gradient, expert-iteration, prior-control, transformer, brainfuck, macros, termination, learner, gradient-reward; shortest UI route: generator
- **The learner** (learner); paper S2; links: cross-entropy, autoregressive, teacher-forcing, transfer, icl, gradients; shortest UI route: learner

### 02 · Programs & execution

- **A program you can execute** (program); paper A5; links: machine, brainfuck, macros, random-tape, budgets, padding; shortest UI route: program
- **The universal machine and its limits** (machine); paper S2.SS1; links: brainfuck, budgets, solomonoff, program; shortest UI route: machine
- **Eight primitive instructions** (brainfuck); paper A5; links: program, machine, macros, budgets, random-tape, padding; shortest UI route: generator → brainfuck
- **The ten macro instructions** (macros); paper A5; links: brainfuck, prior-control, budgets, program, tabula-rasa; shortest UI route: generator → macros
- **Random input and reproducible execution** (random-tape); paper A5; links: noise-trap, program, replay, seed; shortest UI route: overview → random-tape
- **Emitted bytes versus zero padding** (padding); paper A5; links: cross-entropy, emissions, transfer, zero-reward, gradient-reward; shortest UI route: padding
- **Time, memory and context budgets** (budgets); paper A5; links: machine, context, demo-differences, program-length, padding; shortest UI route: budgets
- **Termination, length caps and likelihood** (termination); paper S2.SS2; links: prior, termination-prior, importance, program-length, kl, policy-gradient, expert-iteration; shortest UI route: generator → termination
- **Why the prior prefers short programs** (termination-prior); paper S2.SS2; links: prior, solomonoff, termination; shortest UI route: program-length → termination-prior

### 03 · The learner

- **Autoregressive factorization** (autoregressive); paper S2.SS3; links: transformer, teacher-forcing, cross-entropy, importance; shortest UI route: generator → autoregressive
- **What the transformer learns** (transformer); paper S2.SS3; links: attention, context, gradients, demo-differences; shortest UI route: transformer
- **Causal attention** (attention); paper S2.SS3; links: transformer, teacher-forcing, icl, validation; shortest UI route: attention
- **Context length and within-sequence adaptation** (context); paper S3.SS2; links: icl, icl-tasks, datasets, demo-differences; shortest UI route: context
- **What the displayed predictions receive** (teacher-forcing); paper S2.SS2; links: autoregressive, cross-entropy, bpb, learner; shortest UI route: teacher-forcing
- **Cross-entropy: the learner’s objective** (cross-entropy); paper S2.SS2; links: bpb, gradients, teacher-forcing, gradient-reward, transfer; shortest UI route: cross-entropy
- **Bits per byte · lower is better** (bpb); paper S3.SS1; links: cross-entropy, entropy, prior-control, transfer; shortest UI route: bpb
- **Entropy, surprise, and learnable structure** (entropy); paper S5.SS0.SSS0.Px4; links: cross-entropy, exploration, noise-trap, epiplexity; shortest UI route: exploration → entropy

### 04 · The learning-progress reward

- **A gradient is a local direction** (gradients); paper S2.SS2; links: gradient-reward, directional-derivative, preconditioner, adam; shortest UI route: overview → gradients
- **The actual gradient-alignment reward** (gradient-reward); paper S2.SS2; links: gradients, preconditioner, history, absolute-value, reward-example, advantage, reward-ablations, padding, kl; shortest UI route: gradient-reward
- **A worked alignment calculation** (reward-example); paper S2.SS2; links: gradient-reward, absolute-value, preconditioner, directional-derivative, policy-gradient; shortest UI route: gradient-reward → reward-example
- **Why Adam changes the inner product** (preconditioner); paper S2.SS2; links: adam, gradient-reward, directional-derivative, history, gradients, reward-ablations; shortest UI route: overview → preconditioner
- **AdamW and bias-corrected moments** (adam); paper S2.SS2; links: preconditioner, gradients, zero-reward, demo-differences; shortest UI route: adam
- **The growing lookback window** (history); paper S2.SS2; links: shadow-history, preconditioner, absolute-value, reward-ablations; shortest UI route: overview → history
- **Why the absolute value matters** (absolute-value); paper A6; links: gradient-reward, reward-example, reward-ablations, noise-trap; shortest UI route: gradient-reward → absolute-value
- **The reward as a directional derivative** (directional-derivative); paper S2.SS2; links: gradients, gradient-reward, validation, stop-gradient; shortest UI route: reward-ablations → directional-derivative
- **What “the learner’s frontier” means** (frontier); paper S2.SS2; links: gradient-reward, noise-trap, reward-statistics, archive, transfer, prior-control, reward-ablations; shortest UI route: reward-ablations → noise-trap → frontier
- **Why difficulty alone can fail** (noise-trap); paper S2.SS2; links: gradient-reward, entropy, frontier, reward-ablations, cross-entropy; shortest UI route: reward-ablations → noise-trap
- **Why the first round has zero reward** (zero-reward); paper S2.SS2; links: history, advantage, expert-iteration, padding, kl, reward-statistics; shortest UI route: round → zero-reward

### 05 · Teaching the generator

- **The program pool** (pool); paper S2.SS2; links: round, mutation, replay, archive, advantage, generator, policy-gradient, expert-iteration; shortest UI route: pool
- **Local program mutation** (mutation); paper A7; links: archive, pool, expert-iteration, importance, policy-gradient, budgets; shortest UI route: archive → mutation
- **Archive niches: preserving different program behaviors** (archive); paper A7; links: mutation, replay, program-length, exploration; shortest UI route: archive
- **Replay and the original proposal probability** (replay); paper A7; links: importance, expert-iteration, archive, pool, random-tape; shortest UI route: replay
- **How the generator learns from reward** (policy-gradient); paper S2.SS2; links: advantage, importance, stop-gradient, expert-iteration, kl, autoregressive; shortest UI route: policy-gradient
- **From raw rewards to advantages** (advantage); paper S2.SS2; links: reward-statistics, kl, policy-gradient, expert-iteration, zero-reward, pool; shortest UI route: gradient-reward → advantage
- **The uniform program prior** (prior); paper S2.SS2; links: prior-control, termination-prior, kl, solomonoff; shortest UI route: prior-control → prior
- **KL regularization toward the prior** (kl); paper S2.SS2; links: prior, advantage, exploration, termination; shortest UI route: kl
- **Why replay needs importance ratios** (importance); paper S2.SS2; links: replay, policy-gradient, stop-gradient, mutation, autoregressive, expert-iteration; shortest UI route: checkpoints → importance
- **What stop-gradient holds fixed** (stop-gradient); paper S2.SS2; links: policy-gradient, importance, directional-derivative; shortest UI route: policy-gradient → stop-gradient
- **Reward-weighted expert iteration** (expert-iteration); paper S2.SS2; links: policy-gradient, mutation, zero-reward, pool, cross-entropy; shortest UI route: expert-iteration

### 06 · Evidence & evaluation

- **What evidence would support the method?** (experiments); paper S3; links: scaling, generator-quality, icl-tasks, discovery, reward-ablations, datasets, limitations, prior-control, pcfg, probes, diagnostics, tuning; shortest UI route: overview → experiments
- **Probes are held-out test sequences** (probes); paper S3; links: transfer, prior-control, bpb, datasets, sandbox; shortest UI route: probes
- **What it means for learning to transfer** (transfer); paper S3.SS1; links: probes, prior-control, datasets, icl, compute-frontier; shortest UI route: transfer
- **What the fixed-prior control actually does** (prior-control); paper S3.SS1; links: prior, bpb, transfer, pool, reward-ablations, brainfuck, macros, padding, gradient-reward; shortest UI route: prior-control
- **The random-grammar baseline** (pcfg); paper A8; links: prior-control, datasets, related-work, experiments, program; shortest UI route: overview → experiments → pcfg
- **Reading a scaling law** (scaling); paper S3.SS1; links: compute-frontier, ensemble, tuning, ansatz, exponents; shortest UI route: overview → experiments → scaling
- **Compute-optimal frontiers** (compute-frontier); paper S3.SS1; links: scaling, ensemble, tuning, prior-control, shadow-history; shortest UI route: transfer → compute-frontier
- **Ensembling predictive distributions** (ensemble); paper S3.SS1; links: scaling, compute-frontier, seed, cross-entropy; shortest UI route: transfer → compute-frontier → ensemble
- **Hyperparameter selection and leakage** (tuning); paper S3.SS1; links: tabula-rasa, probes, compute-frontier, pre-pretraining, seed; shortest UI route: seed → tuning
- **Structure relative to a bounded learner** (epiplexity); paper S5.SS0.SSS0.Px4; links: generator-quality, entropy, solomonoff, diagnostics; shortest UI route: machine → solomonoff → epiplexity
- **Does the generator itself improve?** (generator-quality); paper S3.SS1; links: epiplexity, experiments, transfer, diagnostics, exploration, program-length, archive; shortest UI route: overview → experiments → generator-quality
- **In-context learning without weight updates** (icl); paper S3.SS2; links: icl-tasks, sum-strategies, context, attention; shortest UI route: icl
- **The paper’s in-context task suite** (icl-tasks); paper A4; links: icl, sum-strategies, probes, attention; shortest UI route: icl → icl-tasks
- **How a SUM prediction can change** (sum-strategies); paper S3.SS2; links: icl-tasks, entropy, limitations, bpb; shortest UI route: icl → sum-strategies
- **Recognizable mathematical output** (discovery); paper A3; links: random-discovery, machine, program-length, limitations; shortest UI route: overview → experiments → discovery
- **Comparing discovery with random search** (random-discovery); paper A3; links: discovery, prior-control, limitations, transfer; shortest UI route: overview → experiments → discovery → random-discovery
- **What the reward ablations test** (reward-ablations); paper A6; links: absolute-value, history, directional-derivative, noise-trap, prior-control; shortest UI route: reward-ablations
- **Why encode different domains as bytes?** (datasets); paper A2; links: text-data, image-data, audio-data, music-data, dna-data, code-math-data, cross-entropy, context, scaling; shortest UI route: transfer → datasets
- **Natural text as UTF-8** (text-data); paper A2.SS1; links: datasets, contingent, probes, transfer, ansatz; shortest UI route: transfer → datasets → text-data
- **Images as raw pixel streams** (image-data); paper A2.SS2; links: datasets, context, transfer; shortest UI route: transfer → datasets → image-data
- **Audio, sampling rates and context** (audio-data); paper A2.SS3; links: datasets, context, music-data, scaling; shortest UI route: transfer → datasets → audio-data
- **Symbolic melody rather than raw sound** (music-data); paper A2.SS4; links: datasets, audio-data, context; shortest UI route: transfer → datasets → music-data
- **DNA and restricted symbol inventories** (dna-data); paper A2.SS5; links: datasets, bpb, entropy, transfer; shortest UI route: transfer → datasets → dna-data
- **Formal proofs and source code as data** (code-math-data); paper A2.SS6; links: datasets, program, learner, transfer, bpb; shortest UI route: transfer → datasets → code-math-data
- **Self-play as a warm start** (pre-pretraining); paper A1.SS1; links: tabula-rasa, tuning, contingent, experiments; shortest UI route: seed → tuning → pre-pretraining

### 07 · Universal data & scaling

- **Reusable predictive structure** (universal-structure); paper S4; links: contingent, ansatz, solomonoff, epiplexity; shortest UI route: overview → tabula-rasa → universal-structure
- **Information about this particular world** (contingent); paper S6; links: universal-structure, ansatz, context, pre-pretraining, transfer; shortest UI route: overview → tabula-rasa → contingent
- **Universal prediction as motivation** (solomonoff); paper S5.SS0.SSS0.Px3; links: prior, termination-prior, universal-structure, epiplexity; shortest UI route: machine → solomonoff
- **The universal-data scaling hypothesis** (ansatz); paper S4; links: universal-structure, contingent, exponents, scaling, limitations; shortest UI route: overview → ansatz
- **Why similar scaling exponents are suggestive** (exponents); paper S4; links: ansatz, scaling, compute-frontier, limitations; shortest UI route: overview → ansatz → exponents
- **How this connects to other approaches** (related-work); paper S5; links: solomonoff, tabula-rasa, noise-trap, epiplexity, gradient-reward; shortest UI route: overview → experiments → pcfg → related-work
- **What this experiment does not establish** (limitations); paper S6; links: experiments, tuning, probes, validation, ansatz, padding, archive, gradient-reward; shortest UI route: overview → limitations

### 08 · This browser experiment

- **What the optional diagnostics measure** (diagnostics); paper S3; links: reward-statistics, program-length, emissions, exploration, archive; shortest UI route: diagnostics
- **Reward mean and standard deviation** (reward-statistics); paper S2.SS2; links: advantage, gradient-reward, diagnostics, zero-reward; shortest UI route: reward-statistics
- **Source length is not complexity** (program-length); paper A7; links: termination-prior, archive, kl, diagnostics, exploration, transfer; shortest UI route: program-length
- **How much of the row was actually printed?** (emissions); paper A5; links: padding, program, noise-trap, diagnostics, probes; shortest UI route: emissions
- **Inspecting generator exploration** (exploration); paper A7; links: entropy, archive, mutation, kl, diagnostics, transfer, program-length, replay; shortest UI route: exploration
- **Exact historical checkpoints with bounded growth** (shadow-history); paper S2.SS2; links: history, checkpoints, seed, validation; shortest UI route: checkpoints → shadow-history
- **Seeds and reproducibility** (seed); paper S3; links: tabula-rasa, tuning, random-tape, checkpoints; shortest UI route: seed
- **What a saved run contains** (checkpoints); paper S2; links: shadow-history, adam, importance, diagnostics, preconditioner; shortest UI route: checkpoints
- **The program sandbox is evaluation only** (sandbox); paper A5; links: program, probes, teacher-forcing, tabula-rasa, brainfuck; shortest UI route: sandbox
- **Browser scale versus paper scale** (demo-differences); paper S2.SS3; links: transformer, budgets, probes, validation, limitations, gpu, prior-control; shortest UI route: demo-differences
- **How the implementation is checked** (validation); paper S2; links: directional-derivative, shadow-history, program, limitations, transfer, scaling; shortest UI route: demo-differences → validation
- **GPU and CPU execution** (gpu); paper S2.SS3; links: transformer, gradient-reward, shadow-history, checkpoints, validation, demo-differences; shortest UI route: gpu
