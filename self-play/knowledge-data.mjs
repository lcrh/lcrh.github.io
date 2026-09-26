// Original explanatory companion. [[id|words]] links are shared by wiki and tooltips.
// Mathematics is TeX; String.raw preserves its backslashes. No prose enters training.
export const CHAPTERS=[
 {id:'idea',title:'01 · The central idea',source:'S1'},
 {id:'programs',title:'02 · Programs & execution',source:'A5'},
 {id:'learner',title:'03 · The learner',source:'S2.SS3'},
 {id:'reward',title:'04 · The learning-progress reward',source:'S2.SS2'},
 {id:'generator',title:'05 · Teaching the generator',source:'S2.SS2'},
 {id:'experiments',title:'06 · Evidence & evaluation',source:'S3'},
 {id:'theory',title:'07 · Universal data & scaling',source:'S4'},
 {id:'practice',title:'08 · This browser experiment',source:'S6'}
];
export const ARTICLES={};
function a(id,title,chapter,summary,body,related,source,kind='Paper companion'){
 ARTICLES[id]={id,title,chapter,summary,body,related:related.split(' '),source:source||CHAPTERS.find(c=>c.id===chapter).source,kind};
}
a('overview','What is self-play pretraining?','idea','Learn a source of training data at the same time as a predictor of that data.',String.raw`
## Two learners, different jobs
The [[generator]] writes a program. The [[machine]] executes it. The [[learner]] predicts the resulting bytes. Prediction errors update the learner; a [[gradient-reward|gradient-based reward]] updates the generator. The next batch therefore depends on what the learner has already become good at.

This is a search over data-generating procedures, rather than a menu of hand-designed tasks. A looping program can express a long sequence compactly. [[random-tape|Random input]] lets one program describe a distribution instead of one fixed example.

## Three separate questions
Does the learner fit the current output? Does the generator find useful new structure? Does that structure [[transfer]] to data outside training? These require different measurements. Low training loss only answers the first question, and [[padding]] can make it easy.

Start with [[round|one complete round]], then follow the reward through [[gradients]], [[preconditioner]], and [[history]]. For the broader argument, explore [[experiments]], [[ansatz]], and [[limitations]].`, 'round tabula-rasa experiments ansatz limitations','S1');
a('tabula-rasa','What does “zero data” mean?','idea','No natural training corpus and no pretrained weights; it does not mean no inductive bias.',String.raw`
Both neural networks begin with random parameters. Training examples are created by [[program|programs]] during the run. The language, [[macros]], architecture, optimizer, resource limits and reward are still human choices. Zero natural data is a statement about the training signal, not an absence of assumptions.

Imagine learning the rule “repeat the last symbol” without ever reading English. That rule might help with repeated characters later. It cannot tell you the name of a city you have never observed. This distinguishes [[universal-structure|reusable structure]] from [[contingent|world-specific information]].

The paper uses this setup to make a scientific attribution cleaner: structure acquired from natural pretraining cannot explain the result. But it still uses natural validation data for [[tuning|hyperparameter selection]]. That qualification matters when interpreting the phrase “zero data.”

This playground begins from scratch too. The editable example is an [[sandbox|evaluation-only sandbox]], and neither it nor the [[probes]] is admitted to training.`, 'overview tuning contingent sandbox','S6.SS0.SSS0.Px2');
a('round','One complete self-play round','idea','Sample programs, execute them, measure gradients, then update both networks.',String.raw`
## Follow the data
First construct a [[pool]] of fresh samples, mutations and replay. Execute each program with a fresh [[random-tape|input stream]] to obtain output \(y_i\). Every output row trains the learner, including [[padding]].

\[\mathcal L_{\rm learner}=\frac1M\sum_{i=1}^{M}\mathcal L(y_i;\theta).\]

At the current learner weights, calculate each row's gradient and [[gradient-reward|alignment reward]]. The generator's objective combines [[policy-gradient]] with [[expert-iteration]]. Then the learner takes an [[adam|AdamW step]] on the pool-average gradient.

## Before versus after
A displayed training-row loss and reward describe the model **before** that round's learner update. A scheduled [[probes|evaluation]] describes the learner **after** its completed updates. They need not be equal, even for the same bytes.

The historical index \(e\) counts completed updates. At the first update, \(e=0\), so the [[history|lookback difference]] is zero. The [[zero-reward|startup behavior]] is a real consequence of the formula.`, 'pool learner gradient-reward policy-gradient zero-reward','S2');
a('generator','The generator','idea','An autoregressive neural policy over instructions, not a catalog of sequence families.',String.raw`
A program is sampled one instruction at a time. Each choice depends on the preceding instructions through a [[transformer]]. If the token sequence is \(x_1,\ldots,x_\ell\), then

\[g_\phi(x)=\prod_{t=1}^{\ell}g_\phi(x_t\mid x_{<t}).\]

The alphabet contains eight [[brainfuck|primitive operations]], ten [[macros]], and [[termination|F]]. The network can change which prefixes, loops and instruction combinations it prefers. It does not directly choose “a Fibonacci example.”

The [[learner]] consumes only the executed output. The generator instead receives a scalar [[gradient-reward|reward]] through [[policy-gradient]] and learns to imitate valuable programs through [[expert-iteration]]. No gradient is propagated through the interpreter.

The [[prior-control]] removes this learned program policy. That makes it a useful comparison: both learners receive outputs from the same executable language, but only self-play adapts the source of examples.`, 'autoregressive program policy-gradient expert-iteration prior-control','S2');
a('learner','The learner','idea','A neural next-byte predictor trained on program outputs, without seeing their code.',String.raw`
Given a prefix of an output stream, the learner produces 256 probabilities. The true next byte supplies the target. [[cross-entropy]] penalizes assigning that byte little probability, and [[gradients]] update all trainable parameters.

\[\pi_\theta(y)=\prod_{t=1}^{T}\pi_\theta(y_t\mid y_{<t}).\]

A model that has learned copying or periodicity might exploit those patterns in unfamiliar data. Whether it does so is a [[transfer|testable question]], not something guaranteed by the objective.

The program is hidden from the learner. Otherwise the task could become interpreter emulation conditioned on source code, which is different from learning to predict arbitrary byte streams.

The live forecasts use [[teacher-forcing]]: each prediction sees the true earlier bytes. Training changes network weights across rounds; [[icl|in-context learning]] changes predictions within a sequence without a weight update.`, 'cross-entropy autoregressive teacher-forcing transfer icl','S2');
a('program','A program you can execute','programs','A sequence of instructions whose real execution supplies the learner’s bytes.',String.raw`
The string in a curriculum row is actual source code. Its loops, reads and writes run on a [[machine|bounded byte-tape machine]]. A small source string can print many bytes; source length and output length are different quantities.

F means stop. A program consisting only of F emits no bytes, but its fixed-length training row still contains zeros in every position. The learner therefore receives a real [[padding|constant-zero prediction task]] from it. The curriculum is sorted by reward by default and also contains replay and mutations; seeing F near the top does not measure how often the generator samples F.

## Read an example
The hand-written illustration +[.++]F starts at zero, increments to one, then repeatedly prints and adds two. With byte wraparound it emits 1, 3, 5, … . This example explains the interpreter; it is not inserted into the training pool.

## Watch this exact execution
On a live row, the execution viewer starts with the same zeroed tape and captured random-input state as that row. Each highlighted instruction is a primitive operation; a [[macros|macro]] may expand into many of them. Output appears only when a print instruction actually executes. On termination, unfilled positions become [[padding]].

A saved hover is a snapshot. Training can continue while you inspect it. The simulator is an observer, never another training source.`, 'machine brainfuck macros random-tape budgets','A5');
a('machine','The universal machine and its limits','programs','An expressive abstract language, executed under finite time, memory and output budgets.',String.raw`
A machine maps a program and input stream to output:

\[y=U(x,\omega)\in\{0,\ldots,255\}^{T}.\]

Its state consists of a program counter, a tape of cells, a pointer into that tape, and an output stream. [[brainfuck|Primitive operations]] modify this state. The input \(\omega\) is a fresh random stream, so the same source can produce different examples.

Universality concerns the language when adequate resources are available. A browser run with 256 cells and a step limit is **not** an unbounded Turing machine. Its finite execution approximates a tiny portion of that space. Increasing [[budgets]] expands the reachable behavior but does not solve the difficulty of searching for useful programs.

All strings have defined execution semantics. Unmatched brackets do nothing; pointers wrap on the circular tape; byte arithmetic wraps modulo 256. Consequently the generator does not need a syntax-validity filter to receive training feedback.`, 'brainfuck budgets solomonoff program','S2.SS1');
a('brainfuck','Eight primitive instructions','programs','Move, change a byte, loop, read, and print: a small set with expressive composition.',String.raw`
< and > move the pointer one cell. + and − add or subtract one modulo 256. A dot prints the current cell. A comma replaces it with the next [[random-tape|random input byte]]. Brackets form a loop that continues while the current cell is nonzero.

## Loops are computation
For +++[>+.<-]F, the first cell counts down from three while the second cell counts up. The emitted bytes are 1, 2, 3; the rest of the fixed output row is [[padding]]. The output does not come from a shortcut that recognizes this pattern: every move, increment and branch executes.

A matched opening bracket skips its loop when the current cell is zero. A matched closing bracket returns to the body when the cell is nonzero. Unmatched brackets are no-ops in this dialect.

[[macros]] are compact spellings of longer primitive sequences. The simulator expands them, making the execution budget and intermediate tape changes visible.`, 'program machine macros budgets','A5');
a('macros','The ten macro instructions','programs','Short spellings of ordinary Brainfuck sequences, with no extra semantic oracle.',String.raw`
Z expands to [-], clearing the current cell. R is [->+<], moving its value into the right neighbor by addition. L uses [->+++<], adding three times the value to the right. N is [-<->], subtracting the value from the left neighbor. C is [->+>+<<], distributing it by addition to the next two cells. These operations clear the source cell.

G = [>] and H = [<] scan for a zero. W = [[-]>+<] clears a nonzero current cell and increments its right neighbor. V = [.>] prints cells until reaching zero. X = [-]++++++++++++++++ sets the current cell to 16.

## A useful bias, not a new task family
Macros make certain building blocks cheaper to express. That is an [[tabula-rasa|inductive bias]], even though all expansions use the same primitive language. The [[prior-control|control]] uses exactly the same augmented alphabet, so a comparison does not credit self-play merely for having macros.

Execution counts the expanded primitive steps. A one-token macro can therefore consume substantial time or run until a [[budgets|budget limit]].`, 'brainfuck prior-control budgets program','A5');
a('random-tape','Random input and reproducible execution','programs','A comma reads a new uniform byte; the same program can represent a distribution.',String.raw`
A fixed source need not always print the same sequence. For example, ,[.]F reads one random byte and, unless it is zero, repeatedly prints that value. The learner could practice repetition across many values by seeing different executions of one program.

The formal input stream \(\omega\) consists of independent uniform bytes. The browser uses a deterministic pseudorandom generator, seeded per run, to approximate that stream. [[replay]] re-executes old source with fresh input rather than memorizing its previous output.

## Exact execution replay
Each displayed row records the random-generator state immediately before execution. The execution viewer owns a separate copy initialized with that state. It therefore reproduces the captured row without consuming the training RNG or changing future samples.

A program that prints fresh random input repeatedly can generate hard-to-predict data. That is the [[noise-trap|difficulty reward's failure case]]; randomness and useful structure are different.`, 'noise-trap program replay seed','A5');
a('padding','Emitted bytes versus zero padding','programs','The learner trains on a fixed-length row, even when the program prints almost nothing.',String.raw`
A program may halt after three emissions while the output budget is 64. Its training row contains those three bytes followed by 61 zeros. A zero that was explicitly printed is an emitted byte; an unfilled slot is padding. Their numeric values may be identical, but their origins differ.

This follows the paper's explicit zero-padding convention in Appendix E. F alone prints nothing, yet trains on a full row of zeros. By contrast, .F prints one genuine zero and pads the remaining positions. Both rows contain the same values, so they have identical learner gradients and [[gradient-reward|rewards]] when evaluated in the same round. The learner does not see where padding begins.

\[L_{\rm batch}=qL_{\rm emitted}+(1-q)L_{\rm padding},\]

where \(q\) is the emitted fraction, using token-weighted losses. If every row has the same total length, this agrees with averaging the row means.

## Why the distinction matters
Suppose emitted-byte loss stays at 8 bits while padding loss becomes 0.1 bits, and only 10% of tokens are emitted. Total loss is then about 0.89 bits. That impressive-looking number can coexist with almost no ability to predict informative outputs.

The live chart separates the two portions. The [[transfer]] panel evaluates independent sequences without this training-row padding convention. Never infer broad learning from the main loss number alone.`, 'cross-entropy emissions transfer zero-reward','A5');
a('budgets','Time, memory and context budgets','programs','Resource limits define the actual search space available in this run.',String.raw`
Quick mode allows 48 source tokens, 4,096 primitive execution steps, 256 circular tape cells, and 64 output bytes. Long-run mode allows 96 source tokens, 16,384 steps and 128 output bytes, with the same tape size.

Execution stops when the program halts, the primitive-step budget expires, or the output row fills. Reaching a tape boundary wraps the pointer instead of stopping execution. A program can contain an infinite loop mathematically and still finish its browser evaluation because the step budget expires. Remaining output becomes [[padding]].

A model's [[context|context length]] is a separate resource from the interpreter's memory. Increasing tape size helps programs compute; increasing neural context helps the learner use more preceding bytes. Increasing either can slow the experiment.

These limits bound every simulator run as well. The execution viewer follows the same stopping rules as training, even when its animation is slowed down. Changing playback speed changes presentation time, not the number or meaning of machine steps.`, 'machine context demo-differences program-length','A5','Browser implementation');
a('termination','Termination, length caps and likelihood','programs','F is sampled like any other instruction; a hard length cap is a separate stopping rule.',String.raw`
The generator stops when it samples F or reaches the maximum source length. A sampled F contributes to the sequence log probability, because choosing to stop was a policy action. A maximum-length cutoff is imposed by the environment and is not an extra sampled token.

With 19 equiprobable tokens, the probability of a particular terminated string of length \(\ell\) is \(19^{-\ell}\). The unconstrained expected token count including F is 19. A browser cap truncates that distribution.

The [[kl|prior penalty]] must use the same convention as the generator likelihood. Treating a forced terminator as if the model had sampled it would distort [[importance|replay ratios]] and policy gradients.

A mutation is an edit, not a sample from this autoregressive proposal. Its probability is left undefined for [[policy-gradient|policy-gradient correction]], while [[expert-iteration]] can still learn from it.`, 'prior termination-prior importance program-length','S2.SS2');
// Separate alias article gives the length distribution a mathematical landing point.
a('termination-prior','Why the prior prefers short programs','programs','Short individual descriptions receive more probability, while longer lengths contain more possible strings.',String.raw`
Let \(K=19\), with one token reserved for F. Under independent uniform token sampling, the probability that the first F appears at token \(\ell\) is

\[P(\ell)=\left(\frac{K-1}{K}\right)^{\ell-1}\frac1K.\]

A particular string of that length has probability \(K^{-\ell}\). These statements are consistent: there are \((K-1)^{\ell-1}\) possible nonterminating prefixes. Confusing string probability with length probability would badly overstate how concentrated the prior is on one-token programs.

The finite-length browser sampler additionally places the remaining probability on capped prefixes. The [[termination]] article explains their log-probability convention. This is a concrete program prior inspired by [[solomonoff|algorithmic probability]], not the literal binary universal mixture.`, 'prior solomonoff termination','S2.SS2','Mathematical background');
a('autoregressive','Autoregressive factorization','learner','Predict each next token using only its prefix, then multiply the conditional probabilities.',String.raw`
For a sequence \(z\), the chain rule gives

\[p(z_1,\ldots,z_T)=\prod_{t=1}^T p(z_t\mid z_{<t}),\qquad \log p(z)=\sum_{t=1}^T\log p(z_t\mid z_{<t}).\]

The generator applies this to instruction tokens; the learner applies it to output bytes. The same factorization supports both sampling and likelihood evaluation.

A [[transformer|causal attention mask]] prevents later tokens from leaking into earlier predictions. During [[teacher-forcing|teacher-forced training]], many positions can be processed together because their true prefixes are already known. During generation, later tokens do not exist until earlier ones have been sampled.

The sum of log probabilities is essential for the generator's sequence objective and [[importance|importance ratios]]. The learner uses a per-token mean so rows of equal length have comparable losses. Those two normalizations serve different purposes.`, 'transformer teacher-forcing cross-entropy importance','S2.SS3','Mathematical background');
a('transformer','What the transformer learns','learner','Trainable embeddings, causal attention, and nonlinear layers turn a prefix into next-token probabilities.',String.raw`
Tokens are embedded into vectors. [[attention]] mixes information from earlier positions. Residual connections and nonlinear feed-forward layers transform those representations, and an output projection plus softmax produces probabilities.

\[p_j=\frac{e^{z_j}}{\sum_k e^{z_k}}.\]

Every part of this browser model receives [[gradients]]: token and position embeddings, queries, keys, values, output projections, feed-forward weights, and normalization scales. It is not a frozen feature extractor with only a trainable classifier.

The paper uses decoder-only Llama transformers. The playground's presets use one much narrower layer with learned positions, RMS normalization and ReLU; advanced settings can change the depth. Those are [[demo-differences|substantial architecture reductions]]. The underlying causal prediction and self-play objectives remain actual computations.

A larger network has more capacity, but also fewer training rounds per minute on the same device. Capacity and optimization time should be distinguished when comparing presets.`, 'attention context gradients demo-differences','S2.SS3');
a('attention','Causal attention','learner','A learned weighted combination of earlier token representations.',String.raw`
A head constructs queries \(Q\), keys \(K\), and values \(V\). At position \(t\), attention can inspect only positions \(s\le t\):

\[a_{ts}=\operatorname{softmax}_{s\le t}\left(\frac{q_t^\top k_s}{\sqrt{d_h}}\right),\qquad o_t=\sum_{s\le t}a_{ts}v_s.\]

Queries and keys determine where to look; values carry the information being combined. The square-root factor controls score magnitude as head width changes. Different heads can learn different ways of selecting relevant earlier information.

Attention provides a route to copying, matching a key to a previous value, or recognizing repetition, but merely having attention does not prove that such algorithms were learned. [[icl|In-context tasks]] test the resulting behavior.

The independent test changes current and future targets and checks that earlier predicted distributions stay identical. That guards against accidental target leakage through the mask or input shift.`, 'transformer teacher-forcing icl validation','S2.SS3','Mathematical background');
a('context','Context length and within-sequence adaptation','learner','More visible history can help prediction, but position-dependent losses do not isolate its causal effect.',String.raw`
A predictor cannot use observations outside its input window. A 64-byte context may cover several short cycles, but almost no meaningful duration of high-rate raw audio. This is why [[datasets|data encoding]] affects what structure fits inside a context.

Within a sequence, weights stay fixed while the visible prefix grows. This differs from changing weights with an optimizer. If earlier examples help infer a new task, that behavior may be [[icl|in-context learning]].

The browser's first-versus-last-16-byte statistic is only descriptive. Later positions also contain different targets and position embeddings. A controlled test would keep the query task fixed and vary the number of relevant examples, as in the paper's [[icl-tasks]].

Long-run mode doubles this playground's output context to 128; the paper's experiments use far longer contexts. Matching a trend in this miniature setting does not reproduce the reported task abilities.`, 'icl icl-tasks datasets demo-differences','S3.SS2');
a('teacher-forcing','What the displayed predictions receive','learner','Each prediction sees the true preceding bytes; its own earlier guesses are not fed back.',String.raw`
Suppose the true output begins 3, 6, 9. To predict the third byte, the learner receives 3, 6 even if its guess for the second byte was wrong. The plotted predictions therefore measure one-step conditional prediction.

This is teacher forcing. It gives clean next-token training targets and lets [[cross-entropy]] score every position. Free-running generation instead feeds a model's sampled or greedy outputs back into itself, allowing early mistakes to alter all later inputs.

The green line in the inspector is the executed output; blue points are greedy choices from the learner's conditional probabilities. A wrong greedy guess can still assign the true byte substantial probability. The [[bpb|loss]] captures that nuance; exact-match accuracy does not.

Neither prediction mode would justify giving source code to this learner. Its task is to infer output regularities from the bytes themselves.`, 'autoregressive cross-entropy bpb learner','S2.SS2','Mathematical background');
a('cross-entropy','Cross-entropy: the learner’s objective','learner','The mean negative log probability assigned to the actual next bytes.',String.raw`
For output \(y\) of length \(T\),

\[\mathcal L(y;\theta)=-\frac1T\sum_{t=1}^T\ln\pi_\theta(y_t\mid y_{<t}).\]

A confident correct prediction costs little; a confident wrong one costs a lot. Assigning the true byte probability 1/2 costs one [[bpb|bit]], 1/4 costs two, and 1/256 costs eight. The optimizer uses natural logarithms, while the display divides by \(\ln2\).

For logits with softmax probabilities, the local derivative is \(\partial(-\ln p_y)/\partial z_j=p_j-\mathbf1[j=y]\). Backpropagation carries these errors through every network layer.

The gradient describes how a small parameter change affects the current output's loss. It is not a claim that this output will improve prediction elsewhere. The [[gradient-reward]] tries to select useful examples using the learner's past trajectory; [[transfer|held-out evaluation]] tests what actually generalized.`, 'bpb gradients teacher-forcing gradient-reward','S2.SS2');
a('bpb','Bits per byte · lower is better','learner','A probability-based prediction cost, measured in binary information units.',String.raw`
\[\mathrm{BPB}=-\frac1T\sum_{t=1}^{T}\log_2 p(y_t\mid y_{<t})=\frac{\mathcal L}{\ln2}.\]

Uniform predictions over 256 possible bytes cost exactly 8 bits per byte. Better predictions assign more probability to the bytes that occur, reducing the cost. BPB can exceed 8 when a model confidently favors the wrong values.

One bit does not mean “one wrong bit in an eight-bit byte.” It is an average logarithmic cost. A model predicting each true next byte with probability 1/2 has a one-bit cost even if its most likely byte is sometimes wrong.

On the transfer plot, the horizontal axis is completed training rounds; the vertical axis is held-out BPB. A downward curve is improvement on that selected probe. The dotted 8-bit line is a uniform-probability reference, **not** the learned [[prior-control|fixed-prior learner]].`, 'cross-entropy entropy prior-control transfer','S3.SS1','Mathematical background');
a('entropy','Entropy, surprise, and learnable structure','learner','Uncertainty is not the same as useful information for training.',String.raw`
For a predictive distribution, entropy is

\[H(p)=-\sum_jp_j\log_2p_j.\]

Surprise measures one realized outcome: \(-\log_2p_y\). Entropy averages surprise under the model's own distribution. [[cross-entropy]] averages surprise under observed targets instead.

The diagnostics show generator entropy averaged over the prefix distributions of fresh sampled programs. Its maximum is \(\log_2 19\approx4.25\) bits per instruction. Low entropy indicates concentrated choices, but does not by itself prove harmful collapse; a competent policy can be selective.

A histogram of emitted byte values answers a different question again: how diverse the observed values are. A counting sequence and a random sequence can share a similar histogram while having very different predictability. [[epiplexity]] concerns structure extractable by a bounded learner, rather than raw uncertainty alone.`, 'cross-entropy exploration noise-trap epiplexity','S5.SS0.SSS0.Px4','Mathematical background');
a('gradients','A gradient is a local direction','reward','One coordinate per trainable parameter, recording how loss changes nearby.',String.raw`
For parameter vector \(\theta\), the gradient is \(g=\nabla_\theta\mathcal L\). For a small displacement \(v\),

\[\mathcal L(\theta+hv)\approx\mathcal L(\theta)+h\,g^\top v.\]

The dot product is a [[directional-derivative]]. Gradient descent moves against \(g\), while [[adam]] rescales and smooths the update. A large loss need not imply a large useful gradient; the model may be wrong in directions unrelated to previous learning.

For the paper's reward, each program gets its **own** full learner gradient, evaluated at the same current weights. Those gradients are then averaged for the learner's update. Reusing only a batch gradient for every program would erase the distinction the generator needs.

The hover breakdown groups actual coordinate products by parameter block. A positive block and a negative block can cancel. Summing absolute block contributions would compute a different reward from taking the absolute value of their total.`, 'gradient-reward directional-derivative preconditioner adam','S2.SS2','Mathematical background');
a('gradient-reward','The actual gradient-alignment reward','reward','How strongly this output’s gradient lines up with a preconditioned historical direction.',String.raw`
\[r_i=\left|\left\langle\nabla_\theta\mathcal L(y_i;\theta_e),\;P_e\odot(\theta_{\lfloor e/2\rfloor}-\theta_e)\right\rangle\right|.\]

Read this in pieces: compute the row's [[gradients|loss gradient]]; subtract current weights from the [[history|halfway checkpoint]]; multiply coordinatewise by the [[preconditioner|Adam step scale]]; take their inner product; finally take an [[absolute-value|absolute value]].

## Why use a historical direction?
Training can accumulate repeatable structure across many updates, while unrelated noise can cancel. The reward measures sensitivity along that accumulated direction. This is a heuristic for useful learning progress, not a proof that a rewarded program improves every downstream task.

A high reward can come from learning [[padding|constant zeros]] early in training. It does not certify a complicated program. Neither reward nor its decline is guaranteed to be monotonic: the current gradient, historical checkpoint and Adam scale all change, and the absolute value also rewards a large negative inner product.

## Read the live calculation
The hover diagnostic uses the selected row's actual pre-update gradient. Signed block contributions sum to the displayed inner product. Their positive and negative parts may be individually large; only the magnitude of the net sum is rewarded. [[advantage|Batch normalization]] and the [[kl|prior penalty]] then determine the policy weight.

Follow [[reward-example]] for a hand-worked numerical example. The [[reward-ablations]] explain why raw difficulty or one-step loss change is not interchangeable with this calculation.`, 'gradients preconditioner history absolute-value reward-example advantage reward-ablations','S2.SS2');
a('reward-example','A worked alignment calculation','reward','A two-parameter example showing signs, scaling, and cancellation.',String.raw`
Take a row gradient \(g=(2,-1)\), historical difference \(\delta=(0.3,0.4)\), and preconditioner \(P=(0.1,0.5)\). Then

\[d=P\odot\delta=(0.03,0.20),\qquad g^\top d=0.06-0.20=-0.14,\qquad r=0.14.\]

The reward is 0.14, not 0.26: cancellation occurs before the absolute value. It is also not the gradient norm, output loss, cosine similarity, or observed before/after loss reduction.

The normalized cosine is \(g^\top d/(\|g\|\|d\|)\). It reports directional agreement while removing magnitude. The actual reward retains magnitude, so two equally aligned examples can receive different rewards if one has a stronger gradient.

These are invented numbers for explanation. A hover over a training row instead shows that row's measured values, including its parameter-block decomposition and downstream [[policy-gradient|policy weight]].`, 'gradient-reward absolute-value preconditioner directional-derivative','S2.SS2','Worked example');
a('preconditioner','Why Adam changes the inner product','reward','Parameters with different optimizer scales contribute differently to the reward.',String.raw`
The diagonal scale is

\[P_{e,j}=\frac{\mathrm{lr}}{\sqrt{\hat v_{e,j}}+\epsilon},\qquad r_i=\left|\sum_j g_{i,j}P_{e,j}\delta\theta_{e,j}\right|.\]

\(\hat v\) is Adam's bias-corrected estimate of squared gradients. A coordinate with a larger second moment receives a smaller scale. This compensates for differences in the optimizer's effective step sizes; it does not mean all coordinates become equally important.

The historical difference already records actual parameter motion. Multiplying by \(P\) again is intentional: it is the paper's chosen reward metric, not a reconstruction of an ordinary optimizer step.

Think of a geometry in parameter space with coordinate weights. Using an unweighted dot product would ask a different question. The [[reward-ablations|empirical choice]] should not be confused with a general theorem that this is the only correct geometry.

Follow [[adam]] for the moment equations and [[directional-derivative]] for how the inner product can be calculated without materializing a whole gradient.`, 'adam gradient-reward directional-derivative history','S2.SS2');
a('adam','AdamW and bias-corrected moments','reward','Gradient smoothing, coordinatewise step scales, and decoupled weight decay.',String.raw`
Adam tracks first and second moments:

\[m_e=\beta_1m_{e-1}+(1-\beta_1)g_e,\quad v_e=\beta_2v_{e-1}+(1-\beta_2)g_e^2.\]

\[\hat m_e=\frac{m_e}{1-\beta_1^e},\quad\hat v_e=\frac{v_e}{1-\beta_2^e},\quad \theta\leftarrow\theta-\mathrm{lr}\left(\frac{\hat m_e}{\sqrt{\hat v_e}+\epsilon}+\lambda\theta\right).\]

Squaring, division and square roots are coordinatewise. Bias correction compensates for moments initialized at zero. AdamW applies weight decay separately from the gradient moments.

The browser clips the aggregate gradient norm before updating these moments. The reward, however, uses each row's unaveraged loss gradient and the optimizer's existing second moment **before** the new update. Its [[preconditioner]] uses \(\hat v\), not \(\hat m\).

A zero second moment would make the reciprocal preconditioning scale undefined without epsilon. The initial bias-correction denominator is also zero before any optimizer step. The implementation handles this [[zero-reward|startup]] explicitly by returning a zero reward direction, consistent with the zero initial historical difference.`, 'preconditioner gradients zero-reward demo-differences','S2.SS2','Mathematical background');
a('history','The growing lookback window','reward','Compare current weights with those halfway back through the run.',String.raw`
\[p(e)=\lfloor e/2\rfloor,\qquad\delta\theta_e=\theta_{p(e)}-\theta_e.\]

At round index 100 the reference is checkpoint 50; at index 1,000 it is checkpoint 500. The window grows, averaging over a longer learning trajectory while eventually leaving early events behind.

Notice the sign: **past minus present**. The final absolute value makes reversing this sign irrelevant to the canonical reward, but it would matter in a [[reward-ablations|signed-reward ablation]].

A one-step window would mostly measure immediate motion, which can be noisy or myopic. Replacing the long window with one step would change a loadbearing part of the method.

The browser reconstructs the exact required weights with [[shadow-history|slower deterministic copies]] rather than retaining all checkpoints. This is a memory-management choice; it does not approximate the horizon. The hover shows the reference index associated with the captured batch, not today's index after more training has occurred.`, 'shadow-history preconditioner absolute-value reward-ablations','S2.SS2');
a('absolute-value','Why the absolute value matters','reward','Both signs of a large directional derivative earn positive reward.',String.raw`
The reward takes \(|g^\top d|\). It therefore measures the magnitude of local sensitivity along the historical direction, not only gradients pointing in one signed direction.

This is an important limit of the friendly phrase “align with progress.” A large negative dot product can receive exactly the same reward as a large positive one. Saying the method rewards only beneficial first-order loss decreases would misdescribe it.

The historical direction is also preconditioned, rather than simply equal to a current proposed optimizer step. Consequently the score is not an exact estimate of realized loss improvement.

The paper compares alternative reward definitions empirically. The canonical choice is a successful experimental design choice in the reported setting, not a guarantee against every form of [[noise-trap|reward exploitation]]. The live signed-contribution bars make cancellations and signs visible rather than hiding them behind the positive scalar.`, 'gradient-reward reward-example reward-ablations noise-trap','A6');
a('directional-derivative','The reward as a directional derivative','reward','A dot product with a gradient is the first-order change along a chosen direction.',String.raw`
Set \(d=P_e\odot\delta\theta_e\). Holding \(d\) fixed during differentiation,

\[g^\top d=\left.\frac{d}{dh}\mathcal L(y;\theta_e+hd)\right|_{h=0}.\]

The paper uses forward-mode automatic differentiation to obtain this quantity without storing every coordinate of \(g\). The browser is small enough to form the full reverse-mode gradient and dot it with \(d\). These are two ways of evaluating the same local derivative.

A central finite difference approximates it with \([L(\theta+hd)-L(\theta-hd)]/(2h)\). That is useful as an independent numerical test, but is not the training reward's implementation here. Too large an \(h\) introduces curvature error; too small an \(h\) magnifies floating-point noise.

A measured loss decrease after an optimizer update also includes a different direction, finite step size, and interactions with other training rows. It is not interchangeable with this expression.`, 'gradients gradient-reward validation stop-gradient','S2.SS2','Mathematical background');
a('frontier','What “the learner’s frontier” means','reward','Examples should connect to learnable structure without being completely mastered.',String.raw`
An already-mastered output may produce little gradient. An unrelated random output can produce a large gradient that does not consistently follow a learned direction. Between these extremes are outputs whose structure connects to what the model is acquiring.

The [[gradient-reward]] is a mechanism intended to find that middle region. “Frontier” is an interpretation of its purpose, not a hidden difficulty label or a manually scheduled lesson level.

The live [[reward-statistics|reward distribution]] can change as the learner changes. A once-useful program may become unhelpful; [[archive|archive reward decay]] makes room for replacements. A high-reward row is not necessarily the hardest row or the one with the most diverse output.

The claim that such an adaptive source is useful requires comparisons, including a [[prior-control|nonadaptive source]], [[reward-ablations]], and [[transfer|evaluation on unseen data]]. One attractive program example is suggestive evidence of exploration, not a complete validation.`, 'gradient-reward noise-trap reward-statistics archive transfer','S2.SS2');
a('noise-trap','Why difficulty alone can fail','reward','Fresh random bytes can maximize surprise while offering little reusable predictive structure.',String.raw`
A program that repeatedly reads and prints random bytes can keep next-byte prediction difficult. For genuinely independent uniform bytes, the best possible expected next-byte cost is 8 bits, attained by uniform predictions. A learner favoring particular bytes can do much worse:

\[\mathbb E_{y\sim U}[-\log_2 p(y\mid c)]=8+D_{\rm KL}^{(2)}(U\|p(\cdot\mid c))\ge8.\]

Here \(U\) is uniform over 256 bytes and the divergence is measured in bits. This is an expectation over fresh random targets, not a requirement that every finite Noise probe score exactly eight. A fixed small sample can fluctuate above or below it; strong confidence in the wrong byte frequencies usually raises its loss.

Rewarding high [[cross-entropy|prediction loss]] can therefore encourage the generator to produce noise rather than a learnable curriculum. High [[entropy]] is not equivalent to valuable training structure.

The canonical reward tries to use the learner's historical motion to distinguish reusable signals from directions that cancel. This is a heuristic defense, not a theorem: finite batches and a finite model can still make noisy examples look rewarding.

Select **Difficulty · ablation** to replace the alignment score with the row's cross-entropy in a fresh run. Compare both output behavior and held-out probes. An isolated rise in generator reward under that different objective is not evidence of improvement.`, 'gradient-reward entropy frontier reward-ablations','S2.SS2');
a('zero-reward','Why the first round has zero reward','reward','At initialization, the past and present learner are the same model.',String.raw`
At index \(e=0\), the historical reference is also checkpoint zero, so \(\delta\theta_0=0\). Every gradient-alignment reward is zero regardless of how difficult the output is.

The learner still trains on the first randomly generated pool. Once it moves, the next round has a nonzero historical difference and can produce informative rewards. The generator also has its [[kl|prior regularizer]], whose value may be small near its near-uniform initialization.

Zero emitted bytes does not mean zero reward. F stops immediately, but its [[padding|full zero-padded row]] trains the learner. After the initial round, that simple task can have a large alignment reward while the learner is still acquiring it. This is different from the zero historical difference at startup.

The reward-standardization denominator has an epsilon. [[expert-iteration]] contributes zero when its positive-reward sum is zero; dividing blindly by that sum would create NaNs. These are numerical edge cases, not reasons to seed the run with hand-designed examples.

A pool of almost identical [[padding|zero-heavy outputs]] can also produce little reward variation later. Inspect the [[reward-statistics|standard deviation]] before interpreting large-looking standardized advantages.`, 'history advantage expert-iteration padding','S2.SS2');
a('pool','The program pool','generator','Fresh sampling, local edits, and replay contribute complementary training examples.',String.raw`
\[\mathcal B_e=\mathcal B_e^{\rm fresh}\mathbin{\dot\cup}\mathcal B_e^{\rm mut}\mathbin{\dot\cup}\mathcal B_e^{\rm replay}.\]

Fresh rows explore using the current [[generator]]. [[mutation|Mutated rows]] explore near useful existing programs. [[replay|Replayed rows]] retain earlier behaviors. Every row's executed output trains the learner, and rewards are normalized over the whole pool.

The default browser mixture is 6 fresh, 3 mutated and 3 replayed programs per 12-row batch; an empty source falls back to fresh sampling. You can inspect the source label on each row. Source labels describe how programs were obtained, not what kind of output they should produce.

Only rows with a valid sampled proposal probability enter [[policy-gradient]]. Mutations still enter [[expert-iteration]], including when a mutation-derived program is later replayed. These distinctions are necessary because the two objectives use different statistical assumptions.`, 'round mutation replay archive advantage','S2.SS2');
a('mutation','Local program mutation','generator','One insertion, deletion or substitution explores a neighbor of a useful program.',String.raw`
A mutation edits source tokens, then executes the result from a fresh zero tape. It does not edit the output or choose a desired mathematical sequence. A one-token edit can change a loop dramatically; it can also be irrelevant if execution never reaches that token.

Parents come from the [[archive|quality-diversity archive]]. Selecting a niche first prevents a populous kind of program from automatically monopolizing local exploration. Rewards are recomputed against the current learner rather than inherited from the parent.

The edited program was not sampled directly from the generator's autoregressive distribution. The implementation therefore excludes it from [[policy-gradient|policy-gradient updates]] rather than inventing a sampling probability. [[expert-iteration]] can still increase its likelihood according to its reward.

A mixture of local edits and fresh global proposals can explore differently from either alone. The finite browser [[budgets]] constrain both equally.`, 'archive pool expert-iteration importance','A7');
a('archive','Archive niches: preserving different program behaviors','generator','A niche is a bucket defined by loop depth and source length, not a discovered task label.',String.raw`
The archive partitions programs along two measured descriptors: maximum dynamically active loop depth, clamped to 0–8, and source-body length in buckets ≤8, 9–16, 17–32 and >32. In the browser, loop depth includes the loops inside expanded macros; body length counts source tokens before F. That gives \(9\times4=36\) possible niches.

## What “24 occupied niches” means
At least one retained program exists in 24 of those descriptor buckets. It does not mean 24 concepts were learned, 24 mathematical families discovered, or 24 neural subnetworks formed.

Within each niche, up to eight positively rewarded programs are retained. Stored scores decay by 0.97 each round, making old entries easier to replace as their relevance changes. A [[mutation]] parent is chosen by first selecting an occupied niche uniformly, then an entry within it.

This is a MAP-Elites-style quality-diversity archive: selection considers both measured quality and coverage of descriptor space. The [[replay|replay bank]] is different: a bounded history of non-replayed programs, sampled to preserve prior experience.`, 'mutation replay program-length exploration','A7');
a('replay','Replay and the original proposal probability','generator','Revisit source code from earlier rounds, but execute it with new random input.',String.raw`
The replay bank retains programs rather than a fixed output corpus. Re-execution with a fresh [[random-tape|input stream]] creates a new draw from the program's output distribution.

For an originally sampled program, the bank also stores its log probability under the generator that produced it. [[importance|Importance correction]] compares its present likelihood with that stored proposal likelihood. Overwriting the old value with today's likelihood would silently make every ratio one.

The browser samples bank entries without replacement and uses a bounded FIFO bank. Mutation-derived entries retain a missing proposal probability; they contribute to learner training and [[expert-iteration]], but not to policy gradients.

Replay does not ensure that an old example remains useful. Its reward is measured again against current learner state. The [[archive]] instead deliberately retains high-reward representatives within structural buckets for mutation.`, 'importance expert-iteration archive pool','A7');
a('policy-gradient','How the generator learns from reward','generator','Increase the log probability of favored sampled programs, with regularization and replay correction.',String.raw`
The score-function identity supplies the basic idea:

\[\nabla_\phi\mathbb E_{x\sim g_\phi}[r(x)]=\mathbb E[r(x)\nabla_\phi\log g_\phi(x)],\]

when the sampled reward is treated as fixed for this policy update. No derivative through discrete instruction sampling or the interpreter is required.

For eligible rows \(\mathcal E\), the implemented loss is

\[\mathcal L_{\rm PG}=-\frac1{|\mathcal E|}\sum_{i\in\mathcal E}\operatorname{sg}(\rho_i)\operatorname{sg}(A_i)\log g_\phi(x_i).\]

\(A_i\) is the [[advantage|normalized, regularized advantage]] and \(\rho_i\) the [[importance|proposal ratio]]. [[stop-gradient|Stop-gradient]] freezes those coefficients. Minimizing a negative weighted log likelihood raises probability for positive weights and lowers it for negative weights.

The total generator update also includes [[expert-iteration]]. In a row hover, the PG coefficient and EI coefficient are shown separately. A positive raw reward does not guarantee a positive PG coefficient: a below-average reward or the [[kl|prior penalty]] can reverse it.`, 'advantage importance stop-gradient expert-iteration kl','S2.SS2');
a('advantage','From raw rewards to advantages','generator','Compare each reward with its batch, then subtract a prior-deviation penalty.',String.raw`
\[A_i=\frac{r_i-\bar r}{\sigma_r+\epsilon}-\beta\left(\log g_\phi(x_i)-\log g_0(x_i)\right).\]

The mean and population standard deviation use the entire current [[pool]], including mutations. A reward of 2 is relatively weak in a batch centered at 5 and relatively strong in a batch centered at 0.1. Thus raw rewards cannot by themselves tell you the direction of policy learning.

Subtracting a baseline reduces policy-gradient variance; dividing by standard deviation controls scale. Epsilon handles a nearly constant batch. The second term is the [[kl|uniform-prior regularizer]].

The hover shows the selected row's reward relative to the actual batch mean and spread. It also shows the resulting [[policy-gradient|PG weight]], after any replay ratio and eligible-row normalization, and the independent [[expert-iteration|EI weight]]. The two terms can push differently.`, 'reward-statistics kl policy-gradient expert-iteration zero-reward','S2.SS2');
a('prior','The uniform program prior','generator','Independent uniform instruction choices define a nonadaptive source over the same language.',String.raw`
For a particular terminated program of \(\ell\) tokens over alphabet \(\mathcal A\),

\[g_0(x)=|\mathcal A|^{-\ell(x)},\qquad\log g_0(x)=-\ell(x)\log|\mathcal A|.\]

With 19 tokens, every next instruction—including F—is equally likely. This is a distribution over programs, **not** a uniform distribution over emitted bytes. Some programs print zeros, some repeat values, some run loops, and many print very little.

The prior has two roles: the generator's [[kl|regularizer]] refers to it, and the [[prior-control]] samples it directly. The latter still trains a real learner. Confusing that learner with an 8-bit uniform byte predictor obscures the experiment.

The [[termination-prior|length distribution]] favors shorter descriptions probabilistically. [[solomonoff|Universal prediction]] motivates searching this kind of space, but this bounded, augmented language is a practical choice with its own biases.`, 'prior-control termination-prior kl solomonoff','S2.SS2');
a('kl','KL regularization toward the prior','generator','Keep the learned policy from moving arbitrarily far from a simple reference distribution.',String.raw`
The intended reward-regularized objective is

\[J(\phi)=\mathbb E_{g_\phi}[r(x)]-\beta\,D_{\rm KL}(g_\phi\|g_0),\qquad D_{\rm KL}(p\|q)=\mathbb E_p\!\left[\log\frac pq\right].\]

The sampled log ratio \(\log g_\phi(x)-\log g_0(x)\) appears in the [[advantage]]. A program much more likely under the current generator than under the prior receives a larger penalty. A sampled log ratio can be negative even though the expectation defining KL is nonnegative.

The coefficient \(\beta\) trades reward-seeking against prior adherence. It does not impose a hard probability floor or guarantee diversity. Inspect [[exploration|generator entropy]], lengths and archive coverage separately.

Because sequence likelihood is a product of token probabilities, its log ratio is a sum across tokens. Using a token-average ratio here would alter the length dependence of the stated objective.`, 'prior advantage exploration termination','S2.SS2');
a('importance','Why replay needs importance ratios','generator','Correct for evaluating an old policy’s samples under a new policy.',String.raw`
\[\rho_i=\frac{g_\phi(x_i)}{g_{\phi_{\rm old},i}(x_i)}=\exp\!\left(\log g_\phi(x_i)-\log g_{\phi_{\rm old},i}(x_i)\right).\]

For a fresh sample scored under the policy that sampled it, the ratio is one, up to rounding. For replay, the denominator is the stored original proposal probability. A previously unlikely program that is now more likely receives a larger ratio.

The implementation clips the log ratio to ±20 before exponentiating. This bounds extreme numerical values but also makes the estimator biased when clipping is active. The ratio is a frozen multiplier during [[policy-gradient|policy backpropagation]].

A [[mutation|mutated program]] has no recorded autoregressive proposal probability, so this correction is not available. Its contribution comes through [[expert-iteration]] instead. The live diagnostic shows “not eligible” rather than manufacturing a ratio for such rows.`, 'replay policy-gradient stop-gradient mutation','S2.SS2');
a('stop-gradient','What stop-gradient holds fixed','generator','A quantity can depend on a model numerically without being differentiated in this update.',String.raw`
Consider \(-\rho A\log g_\phi(x)\). The intended policy-gradient term differentiates only the final log likelihood, treating the sampled reward, advantage and ratio as coefficients.

\[\nabla_\phi\mathcal L=-\operatorname{sg}(\rho A)\nabla_\phi\log g_\phi(x).\]

Differentiating through \(\rho\) and \(A\) as well would create extra terms and implement a different estimator. In this explicit JavaScript implementation, the coefficients are ordinary measured numbers passed to the generator's backward pass, so no derivative path through them exists.

Likewise, when interpreting the learner's [[directional-derivative|directional derivative]], the historical direction is held fixed. It is not a meta-gradient through the entire history of training. “Two models learn together” does not mean differentiating through all their past updates.`, 'policy-gradient importance directional-derivative','S2.SS2','Mathematical background');
a('expert-iteration','Reward-weighted expert iteration','generator','Imitate useful programs from the whole pool, including mutations.',String.raw`
\[w_i=\frac{[r_i]_+}{\sum_j[r_j]_+},\qquad\mathcal L_{\rm EI}=-\sum_iw_i\log g_\phi(x_i).\]

This is supervised likelihood training on rewarded source strings. Unlike [[policy-gradient]], it does not need a sampling-proposal correction to define its imitation objective, so it can include [[mutation|edited programs]].

The combined generator loss is \(\mathcal L_G=\mathcal L_{\rm PG}+\lambda_{\rm EI}\mathcal L_{\rm EI}\). A high-reward mutation can be incorporated into the neural policy even though it was not originally sampled from that policy.

The weights sum to one when the reward sum is positive. When all rewards are zero, the browser sets the term to zero. The normalization is at the program level; it multiplies summed sequence log probabilities, not a per-token mean.

This does not turn the learner into a source-code predictor: the learner still receives only output bytes. The source imitation applies exclusively to the generator.`, 'policy-gradient mutation zero-reward pool','S2.SS2');
a('experiments','What evidence would support the method?','experiments','Separate fitting generated data, improving its source, and transferring to unseen processes.',String.raw`
The paper asks several distinct questions. [[scaling|Scaling experiments]] evaluate held-out natural-data prediction across compute budgets. [[generator-quality|Frozen-generator corpus tests]] ask whether later generators make better training material for new learners. [[icl-tasks|In-context tasks]] ask whether a frozen model can infer a rule from examples. [[discovery|Program analysis]] looks for recognizable mathematical structure.

Each comparison controls something different. A [[prior-control|fixed program prior]] tests whether adaptive generation helps relative to a nonadaptive source. A [[pcfg|grammar baseline]] tests a different kind of synthetic structure. [[reward-ablations]] isolate parts of the reward definition.

The browser does not run the full experimental suite. It provides tiny fixed [[probes]], real source/output inspection, and descriptive [[diagnostics]]. These can make the mechanism understandable and show suggestive learning; they are insufficient to estimate a universal scaling law.

The paper's [[limitations]] and [[tuning|model-selection procedure]] belong in the interpretation, not in a footnote hidden from the result.`, 'scaling generator-quality icl-tasks discovery reward-ablations datasets limitations','S3');
a('probes','Probes are held-out test sequences','experiments','A probe tests the learner; it never supplies a gradient update or trains the generator.',String.raw`
Choose **Repeat, Cycle, Count, Text or Noise** above the transfer chart. The buttons switch the fixed test set displayed by the graph; they do not change training data or restart the run.

Repeat contains constant-byte sequences, Cycle repeats four-byte patterns, and Count contains arithmetic sequences modulo 256. Noise is a fixed pseudorandom sample intended to approximate independent uniform bytes. Text is a short fixed English passage. Repeat, Cycle, Count and Noise each use four sequences; Text uses one. These are small demonstrations rather than robust benchmarks.

Both the self-play learner and [[prior-control|control learner]] are evaluated on exactly the same selected bytes at the same training-round count. Their [[bpb|bits-per-byte]] scores are plotted; **lower is better**. The 8-bit line is a uniform byte-prediction reference.

Evaluation occurs at initialization, after the first round, and every 20 rounds. The most recent training round can therefore be newer than the last plotted test. The **At start** number is measured from the initial random model, not assumed to equal exactly eight.`, 'transfer prior-control bpb datasets sandbox','S3','Browser implementation');
a('transfer','What it means for learning to transfer','experiments','A training experience improves prediction on a different, unevaluated-by-training source.',String.raw`
Improvement on the current curriculum can result from fitting its quirks. Transfer asks whether the acquired representations or prediction rules also help on data outside that training process.

A clean evaluation holds the test data fixed while model weights change. Otherwise a lower score might only mean the test got easier. Keeping evaluation out of training also prevents the generator from optimizing directly against the benchmark.

The browser's [[probes]] measure limited transfer to simple synthetic patterns and one short text. The paper's [[datasets]] span several natural modalities. Neither should be confused with [[icl|in-context learning]], which varies examples inside a frozen model's context.

The [[prior-control]] helps interpret a gain: a learner can improve even with a nonadaptive source. Compare methods at matched training rounds or token counts, while acknowledging that self-play costs more compute per round. [[compute-frontier|Compute-optimal scaling]] is a separate, more demanding comparison.`, 'probes prior-control datasets icl compute-frontier','S3.SS1');
a('prior-control','What the fixed-prior control actually does','experiments','A second trainable learner gets outputs from uniformly sampled programs; its data source never adapts.',String.raw`
## What stays the same?
The control learner has the same architecture, initial learner weights, optimizer settings, output length and number of examples per round. Its programs use the same [[brainfuck|language]], [[macros]], execution budgets, input convention and [[padding]]. Both learners are tested on the same held-out sequences.

## What changes?
Every control instruction is sampled independently and uniformly from 19 tokens until F or the length cap. There is no trained generator, mutation curriculum or adaptive replay mixture. The source is fixed; **the control learner still learns**.

This is not an 8-bit horizontal line and not uniformly random output bytes. Uniform programs can print highly structured output. Their distribution is simply not steered toward the learner's current needs.

The chart's purple line is that second learner's actual test loss. The comparison matches learner rounds and byte counts, not wall-clock cost. It tests the adaptive pipeline against fixed-prior pretraining; it does not isolate just the [[gradient-reward|reward]] from all other pipeline components.`, 'prior bpb transfer pool reward-ablations','S3.SS1');
a('pcfg','The random-grammar baseline','experiments','A structured synthetic source with a language-like inductive bias.',String.raw`
A probabilistic context-free grammar recursively expands symbols using randomly selected productions. Repeated expansions can generate recurring motifs and nested structure without a natural text corpus.

In the paper's baseline, a fresh small random grammar supplies repeated words or sentences within each output row. Terminal symbols are nonzero bytes; generation is bounded and grammars are repaired to ensure productive derivations. Appendix H specifies the sampling and packing recipe.

This is a different prior over structure from executable [[program|programs]]. It can emphasize composition and reuse characteristic of formal language, but it does not supply a universal program search procedure in the same way.

A comparison with PCFG answers whether general adaptive program generation offers benefits beyond that particular structured source. It is different from the [[prior-control]], which keeps the program language fixed. The browser currently implements the latter, not a PCFG training arm.`, 'prior-control datasets related-work experiments','A8');
a('scaling','Reading a scaling law','experiments','A fitted relationship between held-out prediction loss and a compute budget.',String.raw`
The empirical fit has the form

\[L(C)=E+A C^{-b}.\]

\(E\) is a fitted loss floor, \(A\) a scale factor, and \(b\) a decay exponent. Doubling compute multiplies the excess loss \(L-E\) by \(2^{-b}\), not the total loss by that factor.

The paper fits [[compute-frontier|compute-optimal frontiers]] across model sizes, checkpoints and ensembles. A single browser model's curve against rounds is not such a frontier. Resource use and [[tuning]] choices affect the interpretation.

A power law over an observed range does not guarantee extrapolation forever. Fitted floors and exponents can trade off, and noisy small ranges can make estimates unstable. Similar-looking exponents across modalities motivate the [[ansatz|universal-data interpretation]], but do not prove its assumptions.

The article on [[exponents]] derives how a two-resource picture could produce effective exponents.`, 'compute-frontier ensemble tuning ansatz exponents','S3.SS1');
a('compute-frontier','Compute-optimal frontiers','experiments','Compare the best observed loss available at each compute budget, across configurations.',String.raw`
A point is on the empirical frontier if no measured configuration uses at most as much compute and achieves a lower loss. Dominated points are excluded from the fit.

The candidate set can vary model size, training duration and [[ensemble|ensemble size]]. This is more informative than extending one architecture indefinitely, because a larger model trained for fewer rounds may spend the same budget more effectively.

The frontier is only as good as the configurations explored. It is not a proof of the globally optimal use of compute. [[tuning|Hyperparameter selection]] and compute accounting must therefore be described.

The browser compares learners at matching rounds and token counts. The adaptive arm additionally pays for generator training, gradient rewards and [[shadow-history|historical reconstruction]]. A lower green curve should not be read as a demonstrated wall-clock advantage or a reproduced compute frontier.`, 'scaling ensemble tuning prior-control','S3.SS1');
a('ensemble','Ensembling predictive distributions','experiments','Average several models’ probabilities before scoring the true next byte.',String.raw`
For \(K\) independently trained predictors,

\[p_{\rm ens}(y\mid c)=\frac1K\sum_{k=1}^{K}p_k(y\mid c).\]

Score this average distribution with [[cross-entropy]]. Averaging losses instead would be a different operation. Because negative logarithm is convex, the ensemble's loss is no greater than the average member loss on the same target, although compute and storage increase.

Independent training seeds can make different mistakes; probability averaging can reduce those errors. It cannot guarantee improvement over the best member on every example.

The paper includes ensemble size among the configurations used to construct its [[compute-frontier]]. This browser shows single learners in each arm, not ensembles. A seed selector starts another experiment; it does not average it with the previous one.`, 'scaling compute-frontier seed cross-entropy','S3.SS1','Mathematical background');
a('tuning','Hyperparameter selection and leakage','experiments','No natural-data gradient updates does not mean no natural-data influence on model selection.',String.raw`
The paper uses validation loss from text and DNA to tune important hyperparameters. That introduces a model-selection signal from natural data, even though the generator and learner are not gradient-trained on those examples.

This distinction separates training leakage from selection leakage. Repeatedly choosing settings that score well on a benchmark can adapt a research pipeline to that benchmark without ever differentiating its loss.

The browser's displayed probes do not feed the training algorithm. A human who adjusts settings after inspecting them can still select for those probes. Treat the result as exploratory rather than an untouched final benchmark.

When comparing configurations, separate the best measured result, variation across [[seed|seeds]], the cost of searching configurations, and the training cost of the final run. The [[compute-frontier]] and [[pre-pretraining|warm-start experiment]] use different accounting conventions.`, 'tabula-rasa probes compute-frontier pre-pretraining seed','S3.SS1');
a('epiplexity','Structure relative to a bounded learner','experiments','A source can be compact to describe yet expensive for a finite learner to understand.',String.raw`
A tiny program can generate intricate output whose regularities take substantial training to acquire. Conversely, a long independent random sequence may be expensive to store yet offer little learnable structure. Description length and raw entropy do not capture the same thing as learning difficulty.

Epiplexity motivates measuring the structured information extractable by a compute-bounded observer. In the paper's [[generator-quality|generator-quality experiment]], an operational measure uses excess training loss accumulated while a fresh learner acquires a fixed generated corpus, relative to its converged loss.

The area over a learning curve's eventual floor can increase when there is more structure to acquire. It also depends on the learner and training procedure, so it is not an absolute semantic complexity meter.

This browser does not compute that fixed-corpus experiment or claim its changing-curriculum loss area is epiplexity. Its [[diagnostics]] are direct descriptive quantities with narrower meanings.`, 'generator-quality entropy solomonoff diagnostics','S5.SS0.SSS0.Px4');
a('generator-quality','Does the generator itself improve?','experiments','Freeze generated corpora at different stages and train fresh learners on them.',String.raw`
A learner's improving score alone cannot tell you whether the source improved or the learner simply practiced longer. To study the source, hold the downstream learner protocol fixed and change which generator checkpoints supply its corpus.

The paper constructs corpora from generator snapshots up to different endpoints, then trains new learners from scratch on those fixed corpora. It evaluates both [[epiplexity|extractable structure]] and transfer to natural data. This asks whether later generators contribute additional reusable information rather than merely repeating earlier outputs.

A moving self-play training curve cannot substitute for this experiment: both the source and learner change together. Likewise, a program with more loops is not automatically better training data.

The optional browser graphs show [[exploration|policy entropy]], [[program-length|length]], [[archive|coverage]], and rewards. They help inspect the process, but are not a replacement for the paper's controlled corpus-quality test.`, 'epiplexity experiments transfer diagnostics','S3.SS1');
a('icl','In-context learning without weight updates','experiments','A frozen predictor uses examples in its prefix to infer how to answer a new query.',String.raw`
During in-context evaluation, the optimizer is off. A prompt supplies examples of a relation, followed by a new input. The model must use its fixed learned computation to infer and apply that relation.

For example, pairs could reveal a new symbol mapping. Memorizing a mapping in context differs from training the weights on those pairs. A model may succeed by retrieving an example, identifying a pattern, or applying a more general algorithm; task design determines what alternatives are possible.

The paper measures greedy-answer success as the number of in-context examples increases across [[icl-tasks|several tasks]]. The browser's first-versus-last-token summary is not the same test: target positions and content change too.

[[attention]] makes earlier content accessible, but an attention architecture is not itself evidence of learned in-context abilities. The measured behavior matters. The [[sum-strategies|SUM analysis]] illustrates how predictive strategies can change as examples accumulate.`, 'icl-tasks sum-strategies context attention','S3.SS2');
a('icl-tasks','The paper’s in-context task suite','experiments','Six kinds of tasks test more than fitting common byte frequencies.',String.raw`
Examples use a byte interface, often with a zero sentinel before each input/output pair. The learner then sees a new input and predicts its answer without a gradient update.

Reverse-string asks for a reversed sequence; stack uses push/pop operations and asks for a popped value; associative recall asks for values from a context-provided dictionary. SUM adds bytes modulo 256, while MAX and MIN select an extreme input value.

These tasks require different behaviors: indexing, state tracking, contextual lookup, arithmetic, or comparison. A strong result on one does not establish all the others. Task formats and answer scoring also influence difficulty.

For SUM, the relation is \(f(a,b)=(a+b)\bmod256\). More examples can reveal the rule, but default copying heuristics may already yield nonzero accuracy on some tasks before examples are supplied. Inspect [[sum-strategies]] and distinguish such priors from adaptation.

The browser's [[probes]] are simpler loss tests, not an implementation of this six-task accuracy benchmark.`, 'icl sum-strategies probes attention','A4');
a('sum-strategies','How a SUM prediction can change','experiments','Confidence can first fall as an initial heuristic fails, then rise as a better rule is inferred.',String.raw`
A frozen model prompted with arithmetic examples can initially favor common bytes or copy a visible input. As more examples contradict that shortcut, its predictions can spread out. Later, a more accurate relation can become likely.

The paper's qualitative SUM analysis tracks both inferred response strategies and predictive [[entropy]] as demonstrations are added. It discusses a progression through partial low-bit and high-bit addition rather than an immediate jump to a fully reliable rule.

This is a behavioral interpretation of predictions. It is not by itself a complete circuit-level explanation of how the network computes addition. [[limitations|Mechanistic analysis]] or targeted interventions would be needed for stronger causal claims.

A rising entropy curve is not always deterioration: it can reflect losing unjustified confidence before discovering a better rule. That is why uncertainty, exact-match success and [[bpb|logarithmic loss]] provide different views.`, 'icl-tasks entropy limitations','S3.SS2');
a('discovery','Recognizable mathematical output','experiments','Program outputs can be examined for recurrences, but short coincidences are not discoveries.',String.raw`
The paper examines output sequences for arithmetic, quadratic, cubic, Fibonacci-like and geometric structure modulo 256. These patterns are properties of actual executed bytes, not names selected by the generator.

Arithmetic sequences have constant first differences. Quadratic and cubic sequences have constant second and third differences. Fibonacci-like sequences satisfy \(v_n=v_{n-1}+v_{n-2}\pmod{256}\), while geometric sequences satisfy \(v_n=r v_{n-1}\pmod{256}\).

## Detection needs safeguards
Short constant tails can accidentally match many recurrences. The paper's criteria require a sufficiently long minimal period and permit a bounded unstructured prefix. Its stored sample frequency also limits the precision of “first discovery” timing.

The [[random-discovery|random-program comparison]] asks how often the same criteria trigger without adaptive generation. The browser deliberately shows raw outputs and predictions rather than awarding a “Fibonacci discovered” badge to a short accidental match.`, 'random-discovery machine program-length limitations','A3');
a('random-discovery','Comparing discovery with random search','experiments','Use the same interpreter, alphabet, and detector before comparing discovery rates.',String.raw`
If one method gets helpful macros or generous execution budgets and the other does not, a discovery gap would not isolate the benefit of learning a program distribution. Both should search the same represented program space with the same detection rule.

The paper compares structured outputs with a large uniform-program sample. Rare-event conclusions must account for the number of trials and for differing checkpoint sampling frequencies.

A useful statistical background fact: if zero events occur in \(n\) independent trials, the rough 95% “rule of three” upper bound is \(p\lesssim3/n\). Zero observed successes does not prove impossibility or an exactly zero probability.

Nor does detecting a mathematical family prove that it caused natural-data [[transfer]]. A stronger experiment would remove or manipulate that family while keeping other aspects of training controlled. The paper identifies such causal questions as future work.`, 'discovery prior-control limitations transfer','A3');
a('reward-ablations','What the reward ablations test','experiments','Alter one part of the signal and measure downstream consequences.',String.raw`
A signed reward removes the absolute value. A last-step reward replaces the long historical window with one update. A loss-delta reward substitutes measured immediate loss change. Shuffling rewards across programs preserves the batch's reward histogram while breaking its association with source programs. Negating the signal reverses what it favors; uniform sampling removes adaptive generation.

These interventions test different hypotheses. If shuffling hurts, the mapping from example to reward matters, not just the marginal reward scale. If a short window behaves differently, temporal aggregation matters. None alone proves the complete intuitive story.

Appendix F reports differing results across modalities and seeds, including instability in some variants; a blanket claim that every alternative is worse on every dataset would be too strong.

The browser exposes the [[noise-trap|difficulty ablation]] plus the [[prior-control|fixed-prior comparison]]. Other ablations are explained here but are not silently approximated by those two controls.`, 'absolute-value history directional-derivative noise-trap prior-control','A6');
a('datasets','Why encode different domains as bytes?','experiments','A shared prediction interface allows the same model to score unlike data sources.',String.raw`
Each benchmark becomes a sequence of values from 0 to 255. The model's objective remains next-byte [[cross-entropy]] rather than switching between image classification, speech recognition and language generation.

The representation still matters. UTF-8 text, raw image channels, sampled audio, symbolic melody and DNA symbols expose different correlations within a fixed [[context]]. Removing file headers and metadata helps test the intended content rather than easy container conventions.

Follow [[text-data]], [[image-data]], [[audio-data]], [[music-data]], [[dna-data]], and [[code-math-data]] for how to interpret these formats. These articles explain the paper's benchmark design; the browser does not download or evaluate those full datasets.

Byte-level comparability also has limits. A loss floor can depend on encoding and symbol inventory, and the amount of meaningful time or spatial extent per context can differ radically. Similar [[scaling|scaling curves]] do not erase those differences.`, 'text-data image-data audio-data music-data dna-data code-math-data','A2');
a('text-data','Natural text as UTF-8','experiments','Predict text bytes, not a pretrained word-token vocabulary.',String.raw`
UTF-8 represents characters using one or more bytes. Predicting that stream can reward spelling, punctuation, repeated names and longer linguistic regularities, without introducing a learned subword tokenizer.

The paper's text evaluation uses DCLM-derived content, excluding its JSON container fields. Such a benchmark asks about prediction of text, not instruction following, factual answering or conversation quality.

The browser's Text test is one short fixed English passage. A loss decrease there can reflect byte-frequency or local pattern effects. It is too small to establish language competence, and a self-play model can worsen on it while improving counting.

A [[contingent|specific fact]] absent from training cannot be inferred merely because a model has learned generic syntax. The [[ansatz|universal-data hypothesis]] distinguishes those kinds of predictive information.`, 'datasets contingent probes transfer','A2.SS1');
a('image-data','Images as raw pixel streams','experiments','Remove labels and containers, then ask for the next color-channel byte.',String.raw`
An RGB image can be serialized as all red values, then all green, then all blue (planar), or as RGB triplets in pixel order (interleaved). Both can be lossless, but nearby positions in the stream represent different spatial or channel relationships.

The paper evaluates CIFAR-10 image bytes after removing labels and metadata. Lower next-byte loss can indicate better use of color and spatial regularities; it does not directly measure object recognition.

A one-dimensional context must span enough serialized pixels to expose the structure of interest. Changing channel order or window length changes that opportunity. This illustrates why a shared [[datasets|byte interface]] is not a guarantee of equivalent task difficulty.

The browser has no image benchmark arm; its displayed tapes are a visualization of numeric bytes, not generated CIFAR images.`, 'datasets context transfer','A2.SS2');
a('audio-data','Audio, sampling rates and context','experiments','A short byte window can cover only a few milliseconds of raw sound.',String.raw`
A waveform is a sequence of amplitude measurements. Reducing sample rate or quantizing amplitudes changes how much acoustic time fits into a given number of bytes.

The paper describes Speech Commands audio converted to an 8-bit amplitude representation at several sample rates, after removing the WAV container. Its tables also distinguish audio encodings; “audio loss” must be read with the particular representation in mind.

At 16,000 one-byte samples per second, a 64-byte context spans only 4 milliseconds. Predicting neighboring amplitudes over that window is very different from recognizing a spoken word or reasoning about an utterance.

[[music-data|Symbolic music]] packs events more compactly. Comparisons should therefore account for physical time per context, not merely the numerical byte count or the visual similarity of loss curves.`, 'datasets context music-data scaling','A2.SS3');
a('music-data','Symbolic melody rather than raw sound','experiments','Represent musical events on a time grid, making longer structure visible within a context.',String.raw`
A melody stream can encode a pitch, a held note, or a rest at each time step. This uses far fewer tokens than storing all waveform samples and makes recurring phrases accessible to shorter contexts.

The paper describes a monophonic melody representation on a sixteenth-note grid derived from Mutopia music, discarding MIDI file metadata. Pitch values occupy the MIDI range, with separate symbols for hold, rest and end.

Predicting this representation concerns event and temporal regularities. It is not the same as generating realistic audio, reconstructing a full multi-instrument performance, or judging artistic quality.

The contrast with [[audio-data|raw audio]] shows how an encoding determines the structure available to a finite predictor. A universal-byte model has a common output interface, but the information density of that interface remains a design decision.`, 'datasets audio-data context','A2.SS4');
a('dna-data','DNA and restricted symbol inventories','experiments','A byte vocabulary may be larger than the actual alphabet used by a benchmark.',String.raw`
DNA encodings may distinguish four bases and upper/lowercase variants used for masking. The paper describes an eight-symbol numeric DNA representation obtained from the KoLMogorov benchmark's reference-genome-derived stream.

Uniform prediction over eight known symbols costs \(\log_2 8=3\) bits. Uniform prediction over all 256 possible bytes costs 8 bits. A model can therefore improve substantially just by allocating probability to the active alphabet, before learning longer biological patterns.

That does not make alphabet learning illegitimate, but it limits what a loss reduction proves. Repeated motifs, local composition and masking runs provide further predictive opportunities.

This is a prediction benchmark, not a claim that the model understands biological function or can perform reliable genomic inference. The [[transfer|evaluation question]] remains next-symbol probability under this encoding.`, 'datasets bpb entropy transfer','A2.SS5');
a('code-math-data','Formal proofs and source code as data','experiments','Predict serialized symbols without asking the model to execute code or prove a theorem.',String.raw`
The paper's formal-mathematics evaluation uses serialized Metamath material with preprocessing, while the code evaluation includes a C source corpus. Both become byte-prediction tasks.

Repeated syntax, identifiers, proof labels and common fragments can all improve next-byte likelihood. That improvement is distinct from satisfying a compiler, solving a programming problem, or producing a correct proof.

A source-code dataset and a generated [[program]] also play different roles. The former is held-out evaluation data; the latter is executed to generate training bytes. The learner never receives its own training programs as source-code examples.

When reading a claim of transfer to “code” or “math,” first identify the task and metric. [[bpb|Prediction loss]] supports a narrower conclusion than verified algorithmic correctness.`, 'datasets program learner transfer','A2.SS6');
a('universal-structure','Reusable predictive structure','theory','Patterns such as repetition and composition can be useful across unrelated data sources.',String.raw`
Learning to exploit a repeated substring may help on source code, music events or formatted text. The symbols and subject matter differ, but the predictive operation can be similar.

The paper's interpretation separates this reusable component from [[contingent|information specific to the observed world]]. It does not identify a unique measurable unit of universal structure or prove that all observed gains arise from one such resource.

Program search provides a broad language for expressing data-generating processes. [[solomonoff|Universal prediction]] supplies motivation for considering that space, while the neural learner attempts to amortize useful prediction into finite computation.

The [[ansatz]] turns this distinction into a proposed scaling decomposition. Treat it as an explanatory hypothesis, whose assumptions should be checked, rather than a theorem derived from the self-play update rule.`, 'contingent ansatz solomonoff epiplexity','S4');
a('contingent','Information about this particular world','theory','Generic computation cannot supply unobserved arbitrary facts.',String.raw`
If two possible worlds obey the same structural rules but choose different arbitrary names or dates, a model trained on neither cannot know which choice our world made. That information must enter through observations, interaction or context.

This does not prevent useful [[transfer]]. Generic prediction skills can make later observations easier to learn from. It explains why a zero-natural-data procedure need not replace natural training corpora.

In the [[ansatz]], contingent information is held fixed during self-play and absorbed into an effective loss floor. Taking a literal singular formula at \(D_c=0\) is not the intended operation; the resource model is a phenomenological approximation over its applicable regime.

Very long context could itself supply information about a target source at evaluation time. With finite context, access to that evidence is limited. Distinguish information encoded in weights from information supplied in the test prefix.`, 'universal-structure ansatz context pre-pretraining','S6');
a('solomonoff','Universal prediction as motivation','theory','Consider computable explanations of a sequence, with shorter descriptions receiving more prior weight.',String.raw`
A schematic algorithmic mixture weights programs by description length and sums over those consistent with the observed prefix:

\[M(y)\sim\sum_{p:\,U(p)\sqsupseteq y}2^{-|p|}.\]

A precise construction needs a suitable program coding convention and semimeasure definitions. Exact universal prediction is not generally computable; it is not an algorithm this browser runs.

The connection is motivational: a universal language can represent many data-generating processes, and short procedures can encode recurring structure. The paper replaces exhaustive inference with a learned generator and a trained predictor under finite compute.

The [[prior|19-token program prior]] is a concrete design inspired by description-length weighting. The adaptive generator changes where finite search effort is spent; it does not become an exact Solomonoff posterior. [[epiplexity]] further emphasizes the importance of an observer's computational limits.`, 'prior termination-prior universal-structure epiplexity','S5.SS0.SSS0.Px3');
a('ansatz','The universal-data scaling hypothesis','theory','A proposed decomposition of model capacity, world-specific information, and reusable structure.',String.raw`
Start with a familiar model/data scaling form:

\[L=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.\]

The proposed refinement separates effective data resources:

\[L=E+\frac{A}{N^\alpha}+\frac{B}{D_c^\beta}+\frac{C}{D_u^\gamma}.\]

Here \(N\) is model size, \(D_c\) effective [[contingent|contingent information]], and \(D_u\) effective [[universal-structure|universal structure]]. These are explanatory variables, not quantities directly counted by the browser.

If self-play leaves contingent information fixed, absorb its contribution into \(E'\). If useful generated structure grows as \(D_u(T)\propto T^\eta\), substitution yields

\[L=E'+\frac{A}{N^\alpha}+\frac{C'}{T^{\gamma\eta}}.\]

This algebra explains how a power law could arise under those assumptions. It does not derive the growth assumption from the optimizer or prove continued discovery. [[exponents]] examines the comparison with natural-data exponents, and [[limitations]] separates hypotheses from evidence.`, 'universal-structure contingent exponents scaling limitations','S4');
a('exponents','Why similar scaling exponents are suggestive','theory','A fitted exponent can mix several underlying improvements.',String.raw`
Suppose natural data increases the two proposed resources as \(D_c\propto D^\nu\) and \(D_u\propto D^\mu\). Substitution into the [[ansatz]] gives

\[L=E+\frac A{N^\alpha}+\frac{B'}{D^{\beta\nu}}+\frac{C'}{D^{\gamma\mu}}.\]

At sufficiently large \(D\), the more slowly decaying data term dominates, suggesting

\[\beta_{\rm observed}\approx\min(\beta\nu,\gamma\mu).\]

In the proposed self-play picture, the data-progress exponent is \(\gamma\eta\). Similar observed exponents could therefore be consistent with reusable structure limiting both processes. Similarity alone does not identify these latent exponents or rule out other explanations.

A further distinction is data scaling versus compute scaling. Under a simple compute constraint \(C\propto ND\), balancing terms of a two-term model can give a compute exponent \(\alpha\beta/(\alpha+\beta)\). Such conversions rely on their own allocation assumptions. Do not compare all quoted exponents as if they measured the same axis.`, 'ansatz scaling compute-frontier limitations','S4');
a('pre-pretraining','Self-play as a warm start','experiments','Use learned synthetic structure as initialization before ordinary natural-data training.',String.raw`
A warm-start experiment compares subsequent training from self-play weights with training from random weights, using the same downstream architecture and corpus. It asks whether synthetic experience reduces the amount of later natural-data training needed.

The paper reports this as a separate experiment from zero-shot evaluation. Once natural-data optimization begins, the model is no longer trained exclusively on self-generated bytes. Distinguish the initial zero-shot point from later downstream learning curves.

The reported downstream comparison does not charge the earlier self-play compute to each run; it treats that checkpoint as a reusable one-time investment. Consequently fewer downstream tokens do not automatically imply lower total end-to-end compute.

Both arms also have their own selected optimizer settings. These accounting and [[tuning|selection details]] are essential when interpreting a practical speedup. The browser does not currently implement natural-data continuation.`, 'tabula-rasa tuning contingent experiments','A1.SS1');
a('related-work','How this connects to other approaches','theory','Adaptive program generation sits between universal prediction, synthetic curricula and intrinsic motivation.',String.raw`
[[solomonoff|Universal prediction]] motivates a broad space of computable sources. Earlier algorithmic pretraining explores learning from computations without requiring a natural corpus. The distinction here is an adaptive neural program source trained alongside its learner.

Synthetic-data methods can also rephrase or reorganize a known corpus. Those procedures may extract more learning signal from already observed facts, unlike a strictly [[tabula-rasa|random-initialization]] setting.

Intrinsic motivation and automatic curricula ask which experiences create learning progress rather than merely surprise. The [[noise-trap]] explains why those two criteria differ. The paper's particular [[gradient-reward|historical gradient reward]] is one concrete proposal in that broader family.

[[epiplexity]] offers a way to discuss structured information relative to a compute-bounded observer. The paper's reference list is the starting point for historical attribution and primary sources; this companion explains the conceptual relationships rather than claiming all these ideas originated in one paper.`, 'solomonoff tabula-rasa noise-trap epiplexity','S5');
a('limitations','What this experiment does not establish','theory','Mechanism fidelity is different from proving broad transfer, universal scaling, or inevitable open-ended discovery.',String.raw`
The paper's empirical results concern specific architectures, budgets, program semantics, baselines and evaluation formats. Its universal-data decomposition is an interpretation with assumptions, not a theorem ensuring that more self-play compute always yields better natural-data prediction.

Important questions remain about scaling to larger models, designing more expressive search spaces, separating the causal role of discovered program families, and practical total-compute accounting. Validation-based [[tuning]] also qualifies the “zero data” framing.

The browser is smaller still. A few fixed [[probes]] can improve while other capabilities worsen; different seeds can follow different paths. Low [[padding|padding loss]], many [[archive|archive niches]], high reward or long programs are not interchangeable with useful transfer.

An early high-reward F row can reflect learning the zero padding supplied after an empty execution. The reward-sorted curriculum includes replay and mutations, so its leading rows are not an estimate of the generator's sampling probabilities. Changing model depth, rates or budgets can change the trajectory; a healthy run at one setting does not rule out collapse elsewhere. The [[gradient-reward|absolute alignment heuristic]] does not guarantee that each simple program's reward falls steadily or that the curriculum eventually becomes more complex.

The [[validation|independent audit]] checks whether the stated mechanism is genuinely implemented. It cannot certify all scientific conclusions of the paper or predict what this particular run will eventually discover.`, 'experiments tuning probes validation ansatz','S6');
a('diagnostics','What the optional diagnostics measure','practice','Direct observations of this run, with hover explanations tied to captured data.',String.raw`
The foldout graphs track within-batch reward mean and standard deviation, source-token length mean and standard deviation, emitted-token fraction, and generator entropy. Archive occupancy and output-value coverage are additional descriptions of exploration.

Each quantity has a distinct meaning. [[reward-statistics|Reward spread]] affects standardized advantages. [[program-length|Length]] describes source strings, not semantic complexity. [[emissions|Emission rate]] distinguishes actual machine output from padding. [[exploration|Policy entropy]] describes local sampling uncertainty.

Hover a plot to inspect a retained sample nearest that horizontal position. Hover a program, reward, or output row to capture that actual row's computation. Nested explanations inherit the same captured context instead of jumping to a newer training round.

Snapshots stay fixed while you read, so a changing curriculum does not silently change the numbers under an explanation. Close the tooltip and hover again for current data. None of these displays feeds back into training.`, 'reward-statistics program-length emissions exploration archive','S3','Browser diagnostics');
a('reward-statistics','Reward mean and standard deviation','practice','A batch’s center and spread describe the raw signal before advantage normalization.',String.raw`
\[\bar r=\frac1M\sum_i r_i,\qquad\sigma_r=\sqrt{\frac1M\sum_i(r_i-\bar r)^2}.\]

This is the population standard deviation of the current pool, not uncertainty about an estimated generalization score. A large spread means a few rows can differ strongly from the rest. It does not imply statistically significant improvement.

The generator uses these values in its [[advantage]]. If the spread is tiny, small differences can become prominent after normalization, with epsilon limiting division. Rewards also depend on current weights, optimizer state and historical displacement, so their absolute scale can drift across training.

The optional graph plots the actual mean with mean ± standard deviation. The hover breakdown shows individual rewards for the captured batch. Neither curve has a universal “higher is better” interpretation.`, 'advantage gradient-reward diagnostics zero-reward','S2.SS2','Browser diagnostics');
a('program-length','Source length is not complexity','practice','Count sampled instruction tokens, including F when present; keep this separate from emitted output length.',String.raw`
A short loop can print a long structured stream. A long string can immediately jump over most of its code or produce no output. Source length is therefore a useful search descriptor, not a semantic complexity score.

The diagnostics average token counts including a sampled F. The [[archive]] instead bins program-body lengths before F. Those conventions differ by one for normally terminated programs and are labeled explicitly.

Length also affects sequence log probability, the [[kl|prior penalty]], and the hard cap. Inspect a shift toward very short or capped programs alongside [[exploration|entropy]], output behavior and [[transfer|test loss]], rather than assuming longer is better.

The [[termination-prior|uniform prior's length distribution]] supplies a mathematical reference for why many short programs occur even without learning.`, 'termination-prior archive kl diagnostics','A7','Browser diagnostics');
a('emissions','How much of the row was actually printed?','practice','The emitted fraction counts real output instructions, including emitted zeros.',String.raw`
\[q=\frac{\sum_i n_i^{\rm emitted}}{M T}.\]

A high value means programs fill more of the learner's fixed-length output budget. It does not guarantee useful structure: a random-printing loop can fill every position. A low value means padding has a large influence on aggregate loss.

The row tooltip separates emitted positions from padded positions and plots actual token surprises. The latter can reveal that a low total loss comes mostly from easy zeros after early termination.

The [[program|execution viewer]] shows emitted output as it happens. It adds no pretend intermediate bytes. Only when execution ends do the remaining slots acquire the training convention of zero padding.

Interpret emission rate together with [[padding]], [[noise-trap|randomness]], and [[probes|held-out prediction]].`, 'padding program noise-trap diagnostics','A5','Browser diagnostics');
a('exploration','Inspecting generator exploration','practice','Policy entropy, lengths and archive coverage describe different aspects of program search.',String.raw`
Generator entropy is measured from the actual prefix distributions of freshly sampled programs: average over positions within each program, then average those values across fresh programs. It is measured in bits per next instruction, with maximum \(\log_2 19\). It is not the entropy of output bytes or the number of distinct complete programs the policy can produce.

A concentrated policy may be exploiting a useful pattern or collapsing onto unhelpful output. A diffuse policy may be productively exploring or wasting effort. The distinction requires observing outputs and [[transfer|evaluation]], not entropy alone.

[[archive|Niche occupancy]] measures coverage of a hand-defined loop-depth/length grid. [[program-length]] measures source size. Neither guarantees coverage of the space of semantic behaviors.

The [[kl|prior regularizer]], fresh sampling, [[mutation]] and [[replay]] affect exploration in different ways. Their combined behavior can be studied with the diagnostics, but causal attribution requires controlled ablations.`, 'entropy archive mutation kl diagnostics','A7','Browser diagnostics');
a('shadow-history','Exact historical checkpoints with bounded growth','practice','Deterministic slower copies reconstruct the requested halfway state instead of storing every weight snapshot.',String.raw`
A trainer at round \(e\) maintains an identical copy advanced to \(\lfloor e/2\rfloor\). That copy maintains its own halfway copy, and so on. All share the same initial seed and fixed configuration, but each owns its RNG and model state.

Because the training process is deterministic within a runtime, the slower copy reproduces the earlier weights exactly. Total extra training rounds are bounded by the geometric series \(e/2+e/4+\cdots<e\), with logarithmically many trainer states and bounded program banks.

UI actions and diagnostics consume no training randomness. Otherwise this reconstruction would fail. The [[validation|tests]] compare full weights and rewards against a simpler implementation retaining all historical snapshots.

A [[checkpoints|downloaded checkpoint]] stores these historical trainers too, so restoring does not silently change the future reward. Floating-point differences across browser engines can still alter long-run sampled trajectories.`, 'history checkpoints seed validation','S2.SS2','Browser implementation');
a('seed','Seeds and reproducibility','practice','A seed fixes the pseudorandom stream and initial weights, not a guarantee of a particular discovery.',String.raw`
Random initialization, token sampling, mutation and interpreter input all affect the training path. A seeded pseudorandom generator makes these choices reproducible when configuration and runtime behavior match.

Probe data uses a separate fixed seed. Execution viewers use captured copies of execution RNG state. Opening a tooltip or switching the displayed test therefore cannot spend training randomness or change the curriculum.

Tiny numerical differences between JavaScript engines can eventually cross a sampling threshold and change later choices. Reproducible within one tested runtime is a narrower claim than bit-identical behavior on every device.

Compare multiple seeds before treating a favorable run as representative. A seed is an experimental condition; selecting it after observing outcomes is part of [[tuning|model selection]].`, 'tabula-rasa tuning random-tape checkpoints','S3','Browser implementation');
a('checkpoints','What a saved run contains','practice','Weights alone are not enough to continue this training process faithfully.',String.raw`
A complete checkpoint includes both models, optimizer moments, RNG state, replay entries and their proposal probabilities, quality-diversity archives, [[shadow-history|historical trainers]], and plot history.

Restoring just the learner weights would change future rewards and examples. Restoring the models without Adam moments would change both updates and the [[preconditioner]]. Losing original replay probabilities would alter [[importance|importance correction]].

The page autosaves locally and can download a JSON checkpoint. Loading resumes in a paused state. A checkpoint does not publish data to a server, and closing the browser stops computation.

Older checkpoints may lack the newer diagnostic measurements. Those observations cannot be reconstructed from a scalar loss alone; the page shows missing history rather than inventing it. New rounds supply the added diagnostics.`, 'shadow-history adam importance diagnostics','S2','Browser implementation');
a('sandbox','The program sandbox is evaluation only','practice','Run a chosen program against the current learner without teaching either model from it.',String.raw`
The sandbox executes user-supplied source using the current machine budgets and chosen input seed, then asks the learner for conditional predictions on its output. It makes no optimizer update and adds nothing to the replay bank or archive.

That separation matters: injecting a hand-written counting program into training would change the claim that the curriculum started without seeded examples. Running it as a test is a different operation.

Use it to inspect [[brainfuck|instruction semantics]], compare [[teacher-forcing|predictions]], or test a hypothesis about the learner's current behavior. Repeated human-guided experimentation can still amount to selecting tests or configurations after seeing results, so it is exploratory evidence rather than a pristine benchmark.

The simulator replays the sandbox execution with its chosen seed, using the same primitive interpreter as training.`, 'program probes teacher-forcing tabula-rasa','A5','Browser implementation');
a('demo-differences','Browser scale versus paper scale','practice','The mechanism is preserved, while architecture and resource budgets are deliberately small.',String.raw`
This playground uses small causal transformers with RMS normalization, learned positions and ReLU feed-forward blocks. The paper uses Llama-style models at larger scales and contexts. The byte prediction, executable program search, historical gradient reward, policy gradients and expert iteration are genuine computations here.

Quick and Long run are preset configurations. The advanced foldout exposes actual architecture and training settings and shows the resulting transformer shape. Larger dimensions, more layers, longer context and larger pools can substantially reduce rounds per minute.

The browser offers a small [[probes|test suite]], not the paper's full natural-data benchmarks, ICL tasks, ensembles or fitted compute frontiers. The [[prior-control]] comparison matches learner tokens, not total computation.

These limits should be kept visible when interpreting a result. The independent [[validation|audit]] establishes that core concepts were not replaced with scripted stand-ins; it does not equate this model's capabilities with those in the paper.`, 'transformer budgets probes validation limitations gpu','S2.SS3','Browser implementation');
a('validation','How the implementation is checked','practice','Numerical and structural tests inspect the computation, not just whether the page animates.',String.raw`
Finite-difference tests check gradients across parameter blocks. Causality tests alter future targets and require earlier predictions to stay unchanged. Sampling tests compare incremental generation likelihood with full teacher-forced likelihood.

The reward is independently compared against a [[directional-derivative|finite directional derivative]], including the Adam preconditioner and historical difference. [[shadow-history|Historical reconstruction]] is compared with retained snapshots. Checkpoint tests require continued training to agree exactly on the same device and backend within the test runtime.

The execution viewer uses the same stepper as batch execution, with captured input state. Diagnostic sums are checked against the actual reward. Graph-reachability tests ensure explanatory concepts are linked from playground entrypoints rather than left as disconnected articles.

A separate agent's conceptual audit is included with the source. These tests reduce implementation risk; they do not substitute for larger scientific experiments about [[transfer]] or [[scaling]].`, 'directional-derivative shadow-history program limitations','S2','Browser implementation');

a('gpu','GPU and CPU execution','practice','WebGPU computes the actual transformer forward passes and full parameter gradients on this device.',String.raw`
The default backend uses WebGPU through a locally bundled TensorFlow.js runtime. Learner losses, learner parameter gradients, generator log probabilities, and the weighted generator gradient are tensor computations on the GPU. There is no remote inference service and no pretrained network behind the playground.

The interpreter, cached autoregressive sampling, reward dot products, Adam updates, and held-out evaluation run on the CPU. Weight and gradient transfers cross the CPU/GPU boundary each round. This is real GPU training, but it is not a fully GPU-resident training pipeline. Small models can be limited by dispatch and transfer overhead; increasing model size gives each dispatch more work.

The [[gradient-reward|historical reward]] still uses all parameters. Recursive [[shadow-history|historical trainers]] use the same training backend as the live trainer. GPU floating-point reductions differ slightly from the manual CPU path; tiny differences can eventually change sampled programs. Results should be reproduced on the same device, backend and browser/runtime. Across devices, bit-identical continuation is not promised.

If WebGPU initialization fails, choose CPU explicitly in the model settings or use a WebGPU-capable browser. The page never silently labels CPU training as GPU training. A failure midway through a training round makes that run unusable: restart or restore a completed checkpoint, so a partial update cannot corrupt the historical comparison.

Numerical [[validation|checks]] compare GPU losses and every parameter gradient with the independent CPU implementation, including multiple layers and signed generator weights. They also compare historical reconstruction with retained checkpoints and check that temporary tensors are disposed.`, 'transformer gradient-reward shadow-history checkpoints validation demo-differences','S2.SS3','Browser implementation');

export const PREREQUISITES={
 'gradient-reward':['gradients','history','preconditioner'],'preconditioner':['adam','gradients'],
 'adam':['gradients'],'directional-derivative':['gradients'],'policy-gradient':['autoregressive','advantage'],
 'advantage':['reward-statistics','kl'],'importance':['autoregressive','replay'],'expert-iteration':['cross-entropy'],
 'ansatz':['universal-structure','contingent','scaling'],'exponents':['ansatz'],
 'attention':['transformer'],'icl-tasks':['icl'],'shadow-history':['history','seed']
};
export function outgoing(id){const a=ARTICLES[id];return [...new Set([...a.related,...(PREREQUISITES[id]??[]),...Array.from(a.body.matchAll(/\[\[([\w-]+)(?:\|[^\]]+)?\]\]/g),m=>m[1])])].filter(x=>x!==id);}
export function backlinks(id){return Object.keys(ARTICLES).filter(k=>outgoing(k).includes(id));}
export function titleOf(id){return ARTICLES[id]?.title??id;}
