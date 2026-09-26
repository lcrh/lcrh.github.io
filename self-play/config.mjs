import {DEFAULTS,PRESETS} from './trainer.mjs';
export {PRESETS};
// Conservative single-array byte-addressing bound, not available RAM.
// GPU limits come from the actual initialized device rather than a common minimum.
export const MAX_TYPED_ARRAY_BYTES=2**31-4;
export const FIELDS=[
 ['dim','Model width',2,8192,1,'transformer'],['heads','Attention heads',1,128,1,'attention'],['layers','Transformer layers',1,128,1,'transformer'],['ffMultiplier','MLP width multiplier',1,32,1,'transformer'],
 ['context','Output / learner context (bytes)',4,32768,1,'context'],['programLength','Generator context (tokens)',1,32768,1,'program'],['pool','Programs per round',4,4096,1,'pool'],['steps','Primitive steps per program',1,1000000000,1,'budgets'],['memory','Tape cells',1,16777216,1,'machine'],
 ['lr','Learner learning rate',1e-10,1,'any','adam'],['generatorLR','Generator learning rate',1e-10,1,'any','policy-gradient'],['beta','Prior KL strength β',0,100,'any','kl'],['expert','Expert iteration weight',0,100,'any','expert-iteration'],['bankSize','Replay bank capacity',1,1048576,1,'replay']
];
const REAL_FIELDS=new Set(['lr','generatorLR','beta','expert']);
export function tensorAllocations(c){const d=c.dim,f=d*c.ffMultiplier;return [
 ['learner token embeddings',257*d],['generator token embeddings',20*d],['learner positions',c.context*d],['generator positions',c.programLength*d],
 ['attention projection',d*d],['MLP projection',d*f],['learner attention',c.heads*c.context*c.context],['generator attention',c.heads*c.programLength*c.programLength],
 ['learner MLP activations',c.context*f],['generator MLP activations',c.programLength*f],['learner probabilities',c.context*256],['generator probabilities',c.programLength*19]
].map(([name,elements])=>({name,elements,bytes:elements*4}));}
export function validateConfig(input,deviceLimits={}){
 const c={...DEFAULTS,...input};
 for(const [key,label,min,max] of FIELDS){const x=c[key];if(!Number.isFinite(x)||x<min||x>max||(!REAL_FIELDS.has(key)&&!Number.isSafeInteger(x)))throw Error(`${label} must be ${min}–${max}${REAL_FIELDS.has(key)?'':' (safe whole numbers)'}.`);}
 if(c.dim%c.heads)throw Error('Model width must divide evenly into attention heads.');
 const knownLimits=['maxStorageBufferBindingSize','maxBufferSize'].map(k=>deviceLimits?.[k]).filter(x=>Number.isFinite(x)&&x>0),limit=Math.min(MAX_TYPED_ARRAY_BYTES,...knownLimits);
 for(const allocation of tensorAllocations(c))if(!Number.isSafeInteger(allocation.elements)||allocation.bytes>limit)throw Error(`${allocation.name} requires ${formatBytes(allocation.bytes)} in one tensor, exceeding ${formatBytes(limit)} ${knownLimits.length?'on the selected GPU device':'for this implementation’s single CPU array'}. Reduce width, heads, context, or MLP multiplier.`);
 if(!Number.isSafeInteger(c.seed)||c.seed<1||c.seed>999999999)throw Error('Seed must be a whole number from 1 to 999999999.');
 if(!['selfplay','difficulty','prior'].includes(c.mode))throw Error('Unknown training objective.');return c;
}
export function parameterCount(c,vocab,context){const d=c.dim,f=d*c.ffMultiplier;return (vocab+1)*d+context*d+c.layers*(2*d+4*d*d+2*d*f)+d+vocab*d+vocab;}
export function resourceEstimate(c,{compare=true,rounds=10000}={}){
 const learner=parameterCount(c,256,c.context),generator=parameterCount(c,19,c.programLength),parameters=learner+generator;
 // CPU always retains weights, gradients and both Adam moments for each model.
 const pairStateBytes=parameters*4*4,historyStates=c.mode==='prior'?1:2+Math.floor(Math.log2(Math.max(1,rounds)));
 const activations=(n,v)=>4*(c.layers*(c.heads*n*n+14*n*c.dim+2*n*c.dim*c.ffMultiplier)+n*v);
 const workingBytes=4*(activations(c.context,256)+c.pool*activations(c.programLength,19))+8*parameters;
 return {learner,generator,parameters,pairStateBytes,historyStates,initialStateBytes:pairStateBytes*((c.mode==='prior'?1:2)+(compare?1:0)),longRunStateBytes:pairStateBytes*(historyStates+(compare?1:0)),workingBytes,bankTokenSlots:c.bankSize*c.programLength*(historyStates+(compare?1:0)),attentionCells:c.pool*c.layers*c.heads*(c.context*c.context+c.programLength*c.programLength)};
}
export function formatBytes(bytes){return bytes>=1024**3?`${(bytes/1024**3).toFixed(2)} GiB`:`${(bytes/1024**2).toFixed(1)} MiB`;}
export function shapeHTML(c,options={}){const e=resourceEstimate(c,options);return `<div class="shape"><span>Token + position embeddings<br><b>width ${c.dim}</b></span><i>→</i><span><b>${c.layers} transformer layer${c.layers===1?'':'s'}</b><br>RMSNorm → causal attention: ${c.heads} × ${c.dim/c.heads}<br>residual → RMSNorm → MLP: ${c.dim} → ${c.dim*c.ffMultiplier} → ${c.dim}<br>residual</span><i>→</i><span>RMSNorm → output<br><b>19 instructions / 256 bytes</b></span></div><p>Two independent models: generator ${e.generator.toLocaleString()} parameters, learner ${e.learner.toLocaleString()}. Each round trains on ${(c.pool*c.context).toLocaleString()} learner bytes, including padding.</p><p><b>Resource estimate, not a device-fit guarantee:</b> ${formatBytes(e.pairStateBytes)} of CPU arrays per model pair for weights, gradients and Adam moments. The initial historical copy${options.compare===false?'':' and fixed-prior control'} brings that to about ${formatBytes(e.initialStateBytes)}. Around 10,000 rounds, up to ${e.historyStates} historical trainer states${options.compare===false?'':' plus the control'} can require ${formatBytes(e.longRunStateBytes)} for those arrays alone.</p><p>Forward/backward working storage is roughly ${formatBytes(e.workingBytes)} in addition; backend workspaces, transfers, JavaScript program banks and checkpoint serialization add more. This is an algorithmic estimate, not measured free RAM or peak GPU memory. The configured replay banks can retain up to ${e.bankTokenSlots.toLocaleString()} token slots across those states. Attention work grows quadratically with context (${e.attentionCells.toLocaleString()} attention cells per round before backward passes and historical reconstruction).</p><p>Larger settings may exhaust device memory or make rounds very slow. Neural gradients run on the selected backend; sampling, Adam, the interpreter and retained state still use the CPU. A larger model does not guarantee high GPU utilization. ${options.deviceLimits?.maxStorageBufferBindingSize?`The initialized GPU allows ${formatBytes(options.deviceLimits.maxStorageBufferBindingSize)} per storage-buffer binding and ${formatBytes(options.deviceLimits.maxBufferSize)} per buffer.`:'GPU buffer limits are checked against the actual device before models are allocated; they are not yet available for this draft.'} A single CPU Float32 array must remain below the implementation’s 2 GiB addressing bound.</p>`;}
