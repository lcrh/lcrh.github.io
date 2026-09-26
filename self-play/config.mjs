import {DEFAULTS,PRESETS} from './trainer.mjs';
export {PRESETS};
export const FIELDS=[
 ['dim','Model width',8,96,8,'transformer'],['heads','Attention heads',1,8,1,'attention'],['layers','Transformer layers',1,3,1,'transformer'],['ffMultiplier','MLP width multiplier',1,4,1,'transformer'],
 ['context','Output / learner context (bytes)',32,256,16,'context'],['programLength','Generator context (tokens)',16,128,8,'program'],['pool','Programs per round',4,48,4,'pool'],['steps','Primitive steps per program',256,65536,256,'budgets'],['memory','Tape cells',64,1024,64,'machine'],
 ['lr','Learner learning rate',.00001,.03,.00001,'adam'],['generatorLR','Generator learning rate',.00001,.03,.00001,'policy-gradient'],['beta','Prior KL strength β',0,1,.001,'kl'],['expert','Expert iteration weight',0,5,.1,'expert-iteration'],['bankSize','Replay bank capacity',32,2048,32,'replay']
];
export function validateConfig(input){
 const c={...DEFAULTS,...input};
 for(const [key,label,min,max] of FIELDS){const x=c[key];if(!Number.isFinite(x)||x<min||x>max||(!['lr','generatorLR','beta','expert'].includes(key)&&!Number.isInteger(x)))throw Error(`${label} must be ${min}–${max}${['lr','generatorLR','beta','expert'].includes(key)?'':' (whole numbers)'}.`);}
 if(c.dim%c.heads)throw Error('Model width must divide evenly into attention heads.');
 if(!Number.isSafeInteger(c.seed)||c.seed<1||c.seed>999999999)throw Error('Seed must be a whole number from 1 to 999999999.');
 if(!['selfplay','difficulty','prior'].includes(c.mode))throw Error('Unknown training objective.');return c;
}
export function parameterCount(c,vocab,context){const d=c.dim,f=d*c.ffMultiplier;return (vocab+1)*d+context*d+c.layers*(2*d+4*d*d+2*d*f)+d+vocab*d+vocab;}
export function shapeHTML(c){return `<div class="shape"><span>Token + position embeddings<br><b>width ${c.dim}</b></span><i>→</i><span><b>${c.layers} transformer layer${c.layers===1?'':'s'}</b><br>RMSNorm → causal attention: ${c.heads} × ${c.dim/c.heads}<br>residual → RMSNorm → MLP: ${c.dim} → ${c.dim*c.ffMultiplier} → ${c.dim}<br>residual</span><i>→</i><span>RMSNorm → output<br><b>19 instructions / 256 bytes</b></span></div><p>Two independent models: generator ${parameterCount(c,19,c.programLength).toLocaleString()} parameters, learner ${parameterCount(c,256,c.context).toLocaleString()}. Each round trains on ${(c.pool*c.context).toLocaleString()} learner bytes, including padding. Attention work grows quadratically with context length; larger settings can make each round much slower.</p>`;}
