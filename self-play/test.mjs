import assert from 'node:assert/strict';
import {Random,execute,ALPHABET,END,tokensOf,MACROS} from './machine.mjs';
import {Transformer} from './model.mjs';
import {Trainer,generatorWeights,evaluate} from './trainer.mjs';
const close=(a,b,tol=1e-4)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
assert.equal(ALPHABET.length,19);
assert.deepEqual(Array.from(execute('+++[>+.<-]F',new Random(),{length:5}).output),[1,2,3,0,0]);
assert.deepEqual(Array.from(execute('+[.++]F',new Random(),{length:6}).output),[1,3,5,7,9,11]);
assert.equal(execute(']+.[F',new Random(),{length:2}).output[0],1);
assert.equal(execute('<+.F',new Random(),{length:2,memory:4}).output[0],1);
assert.equal(execute('-.[.]F',new Random(),{length:3}).output[0],255);
for(const [m,e] of Object.entries(MACROS))assert.deepEqual(execute(`+++${m}.F`,new Random()).output,execute(`+++${e}.F`,new Random()).output);
assert.equal(execute('+[]',new Random(),{steps:50}).steps,50);
assert.equal(execute('+[>+[-]<-]',new Random()).depth,2);
console.log('VM: output, loops, wrapping, unmatched brackets, macro equivalence, budgets pass');

const model=new Transformer({vocab:7,dim:8,heads:2,context:8,seed:13}),seq=[2,4,1,2,6,3];
model.zero();model.backward(model.forward(seq));
let checked=0,worst=0;
for(const b of model.blocks)for(const i of [...new Set([0,Math.floor(b.w.length*.37),b.w.length-1])]){
  const original=b.w[i],eps=.002,g=b.g[i];b.w[i]=original+eps;const plus=model.forward(seq).loss;b.w[i]=original-eps;const minus=model.forward(seq).loss;b.w[i]=original;
  const finite=(plus-minus)/(2*eps),error=Math.abs(g-finite);worst=Math.max(worst,error);assert.ok(error<.0015,`${b.name}[${i}]: analytic ${g} numeric ${finite}`);checked++;
}
console.log(`Gradients: ${checked} finite differences across every parameter block; worst error ${worst}`);
const initial=model.forward(seq).loss;
for(let i=0;i<40;i++){model.zero();model.backward(model.forward(seq));model.update(.01);}
assert.ok(model.forward(seq).loss<initial*.5);
const sampled=model.sample(new Random(42),6,8),lp=-model.forward(sampled.tokens).loss*sampled.tokens.length;
close(lp,sampled.logp,2e-5);
console.log('Network: loss decreases; cached sampling logp matches full teacher-forced likelihood');

const config={dim:8,context:12,programLength:12,pool:4,steps:128,seed:19};
const shadow=new Trainer(config),retained=new Trainer(config,{retainHistory:true});
for(let e=0;e<16;e++){
  const x=shadow.advance(),y=retained.advance();
  assert.deepEqual(x,y);
  for(let i=0;i<shadow.learner.blocks.length;i++)assert.deepEqual(shadow.learner.blocks[i].w,retained.learner.blocks[i].w);
}
console.log('Historical reward: recursive replay exactly matches retained θ_floor(e/2) at every round, all weights/rewards');
const restored=Trainer.restore(JSON.parse(JSON.stringify(shadow.serialize())));
assert.deepEqual(restored.advance(),shadow.advance());
const before=JSON.stringify(shadow.serialize());evaluate(shadow.learner);assert.equal(JSON.stringify(shadow.serialize()),before);
console.log('Checkpoint resume is bit-exact; probes leave training state untouched');

const weights=generatorWeights([{reward:1,logp:-4,oldLogp:-5,tokens:[0,1],source:'fresh'},{reward:3,logp:-3,oldLogp:null,tokens:[1],source:'mutation'},{reward:2,logp:-2,oldLogp:null,tokens:[2],source:'replay'}],.02,1);
close(weights[0].ratio,Math.E);assert.equal(weights[1].pg,0);assert.equal(weights[2].pg,0);close(weights.reduce((s,x)=>s+x.ei,0),1);
const zeros=generatorWeights([{reward:0,logp:-Math.log(19),oldLogp:-Math.log(19),tokens:[END],source:'fresh'}],.02,1);assert.equal(zeros[0].ei,0);assert.ok(Number.isFinite(zeros[0].weight));
console.log('Policy objective: original proposal likelihood, mutation exclusion (including replay), normalized EI, zero-reward safety pass');
