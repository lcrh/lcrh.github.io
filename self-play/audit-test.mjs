import assert from 'node:assert/strict';
import {Transformer} from './model.mjs';
import {Trainer,generatorWeights,evaluate} from './trainer.mjs';
import {Random,execute,ALPHABET,END} from './machine.mjs';
const m=new Transformer({vocab:256,dim:8,heads:2,context:12,seed:761});
const a=m.forward([3,11,7,4,22,50,55]),b=m.forward([3,11,7,4,99,0,1]);
assert.deepEqual(a.p.slice(0,5*256),b.p.slice(0,5*256));
assert.notDeepEqual(a.p.slice(5*256),b.p.slice(5*256));
console.log('Independent causality check: changing current/future targets cannot change their own or earlier predictions.');
const gen=new Transformer({vocab:19,dim:8,heads:2,context:16,seed:19});
for(let seed=1;seed<=40;seed++) {const s=gen.sample(new Random(seed),END,16);assert.ok(Math.abs(s.logp+gen.forward(s.tokens).loss*s.tokens.length)<2e-5);}
console.log('Independent 40-seed sampling likelihood check passes, including capped sequences.');
const cfg={dim:8,context:12,programLength:12,pool:8,steps:128,seed:311};
const t=new Trainer(cfg,{retainHistory:true}),initial=t.learner.snapshot();
for(let round=0;round<40;round++)t.advance();
for(let i=0;i<initial.length;i++)assert.notDeepEqual(initial[i],t.learner.blocks[i].w,t.learner.blocks[i].name);
const direction=t.learner.direction(t.history[20],cfg.lr??.0015);
let max=0;for(const block of direction)for(const value of block)max=Math.max(max,Math.abs(value));
const seq=Uint8Array.from([4,4,8,8,12,12,16,16,20,20,24,24]);
t.learner.zero();t.learner.backward(t.learner.forward(seq));const analytic=t.learner.alignment(direction)/max;
const saved=t.learner.snapshot(),epsilon=.001;
const shift=sign=>{for(let b=0;b<saved.length;b++)for(let i=0;i<saved[b].length;i++)t.learner.blocks[b].w[i]=saved[b][i]+sign*epsilon*direction[b][i]/max;};
shift(1);const plus=t.learner.forward(seq).loss;shift(-1);const minus=t.learner.forward(seq).loss;shift(0);
const numeric=Math.abs((plus-minus)/(2*epsilon));assert.ok(Math.abs(analytic-numeric)<.001,`${analytic} vs ${numeric}`);
console.log(`Independent all-parameter directional derivative matches reward: analytic=${analytic}, finite difference=${numeric}. Every parameter block changes during training.`);
for(let b=0;b<direction.length;b++)for(let i=0;i<direction[b].length;i++){
 const block=t.learner.blocks[b],expected=.0015/(Math.sqrt(block.v[i]/(1-.999**t.learner.step))+1e-8)*(t.history[20][b][i]-block.w[i]);
 assert.ok(Math.abs(direction[b][i]-expected)<=1e-5*Math.max(1,Math.abs(expected)));
}
console.log('Independent direct formula check: actual historical difference and bias-corrected Adam preconditioner cover every parameter.');
const old=JSON.stringify(t.serialize());evaluate(t.learner);t.learner.forward(execute('+[.++]F',new Random(45),{length:12}).output);assert.equal(JSON.stringify(t.serialize()),old);
const prior=new Trainer({...cfg,mode:'prior'}),fresh=new Trainer(cfg);assert.deepEqual(prior.learner.snapshot(),fresh.learner.snapshot());
const rows=[{reward:2,logp:-4,oldLogp:-5,tokens:[1,2],source:'replay'},{reward:0,logp:-2,oldLogp:-2,tokens:[2],source:'fresh'},{reward:4,logp:-6,oldLogp:null,tokens:[1,2,3],source:'mutation'}];
const ws=generatorWeights(rows,.02,1),average=2,sd=Math.sqrt(8/3);
for(let i=0;i<2;i++){
 const expected=Math.exp(rows[i].logp-rows[i].oldLogp)*((rows[i].reward-average)/(sd+1e-8)-.02*(rows[i].logp+rows[i].tokens.length*Math.log(19)))/2;
 assert.ok(Math.abs(ws[i].pg-expected)<1e-12);
}
assert.equal(ws[2].pg,0);assert.equal(ws[2].ei,4/6);
console.log('Independent objective arithmetic, identical control initialization, and evaluation isolation pass.');
