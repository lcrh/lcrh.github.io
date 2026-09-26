import {Trainer,evaluate} from './trainer.mjs';
const rounds=+(process.argv[2]??200),seed=+(process.argv[3]??7),mode=process.argv[4]??'selfplay';
const t=new Trainer({seed,mode});const start=performance.now();
console.log(JSON.stringify({round:0,mode,seed,eval:evaluate(t.learner)}));
for(let i=0;i<rounds;i++){
  const s=t.advance();
  if(i<3||(i+1)%25===0)console.log(JSON.stringify({round:s.round,seconds:(performance.now()-start)/1000,loss:s.loss,emittedLoss:s.emittedLoss,emittedFraction:s.emittedFraction,niches:s.niches,reward:s.reward,eval:evaluate(t.learner),examples:s.rows.filter(r=>r.emitted>8).slice(0,3).map(r=>({tokens:r.tokens,output:r.output.slice(0,32),emitted:r.emitted,reward:r.reward}))}));
}
