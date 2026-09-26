import {Trainer,PRESETS,evaluate} from './trainer.mjs';
import {execute,Random,tokensOf} from './machine.mjs';
let trainer,control,running=false,busy=false,speed='full',lastSent=0,elapsed=0,history=[],evaluations=[],initial=null,lastEvaluation=null;
function pack() {return {version:1,kind:'self-play-browser',trainer:trainer.serialize(),control:control?.serialize()??null,elapsed,history,evaluations,initial,lastEvaluation};}
function evaluateNow() {
  lastEvaluation={round:trainer.round,self:evaluate(trainer.learner),prior:control?evaluate(control.learner):null};
  evaluations.push(lastEvaluation);compact(evaluations,500);
}
// Preserve the whole time axis while bounding UI/export memory for long runs.
function compact(a,limit) {if(a.length>limit){const old=a.splice(0,Math.floor(limit/2));a.unshift(...old.filter((_,i)=>i%2===0));}}
function state() {return {type:'state',running,round:trainer.round,elapsed,latest:trainer.latest,prior:control?.latest,history,evaluations,initial,evaluation:lastEvaluation,parameters:trainer.learner.size+trainer.generator.size,config:trainer.config};}
async function loop() {
  if(busy||!running)return;busy=true;
  try {
    const start=performance.now(),s=trainer.advance();if(control)control.advance();elapsed+=performance.now()-start;
    history.push({round:s.round,loss:s.loss,emittedLoss:s.emittedLoss,padLoss:s.padLoss,emittedFraction:s.emittedFraction,prior:control?.latest.loss??null});compact(history,1000);
    if(trainer.round<=1||trainer.round%20===0)evaluateNow();
    if(performance.now()-lastSent>250||trainer.round===1){postMessage(state());lastSent=performance.now();}
  } catch(error){running=false;postMessage({type:'error',message:error.stack??String(error)});}
  busy=false;if(running)setTimeout(loop,speed==='gentle'?120:0);
}
onmessage=async({data})=>{
  try {
    if(data.type==='init'){
      trainer=new Trainer({...PRESETS[data.preset??'quick'],seed:data.seed,mode:data.mode??'selfplay'});
      control=data.compare?new Trainer({...trainer.config,mode:'prior'}):null;
      initial=evaluate(trainer.learner);history=[];evaluations=[];elapsed=0;evaluateNow();postMessage(state());
    } else if(data.type==='run'){running=true;postMessage(state());loop();}
    else if(data.type==='pause'){running=false;postMessage(state());}
    else if(data.type==='speed')speed=data.value;
    else if(data.type==='export'){postMessage({type:'checkpoint',data:pack(),purpose:data.purpose??'download'});}
    else if(data.type==='restore'){
      const p=data.data;if(p.kind!=='self-play-browser'||p.version!==1)throw Error('This is not a compatible Self-play checkpoint.');
      validate(p.trainer);if(p.control)validate(p.control);
      trainer=Trainer.restore(p.trainer);control=p.control?Trainer.restore(p.control):null;elapsed=p.elapsed;history=p.history;evaluations=p.evaluations;initial=p.initial;lastEvaluation=p.lastEvaluation;running=false;postMessage(state());
    } else if(data.type==='inspect'){
      const tokens=tokensOf(data.code).slice(0,trainer.config.programLength),r=execute(tokens,new Random(data.seed),{length:trainer.config.context,steps:trainer.config.steps,memory:trainer.config.memory}),f=trainer.learner.forward(r.output);
      const predictions=Array.from({length:f.n},(_,t)=>{let j=0;for(let i=1;i<256;i++)if(f.p[t*256+i]>f.p[t*256+j])j=i;return j;});
      postMessage({type:'inspect',row:{...r,output:Array.from(r.output),tokens,source:'sandbox',reward:null,loss:f.loss/Math.LN2,surprise:Array.from(f.tokenLoss,x=>x/Math.LN2),predictions}});
    }
  } catch(error){postMessage({type:'error',message:error.message});}
};
function validate(t,depth=0){
  if(depth>32||t.version!==1||!Number.isSafeInteger(t.round)||t.round<0)throw Error('Invalid checkpoint state');
  const c=t.config;
  if(![24,48].includes(c.dim)||![64,128].includes(c.context)||![48,96].includes(c.programLength)||![12,24].includes(c.pool)||!['selfplay','prior','difficulty'].includes(c.mode)||!Number.isFinite(c.lr)||c.lr<=0||c.lr>.1||!Number.isFinite(c.generatorLR)||c.generatorLR<=0||c.generatorLR>.1||!Number.isFinite(c.beta)||c.beta<0||c.beta>10||!Number.isFinite(c.expert)||c.expert<0||c.expert>10||c.steps>16384||c.steps<1||c.memory!==256||c.bankSize>512||c.bankSize<1)throw Error('Invalid model configuration');
  const reference=new Trainer(c);
  for(const name of ['learner','generator']){
    const actual=t[name],expected=reference[name];
    for(const key of ['vocab','dim','heads','context'])if(actual.config[key]!==expected.config[key])throw Error('Inconsistent model dimensions');
    if(actual.blocks.length!==expected.blocks.length||actual.step!==(name==='generator'&&c.mode==='prior'?0:t.round))throw Error('Invalid model state');
    for(let b=0;b<expected.blocks.length;b++)for(const field of ['w','m','v'])if(actual.blocks[b][field].length!==expected.blocks[b].w.length||!actual.blocks[b][field].every(x=>Number.isFinite(x)&&Math.abs(x)<1e10))throw Error('Invalid model weights');
  }
  if(t.bank.length>c.bankSize||t.archive.length>36)throw Error('Invalid program bank');
  const entry=e=>{if(!Array.isArray(e.tokens)||!e.tokens.length||e.tokens.length>c.programLength||!e.tokens.every(x=>Number.isInteger(x)&&x>=0&&x<19))throw Error('Invalid archived program');};
  t.bank.forEach(entry);for(const [,niche] of t.archive){if(niche.length>8)throw Error('Invalid archive niche');niche.forEach(entry);}
  if(t.shadow)validate(t.shadow,depth+1);
}
