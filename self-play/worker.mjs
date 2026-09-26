import {Trainer,PRESETS,evaluate} from './trainer.mjs';
import {createGPUBackend} from './gpu.mjs';
import {validateConfig} from './config.mjs';
import {execute,Random,tokensOf} from './machine.mjs';
const GPU_REVISION='webgpu-rowwise-v2',CPU_REVISION='cpu-v1';
let fatal=false,backend=null,trainer,control,running=false,busy=false,speed='full',lastSent=0,elapsed=0,history=[],evaluations=[],initial=null,lastEvaluation=null;
function pack() {return {version:2,kind:'self-play-browser',backend:backend?.name??'CPU',backendRevision:backend?GPU_REVISION:CPU_REVISION,trainer:trainer.serialize(),control:control?.serialize()??null,elapsed,history,evaluations,initial,lastEvaluation};}
function evaluateNow() {
  lastEvaluation={round:trainer.round,self:evaluate(trainer.learner),prior:control?evaluate(control.learner):null};
  evaluations.push(lastEvaluation);compact(evaluations,500);
}
// Preserve the whole time axis while bounding UI/export memory for long runs.
function compact(a,limit) {if(a.length>limit){const old=a.splice(0,Math.floor(limit/2));a.unshift(...old.filter((_,i)=>i%2===0));}}
function state() {return {type:'state',running,round:trainer.round,elapsed,latest:trainer.latest,prior:control?.latest,history,evaluations,initial,evaluation:lastEvaluation,parameters:trainer.learner.size+trainer.generator.size,config:trainer.config,backend:backend?.name??'CPU',backendRevision:backend?GPU_REVISION:CPU_REVISION,gpuPasses:backend?.passes??0,gpuLimits:backend?.limits??null,fatal};}
async function loop() {
  if(busy||!running)return;if(pendingMessages){setTimeout(loop,10);return;}busy=true;
  try {
    const start=performance.now(),s=backend?await trainer.advanceWith(backend):trainer.advance();if(control){if(backend)await control.advanceWith(backend);else control.advance();}elapsed+=performance.now()-start;
    history.push({round:s.round,loss:s.loss,emittedLoss:s.emittedLoss,padLoss:s.padLoss,emittedFraction:s.emittedFraction,reward:s.reward,rewardStd:s.rewardStd,lengthMean:s.lengthMean,lengthStd:s.lengthStd,generatorEntropy:s.generatorEntropy,niches:s.niches,uniqueBytes:s.uniqueBytes,prior:control?.latest.loss??null});compact(history,1000);
    if(trainer.round<=1||trainer.round%20===0)evaluateNow();
    if(performance.now()-lastSent>250||trainer.round===1){postMessage(state());lastSent=performance.now();}
  } catch(error){running=false;fatal=true;postMessage({type:'error',fatal:true,message:'Training was interrupted inside a round. Start a new run or restore a completed checkpoint. '+(error.stack??String(error))});}
  busy=false;if(!running)postMessage(state());if(running)setTimeout(loop,speed==='gentle'?120:0);
}
let messages=Promise.resolve(),pendingMessages=0;
onmessage=({data})=>{if(data.type==='pause'){running=false;if(!busy&&trainer)postMessage(state());return;}pendingMessages++;messages=messages.then(async()=>{try{while(busy)await new Promise(r=>setTimeout(r,10));await handle(data);}finally{pendingMessages--;}});};
async function handle(data){
  try {
    if(fatal&&!['init','restore','speed'].includes(data.type))throw Error('This interrupted run cannot continue or be exported. Start a new run or restore a completed checkpoint.');
    if(data.type==='init'){
      backend=data.backend==='cpu'?null:await createGPUBackend();
      trainer=new Trainer(validateConfig({...PRESETS[data.preset??'quick'],...data.config,seed:data.seed,mode:data.mode??'selfplay'},backend?.limits??{}));
      control=data.compare?new Trainer({...trainer.config,mode:'prior'}):null;
      fatal=false;initial=evaluate(trainer.learner);history=[];evaluations=[];elapsed=0;evaluateNow();postMessage(state());
    } else if(data.type==='run'){running=true;postMessage(state());loop();}
    else if(data.type==='pause'){running=false;postMessage(state());}
    else if(data.type==='speed')speed=data.value;
    else if(data.type==='export'){postMessage({type:'checkpoint',data:pack(),purpose:data.purpose??'download'});}
    else if(data.type==='restore'){
      const p=data.data;if(p.kind!=='self-play-browser'||p.version!==2)throw Error('This checkpoint uses an unsupported format. Only version 2 checkpoints can be resumed. Start a new run.');
      const savedBackend=p.backend;
      if(!['CPU','WebGPU'].includes(savedBackend))throw Error('This checkpoint must name a supported numerical backend.');
      // Recursive replay must use the numerical schedule that created every
      // historical update, not merely a backend with the same visible name.
      if(savedBackend==='WebGPU'&&p.backendRevision!==GPU_REVISION)throw Error('This GPU checkpoint predates the corrected gradient calculation or uses an incompatible numerical revision. Old GPU gradients could be incorrect, and historical replay cannot be reconstructed with the corrected calculation. Start a new GPU run; this checkpoint cannot be resumed.');
      if(savedBackend==='CPU'&&p.backendRevision!==CPU_REVISION)throw Error('This CPU checkpoint uses an incompatible numerical revision.');
      if(savedBackend!==(data.backend==='cpu'?'CPU':'WebGPU'))throw Error('Resume a checkpoint on the backend that created it to preserve historical reconstruction.');
      backend=data.backend==='cpu'?null:await createGPUBackend();
      validate(p.trainer);if(p.control){validate(p.control);if(p.control.round!==p.trainer.round)throw Error('Checkpoint control and learner rounds do not match.');}
      trainer=Trainer.restore(p.trainer);control=p.control?Trainer.restore(p.control):null;elapsed=p.elapsed;history=p.history;evaluations=p.evaluations;initial=p.initial;lastEvaluation=p.lastEvaluation;running=false;fatal=false;postMessage(state());
    } else if(data.type==='evaluate-custom'){
      if(!Array.isArray(data.bytes)||!data.bytes.length||data.bytes.length>16384||!data.bytes.every(x=>Number.isInteger(x)&&x>=0&&x<=255))throw Error('Enter 1–16384 bytes, each from 0 to 255.');
      const score=model=>{const surprise=[],predictions=[];for(let start=0;start<data.bytes.length;start+=model.config.context){const f=model.forward(data.bytes.slice(start,start+model.config.context));surprise.push(...Array.from(f.tokenLoss,x=>x/Math.LN2));for(let t=0;t<f.n;t++){let best=0;for(let i=1;i<256;i++)if(f.p[t*256+i]>f.p[t*256+best])best=i;predictions.push(best);}}return {loss:surprise.reduce((s,x)=>s+x,0)/surprise.length,surprise,predictions};};
      postMessage({type:'custom-evaluation',round:trainer.round,context:trainer.config.context,bytes:data.bytes,self:score(trainer.learner),prior:control?score(control.learner):null});
    } else if(data.type==='inspect'){
      const tokens=tokensOf(data.code).slice(0,trainer.config.programLength),limits={length:trainer.config.context,steps:trainer.config.steps,memory:trainer.config.memory},r=execute(tokens,new Random(data.seed),limits),f=trainer.learner.forward(r.output);
      const predictions=Array.from({length:f.n},(_,t)=>{let j=0;for(let i=1;i<256;i++)if(f.p[t*256+i]>f.p[t*256+j])j=i;return j;});
      postMessage({type:'inspect',row:{...r,output:Array.from(r.output),tokens,inputState:data.seed>>>0,limits,round:trainer.round,source:'sandbox',reward:null,loss:f.loss/Math.LN2,surprise:Array.from(f.tokenLoss,x=>x/Math.LN2),predictions}});
    }
  } catch(error){postMessage({type:'error',message:error.message});}
}
function validate(t,depth=0){
  if(depth>32||t.version!==1||!Number.isSafeInteger(t.round)||t.round<0)throw Error('Invalid checkpoint state');
  const c=t.config;
  validateConfig(c,backend?.limits??{});
  const reference=new Trainer(c);
  for(const name of ['learner','generator']){
    const actual=t[name],expected=reference[name];
    for(const key of ['vocab','dim','heads','context','layers','ffMultiplier'])if((actual.config[key]??(key==='layers'?1:key==='ffMultiplier'?2:undefined))!==expected.config[key])throw Error('Inconsistent model dimensions');
    if(actual.blocks.length!==expected.blocks.length||actual.step!==(name==='generator'&&c.mode==='prior'?0:t.round))throw Error('Invalid model state');
    for(let b=0;b<expected.blocks.length;b++)for(const field of ['w','m','v'])if(actual.blocks[b][field].length!==expected.blocks[b].w.length||!actual.blocks[b][field].every(x=>Number.isFinite(x)&&Math.abs(x)<1e10))throw Error('Invalid model weights');
  }
  if(t.bank.length>c.bankSize||t.archive.length>36)throw Error('Invalid program bank');
  const entry=e=>{if(!Array.isArray(e.tokens)||!e.tokens.length||e.tokens.length>c.programLength||!e.tokens.every(x=>Number.isInteger(x)&&x>=0&&x<19))throw Error('Invalid archived program');};
  t.bank.forEach(entry);for(const [,niche] of t.archive){if(niche.length>8)throw Error('Invalid archive niche');niche.forEach(entry);}
  if(t.shadow)validate(t.shadow,depth+1);
}
