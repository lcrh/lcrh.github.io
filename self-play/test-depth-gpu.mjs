import {createGPUBackend} from './gpu.mjs';
import {Trainer,DEFAULTS} from './trainer.mjs';
import {END,ALPHABET,textOf} from './machine.mjs';
import {Transformer} from './model.mjs';

const $=id=>document.getElementById(id),query=new URLSearchParams(location.search);
const kind=query.get('case')==='control'?'control':'default';
const requested=Number(query.get('rounds')??500);
const maxRounds=Number.isInteger(requested)&&requested>=1?Math.min(500,requested):500;
const config={...DEFAULTS,layers:4,...(kind==='control'?{beta:.2,expert:.25,generatorLR:.0002}:{})};
$('configuration').textContent=`Case: ${kind}. Seed ${config.seed}; width ${config.dim}; ${config.layers} layers; learner context ${config.context}; program limit ${config.programLength}; pool ${config.pool}; up to ${maxRounds} rounds.`;
const result={diagnostic:'depth-termination-webgpu-v1',case:kind,config,maxRounds,status:'initializing',checkpoints:[],meaning:{pF:'Probability of F as the first token, evaluated on current generator weights.',freshF:'Fraction of fresh generator programs consisting only of F in the last 25 rounds.',sortedTop3F:'Fraction of the three highest-reward displayed rows which are F; each round sorted separately.',weights:'Actual row PG/EI coefficients. KL coefficient is its contribution inside PG, not an extra optimizer step.',window:'All row statistics refer to at most the latest 25 completed main-trainer rounds; internal historical trainers are excluded.',history:'Unmodified advanceWith recomputes exact halfway history using the same GPU backend.',tensorCount:'Live TF tensors after a complete main round; zero is expected.'}};
let trainer,gpu,windowRows=[],paused=false,stopped=false,busy=false,lastCompleteCheckpoint,lastOperation;
const start=performance.now();
const mean=xs=>xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:null;
const fraction=(rows,predicate)=>mean(rows.map(r=>Number(predicate(r))));
const isF=r=>r.tokens.length===1&&r.tokens[0]===END;
const allZero=r=>r.output.every(x=>x===0);
const render=()=>{$('result').textContent=JSON.stringify(result,null,2);};
function badArray(a){for(let i=0;i<a.length;i++)if(!Number.isFinite(a[i]))return {index:i,value:String(a[i])};return null;}
function badModel(model,fields){for(const block of model.blocks)for(const field of fields){const bad=badArray(block[field]);if(bad)return {block:block.name,field,...bad};}return null;}
function magnitude(model,field){let max=0;for(const b of model.blocks)for(const v of b[field])max=Math.max(max,Math.abs(v));return max;}
function fail(stage,detail){result.failure={stage,mainRound:trainer?.round,operation:lastOperation,...detail};throw Error('First nonfinite value: '+stage+' '+JSON.stringify(detail));}
function captureGradientFailure(op,bad){
 const cpu=Transformer.restore(op.model.serialize());cpu.zero();
 if(op.kind==='generator')for(let i=0;i<op.rows.length;i++)cpu.backward(cpu.forward(op.rows[i].tokens),op.weights[i].weight,false);
 else cpu.backward(cpu.forward(op.targets));
 result.reproduction={model:op.model.serialize(),kind:op.kind,targets:op.targets?Array.from(op.targets):null,rows:op.kind==='generator'?lastOperation.rows:null,cpuBad:badModel(cpu,['g']),cpuGradientMax:magnitude(cpu,'g')};
 fail('GPU gradient',{...bad,cpuBad:result.reproduction.cpuBad,cpuGradientMax:result.reproduction.cpuGradientMax});
}
function installChecks(){
 const originalUpdate=Transformer.prototype.update;
 Transformer.prototype.update=function(...args){const before=badModel(this,['w','m','v','g']);if(before)fail('before Adam',before);const value=originalUpdate.apply(this,args);const after=badModel(this,['w','m','v']);if(after)fail('after Adam',after);return value;};
 const originalOperation=gpu.operation.bind(gpu);
 gpu.operation=async op=>{
  lastOperation={kind:op.kind,modelVocab:op.model.config.vocab,optimizerStep:op.model.step,weightMax:magnitude(op.model,'w'),targets:op.targets?Array.from(op.targets):null};
  const before=badModel(op.model,['w','m','v']);if(before)fail('before GPU',before);
  if(op.kind==='generator'){
   lastOperation.rows=op.rows.map((r,i)=>({program:textOf(r.tokens),tokens:Array.from(r.tokens),source:r.source,logp:r.logp,oldLogp:r.oldLogp,reward:r.reward,weights:op.weights[i]}));
   for(const row of lastOperation.rows)if(!Number.isFinite(row.logp)||!Number.isFinite(row.reward)||Object.values(row.weights).some(x=>!Number.isFinite(x)))fail('generator inputs',{row});
  }
  let out;try{out=await originalOperation(op);}catch(error){if(error.nonfinite?.stage==='GPU gradient')captureGradientFailure(op,error.nonfinite);throw error;}
  if(out){const bad=badArray(out.p)||badArray(out.tokenLoss);if(bad||!Number.isFinite(out.loss))fail('GPU forward',{bad,loss:out.loss});}
  if(op.kind==='generator'||op.kind==='learner'){
   const bad=badModel(op.model,['g']);
   if(bad){
    captureGradientFailure(op,bad);
   }
  }
  return out;
 };
}
function downloadFailure(){
 const payload={diagnostic:result,previousCompleteTrainer:lastCompleteCheckpoint};
 const blob=new Blob([JSON.stringify(payload)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='depth-gpu-first-failure.json';a.textContent='Download first-failure checkpoint and exact operation';a.id='download-failure';$('result').before(a);
 // Keep the displayed report compact; the download retains full model data.
 if(result.reproduction){result.reproductionSummary={kind:result.reproduction.kind,cpuBad:result.reproduction.cpuBad,cpuGradientMax:result.reproduction.cpuGradientMax,weightsIncludedInDownload:true};delete result.reproduction;}
}
function rowStats(rows){return {rows:rows.length,F:fraction(rows,isF),allZeroOutput:fraction(rows,allZero),noEmissions:fraction(rows,r=>r.emitted===0),meanLength:mean(rows.map(r=>r.tokens.length)),meanReward:mean(rows.map(r=>r.reward)),meanPG:mean(rows.map(r=>r.policy?.pg??0)),meanEI:mean(rows.map(r=>r.policy?.ei??0)),meanKLWithinPG:mean(rows.map(r=>r.klPG??0)),meanImportance:mean(rows.map(r=>r.policy?.ratio??0)),maxImportance:rows.length?Math.max(...rows.map(r=>r.policy?.ratio??0)):null};}
function checkpoint(){
 const p=trainer.generator.forward([END]).p;
 const rows=windowRows.flatMap(w=>w.rows);
 let entropy=0;for(const q of p)if(q>0)entropy-=q*Math.log2(q);
 const bank=trainer.bank,archive=[...trainer.archive.values()].flat();
 const diversity=entries=>({entries:entries.length,uniquePrograms:new Set(entries.map(e=>textOf(e.tokens))).size,duplicateFraction:entries.length?1-new Set(entries.map(e=>textOf(e.tokens))).size/entries.length:null,F:entries.length?entries.filter(isF).length/entries.length:null});
 const snapshot={round:trainer.round,elapsedSeconds:(performance.now()-start)/1000,windowRounds:windowRows.length,pF:p[END],BOSentropyBits:entropy,uniformEntropyBits:Math.log2(ALPHABET.length),firstTokenDistribution:Object.fromEntries(Array.from(p,(q,i)=>[ALPHABET[i],q])),all:rowStats(rows),sources:Object.fromEntries(['fresh','replay','mutation'].map(source=>[source,rowStats(rows.filter(r=>r.source===source))])),F:rowStats(rows.filter(isF)),otherAllZero:rowStats(rows.filter(r=>!isF(r)&&allZero(r))),nonzero:rowStats(rows.filter(r=>!allZero(r))),maxSameRoundZeroRewardSpread:windowRows.length?Math.max(...windowRows.map(w=>{const z=w.rows.filter(allZero).map(r=>r.reward);return z.length>1?Math.max(...z)-Math.min(...z):0;})):null,sortedTop3F:mean(windowRows.map(w=>fraction([...w.rows].sort((a,b)=>b.reward-a.reward).slice(0,3),isF))),meanEmittedFraction:mean(windowRows.map(w=>w.emittedFraction)),meanRoundReward:mean(windowRows.map(w=>w.reward)),bank:diversity(bank),archive:{...diversity(archive),niches:trainer.archive.size},tensorCount:gpu.tf.memory().numTensors,gpuPasses:gpu.passes,topCurrentPrograms:trainer.latest?[...trainer.latest.rows].sort((a,b)=>b.reward-a.reward).slice(0,6).map(r=>({program:textOf(r.tokens),source:r.source,reward:r.reward,emitted:r.emitted,allZero:allZero(r),policy:r.policy})):[]};
 result.checkpoints.push(snapshot);render();
 if(snapshot.tensorCount!==0)throw Error(`Tensor leak after round ${trainer.round}: ${snapshot.tensorCount}`);
}
function remember(latest){
 const eligible=latest.rows.filter(r=>r.oldLogp!==null&&r.source!=='mutation').length;
 // Copy diagnostic annotations only; never change trainer rows/weights/RNG.
 windowRows.push({...latest,rows:latest.rows.map(r=>({...r,klPG:r.oldLogp!==null&&r.source!=='mutation'?r.policy.ratio*(-config.beta*(r.logp+r.tokens.length*Math.log(ALPHABET.length)))/eligible:0}))});
 if(windowRows.length>25)windowRows.shift();
}
async function run(){
 if(busy||stopped)return;busy=true;
 try{
  while(!paused&&!stopped&&trainer.round<maxRounds){
   lastCompleteCheckpoint=trainer.serialize();
   const latest=await trainer.advanceWith(gpu);remember(latest);
   result.completedRounds=trainer.round;
   $('status').textContent=`${kind}: completed ${trainer.round} / ${maxRounds} rounds · ${((performance.now()-start)/1000).toFixed(1)}s`;
   if(trainer.round===1||trainer.round%25===0||trainer.round===maxRounds)checkpoint();
   // Yield between complete rounds for the pause/stop controls and rendering.
   await new Promise(resolve=>setTimeout(resolve,0));
  }
  if(stopped||trainer.round>=maxRounds){result.status=stopped?'stopped':'complete';if(result.checkpoints.at(-1)?.round!==trainer.round)checkpoint();$('pause').disabled=true;$('stop').disabled=true;$('status').textContent=`${result.status}: ${trainer.round} rounds; ${gpu.tf.memory().numTensors} live tensors.`;}
  else{result.status='paused';if(result.checkpoints.at(-1)?.round!==trainer.round)checkpoint();$('status').textContent=`Paused after round ${trainer.round}.`;}
  render();
 }catch(error){result.status='failed';result.error=String(error.stack??error);downloadFailure();$('status').textContent='Diagnostic failed: '+error.message;$('pause').disabled=true;$('stop').disabled=true;render();}
 finally{busy=false;}
}
$('pause').onclick=()=>{paused=!paused;$('pause').textContent=paused?'Resume':'Pause after current round';if(!paused){result.status='running';run();}};
$('stop').onclick=()=>{stopped=true;if(!busy){result.status='stopped';render();$('status').textContent=`Stopped after round ${trainer.round}.`;$('pause').disabled=true;$('stop').disabled=true;}};
try{gpu=await createGPUBackend();trainer=new Trainer(config);installChecks();result.backend=gpu.name;result.status='running';checkpoint();$('pause').disabled=false;$('stop').disabled=false;await run();}
catch(error){result.status='failed';result.error=String(error.stack??error);$('status').textContent='Initialization failed: '+error.message;render();}
