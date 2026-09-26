// Exercise the exact browser worker message protocol under Node's worker runtime.
import {Worker} from 'node:worker_threads';
import assert from 'node:assert/strict';
import {resourceEstimate} from './config.mjs';
const url=new URL('./worker.mjs',import.meta.url).href;
function makeWorker(){
  const w=new Worker(`const {parentPort}=require('node:worker_threads');globalThis.onmessage=null;globalThis.postMessage=x=>parentPort.postMessage(x);import(${JSON.stringify(url)}).then(()=>parentPort.on('message',data=>globalThis.onmessage({data})));`,{eval:true});
  w.on('error',e=>{throw e;});return w;
}
function until(w,predicate){return new Promise((resolve,reject)=>{const timeout=setTimeout(()=>{w.off('message',listen);reject(Error('worker timeout'));},30000);function listen(m){if(m.type==='error'){clearTimeout(timeout);w.off('message',listen);reject(Error(m.message));}else if(predicate(m)){clearTimeout(timeout);w.off('message',listen);resolve(m);}}w.on('message',listen);});}
async function request(w,message,predicate){const result=until(w,predicate);w.postMessage(message);return result;}
const state=m=>m.type==='state',checkpoint=m=>m.type==='checkpoint';
const a=makeWorker(),b=makeWorker();
try{
  await request(a,{type:'init',backend:'cpu',preset:'quick',seed:7,compare:true},state);
  await request(a,{type:'run'},m=>state(m)&&m.round>=3);
  await request(a,{type:'pause'},m=>state(m)&&!m.running);
  const saved=(await request(a,{type:'export'},checkpoint)).data;
  const restored=await request(b,{type:'restore',backend:'cpu',data:saved},state);
  assert.equal(restored.round,saved.trainer.round);
  const reexport=(await request(b,{type:'export'},checkpoint)).data;
  assert.deepEqual(saved,reexport);
  assert.equal(saved.backendRevision,'cpu-v1');
  const estimate=resourceEstimate(saved.trainer.config);assert.equal(estimate.gradientReductionBytes,12*estimate.generator);
  assert.equal(saved.version,2);
  for(const oldBackend of ['CPU','WebGPU']){
    const old=structuredClone(saved);old.version=1;old.backend=oldBackend;
    await assert.rejects(request(b,{type:'restore',backend:oldBackend==='CPU'?'cpu':'gpu',data:old},state),/Only version 2/);
    assert.deepEqual((await request(b,{type:'export'},checkpoint)).data,saved,'Old format rejection must leave current run intact');
  }
  for(const savedBackend of ['CPU','WebGPU'])for(const revision of [undefined,'old-revision','unknown-future']){
    const incompatible=structuredClone(saved);incompatible.backend=savedBackend;
    if(revision===undefined)delete incompatible.backendRevision;else incompatible.backendRevision=revision;
    await assert.rejects(request(b,{type:'restore',backend:savedBackend==='CPU'?'cpu':'gpu',data:incompatible},state),/numerical revision/);
    assert.deepEqual((await request(b,{type:'export'},checkpoint)).data,saved,'Incompatible revision must leave current run intact');
  }
  const unnamed=structuredClone(saved);delete unnamed.backend;
  await assert.rejects(request(b,{type:'restore',backend:'cpu',data:unnamed},state),/must name a supported numerical backend/);
  assert.deepEqual((await request(b,{type:'export'},checkpoint)).data,saved);
  console.log('Checkpoint format 2: exact CPU roundtrip; old CPU/GPU formats and missing/unknown revisions rejected without changing the run');
  await request(b,{type:'inspect',code:'+[.++]F',seed:42},m=>m.type==='inspect');
  const afterInspect=(await request(b,{type:'export'},checkpoint)).data;
  assert.deepEqual(afterInspect,reexport);
  await request(b,{type:'evaluate-custom',bytes:[65,66,67,65,66,67]},m=>m.type==='custom-evaluation');
  assert.deepEqual((await request(b,{type:'export'},checkpoint)).data,reexport);
  console.log(`Browser worker: ${saved.trainer.round}-round run with fixed-prior control saves/restores exactly; sandbox changes no training state`);
  await request(b,{type:'run'},m=>state(m)&&m.round>saved.trainer.round+1);
  await request(b,{type:'pause'},m=>state(m)&&!m.running);
  console.log('Browser worker: imported run continues training');
  await request(b,{type:'init',backend:'cpu',preset:'deep',seed:7,compare:false},state);
  const deep=await request(b,{type:'run'},m=>state(m)&&m.round>=2);
  await request(b,{type:'pause'},m=>state(m)&&!m.running);
  assert.equal(deep.latest.rows[0].output.length,128);assert.equal(deep.latest.rows[0].predictions.length,128);
  console.log(`Long-run model: ${deep.parameters} parameters; 128-byte rows and forecasts; two full training updates pass`);
}finally{await a.terminate();await b.terminate();}
