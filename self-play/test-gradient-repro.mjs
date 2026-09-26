import {createGPUBackend} from './gpu.mjs';
import {Transformer} from './model.mjs';
import {recordedGradientFixture} from './test-gradient-fixture.mjs';
const out=document.getElementById('result');
const report={status:'waiting',checks:[]};
const render=()=>{out.textContent=JSON.stringify(report,null,2);};
function manual(model,rows,kind,targets){model.zero();if(kind==='learner')model.backward(model.forward(targets));else rows.forEach(r=>model.backward(model.forward(r.tokens),r.weights.weight,false));return model.blocks.map(b=>b.g.slice());}
async function run(data){
 report.status='running';report.checks=[];render();
 let repro=data.diagnostic?.reproduction??data.reproduction;
 if(data.fixtureVersion===1){const f=recordedGradientFixture(data);repro={model:f.model.serialize(),kind:'generator',rows:f.rows.map((r,i)=>({...r,weights:f.weights[i]}))};}
 if(!repro?.model)throw Error('This file has no captured offending operation.');
 const gpu=await createGPUBackend();report.backend=gpu.name;report.originalFailure=data.diagnostic?.failure;
 async function check(label,rows,kind,targets,legacy=false){
  const model=Transformer.restore(repro.model),ref=manual(model,rows,kind,targets);model.zero();
  const item={label,kind,path:legacy?'former pooled graph: diagnostic only':'production',lengths:rows?.map(r=>r.tokens.length)??[targets.length],weights:rows?.map(r=>r.weights.weight)};
  try{
   if(legacy){
    const tf=gpu.tf,w=gpu.tensors(model);let gradients;
    try{gradients=tf.tidy(()=>tf.grads((...params)=>tf.addN(rows.map(r=>gpu.forward(model,r.tokens,params).tokenLoss.sum().mul(r.weights.weight))))(w));const values=await Promise.all(gradients.map(g=>g.data()));model.blocks.forEach((b,i)=>b.g.set(values[i]));}
    finally{tf.dispose(gradients);tf.dispose(w);}
   }else await gpu.operation(kind==='learner'?{kind,model,targets}:{kind:'generator',model,rows,weights:rows.map(r=>r.weights)});
   let maxError=0,maxMagnitude=0;
   model.blocks.forEach((b,bi)=>b.g.forEach((g,j)=>{if(!Number.isFinite(g)||!Number.isFinite(ref[bi][j]))throw Error('Nonfinite comparison');maxError=Math.max(maxError,Math.abs(g-ref[bi][j]));maxMagnitude=Math.max(maxMagnitude,Math.abs(ref[bi][j]));}));
   Object.assign(item,{finite:true,accurate:maxError<=1e-4,maxError,maxCPUGradient:maxMagnitude});
  }catch(error){Object.assign(item,{finite:false,accurate:false,error:String(error),nonfinite:error.nonfinite??null});}
  item.tensorCount=gpu.tf.memory().numTensors;report.checks.push(item);render();
 }
 for(let repeat=0;repeat<3;repeat++)await check('production full objective repeat '+repeat,repro.rows,repro.kind,repro.targets);
 if(repro.kind==='generator')for(let i=0;i<repro.rows.length;i++)await check('individual row '+i,[repro.rows[i]],'generator');
 if(repro.kind==='generator')for(let repeat=0;repeat<3;repeat++)await check('former pooled graph repeat '+repeat,repro.rows,repro.kind,repro.targets,true);
 report.status=report.checks.filter(c=>c.path==='production').every(c=>c.accurate&&c.tensorCount===0)?'production accuracy checks passed':'FAILED production accuracy checks';render();
}
document.getElementById('file').onchange=async event=>{try{await run(JSON.parse(await event.target.files[0].text()));}catch(error){report.status='failed';report.error=String(error.stack??error);render();}};
const fixture=new URLSearchParams(location.search).get('fixture');
if(fixture){const url=new URL(fixture,location.href);if(url.origin!==location.origin)throw Error('Fixture must be served from this origin.');try{await run(await(await fetch(url)).json());}catch(error){report.status='failed';report.error=String(error.stack??error);render();}}
