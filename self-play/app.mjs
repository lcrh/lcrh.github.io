import {textOf} from './machine.mjs';
import {FIELDS,PRESETS,validateConfig,shapeHTML} from './config.mjs';
import {initKnowledge} from './knowledge.mjs';
import {mountSimulator} from './simulator.mjs';
const $=id=>document.getElementById(id);
const initialPrograms=$('programs').innerHTML;
const colors={green:'#b2f576',blue:'#78b9ff',purple:'#d3a0ef',muted:'#738395'};
let worker,state,selected=null,follow=true,probe='Repeat',ready=false,lastSave=0,saved=null,startAfterInit=false;
let runConfig={...PRESETS.quick},simRow=null,selectedState=null;
const simulator=mountSimulator($('execution'),()=>{follow=false;$('follow').disabled=false;});
const format=x=>x===null||x===undefined?'—':x.toFixed(2);
const duration=ms=>ms<60000?`${Math.floor(ms/1000)}s`:`${Math.floor(ms/60000)}m ${Math.floor(ms/1000)%60}s`;
const describe={Repeat:'Four fixed, held-out constant-byte sequences. All 256 bytes remain possible predictions.',Cycle:'Four fixed cycles of four random bytes. These sequences never enter training.',Count:'Four fixed arithmetic sequences, modulo 256. Seeing a counting program is not the same as learning to predict it.',Text:'A fixed, short English passage, evaluated as raw UTF-8 bytes. A sanity check, not a language benchmark.',Noise:'Independent uniform bytes have an optimal expected loss of 8 bits/byte. A learner biased toward patterns can score worse than 8; this finite fixed sample can fluctuate. An increase is not a goal or an inevitable result.'};
function attach(){
  worker?.terminate();worker=new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});ready=false;$('run').disabled=true;
  worker.onerror=e=>showError(e.message||'Training worker could not start. Serve this page over HTTP or HTTPS.');
  worker.onmessage=({data})=>{
    if(data.type==='state'){
      const wasRunning=state?.running,limitsChanged=JSON.stringify(state?.gpuLimits)!==JSON.stringify(data.gpuLimits);state=data;if(limitsChanged){try{$('model-shape').innerHTML=shapeHTML(readConfig(),configOptions());}catch{}}ready=!data.fatal;$('run').disabled=!!data.fatal;render();
      if(startAfterInit){startAfterInit=false;worker.postMessage({type:'run'});}
      if((data.running&&data.round>0&&Date.now()-lastSave>60000)||(!data.running&&wasRunning&&data.round>0)){lastSave=Date.now();worker.postMessage({type:'export',purpose:'local'});}
    }else if(data.type==='error'){showError(data.message);if(data.fatal)ready=false;$('run').disabled=!ready;$('evaluate-custom').disabled=false;}
    else if(data.type==='inspect'){selected=data.row;selectedState=state;follow=false;renderInspector();$('sandbox-status').textContent='Result shown in Inside one program.';$('inspector').scrollIntoView({behavior:'smooth',block:'center'});$('inspector').focus({preventScroll:true});$('inspector').classList.remove('attention-result');requestAnimationFrame(()=>$('inspector').classList.add('attention-result'));}
    else if(data.type==='custom-evaluation'){renderCustom(data);$('evaluate-custom').disabled=false;}
    else if(data.type==='checkpoint'){
      if(data.purpose==='download')download(data.data);
      else saveLocal(data.data);
    }
  };
}
function newRun(start=false){
  attach();state=null;selected=null;follow=true;startAfterInit=start;$('error').hidden=true;
  const seed=Math.max(1,Math.min(999999999,Math.floor(Number($('seed').value)||7)));$('seed').value=seed;
  $('status').textContent='Initializing random models…';
  worker.postMessage({type:'init',preset:$('preset').value,config:runConfig,backend:$('backend').value,seed,compare:$('compare').checked,mode:$('mode').value});
  worker.postMessage({type:'speed',value:$('speed').value});
}
function showError(message){$('error').hidden=false;$('error').textContent=message;document.body.classList.remove('running');$('status').textContent='Stopped · see error above';}
$('run').onclick=()=>{if(!ready)return;worker.postMessage({type:state.running?'pause':'run'});};
$('reset').onclick=()=>newRun(false);
for(const id of ['seed','compare','mode','backend'])$(id).onchange=()=>newRun(false);
$('preset').onchange=()=>{if($('preset').value==='custom'){$('model-settings').open=true;$('config-status').textContent='Custom draft · edit the current values, then Apply. The current run is unchanged.';return;}runConfig={...PRESETS[$('preset').value]};writeConfig(runConfig);newRun(false);};
$('speed').onchange=()=>worker.postMessage({type:'speed',value:$('speed').value});
$('save').onclick=()=>{if(ready)worker.postMessage({type:'export'});};
$('load').onclick=()=>$('file').click();
$('file').onchange=async()=>{
  const file=$('file').files[0];if(!file)return;
  try{if(file.size>100*1024*1024)throw Error('Checkpoint exceeds the 100 MB browser limit.');const data=JSON.parse(await file.text());restore(data);}catch(e){showError(e.message);}
  $('file').value='';
};
$('restore-local').onclick=()=>{if(saved)restore(saved);};
function restore(data){
  if(data.kind!=='self-play-browser'||data.version!==1)throw Error('Choose a Self-play checkpoint downloaded from this playground.');
  const c=data.trainer.config;
  attach();selected=null;follow=true;startAfterInit=false;$('error').hidden=true;
  runConfig=validateConfig(c);writeConfig(runConfig);$('preset').value='custom';$('backend').value=data.backend==='WebGPU'?'webgpu':'cpu';$('seed').value=c.seed;$('mode').value=c.mode;$('compare').checked=!!data.control;
  worker.postMessage({type:'restore',data,backend:$('backend').value});worker.postMessage({type:'speed',value:$('speed').value});
}
function download(data){const url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`self-play-seed-${data.trainer.config.seed}-round-${data.trainer.round}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('saved-status').textContent=`Checkpoint downloaded · round ${data.trainer.round}`;}
async function database(){return await new Promise((resolve,reject)=>{const request=indexedDB.open('self-play-v1',1);request.onupgradeneeded=()=>request.result.createObjectStore('runs');request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});}
async function saveLocal(data){try{const db=await database();await new Promise((resolve,reject)=>{const tx=db.transaction('runs','readwrite');tx.objectStore('runs').put(data,'latest');tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);});db.close();saved=data;$('restore-local').hidden=false;$('saved-status').textContent=`Autosaved locally · round ${data.trainer.round}`;}catch{$('saved-status').textContent='Autosave unavailable · use Download checkpoint';}}
async function findSaved(){try{const db=await database();saved=await new Promise((resolve,reject)=>{const r=db.transaction('runs').objectStore('runs').get('latest');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});db.close();if(saved){$('restore-local').hidden=false;$('saved-status').textContent=`Saved run available · round ${saved.trainer.round} · Run options`;}}catch{}}
document.querySelectorAll('[data-probe]').forEach(b=>b.onclick=()=>{probe=b.dataset.probe;document.querySelectorAll('[data-probe]').forEach(x=>{x.classList.toggle('selected',x===b);x.setAttribute('aria-pressed',String(x===b));});renderProbes();});
$('follow').onclick=()=>{follow=true;renderInspector();renderPrograms();};
$('execute').onclick=()=>{if(ready){$('sandbox-status').textContent='Running… result will appear above.';worker.postMessage({type:'inspect',code:$('sandbox-code').value,seed:Number($('input-seed').value)||42});}};

function render(){
  document.body.classList.toggle('running',state.running);$('run').textContent=state.running?'Pause training':state.round?'Resume training':'Start from zero';
  const s=state.latest;
  $('clock').textContent=`${state.round.toLocaleString()} training ${state.round===1?'round':'rounds'} · ${duration(state.elapsed)}`;
  const phase=!state.round?'Random weights. Start a run to see the first generated programs.':state.config.mode==='difficulty'?'Difficulty reward ablation · same engine, different objective.':state.round<50?'Early lessons: watch the learner pick up zero padding.':'Learning from generated programs · select a program or choose a held-out test below.';
  $('status').textContent=(state.running?'● Running · ':state.round?'Paused · ':'')+phase;
  $('model-info').textContent=`${(state.parameters/1000).toFixed(1)}k parameters · 2 causal transformers · ${state.backend} training`;
  document.querySelector('.local').lastChild.textContent=` Trains on your device · ${state.backend}`;
  $('loss').textContent=format(s?.loss);$('emission').textContent=s?`${Math.round(s.emittedFraction*100)}%`:'—';
  $('batch-info').textContent=s?`${s.rows.length} PROGRAMS · ${s.niches}/36 OCCUPIED NICHES`:'WAITING FOR FIRST BATCH';
  renderPrograms();renderInspector();renderProbes();
  const smoothed=state.history.map((r,i,a)=>{const window=a.slice(Math.max(0,i-9),i+1),avg=key=>{const nums=window.map(x=>x[key]).filter(x=>x!==null);return nums.length?nums.reduce((s,x)=>s+x,0)/nums.length:null;};return {round:r.round,emittedLoss:avg('emittedLoss'),padLoss:avg('padLoss')};});
  chart($('training-chart'),[{data:smoothed.map(r=>[r.round,r.emittedLoss]),color:colors.blue},{data:smoothed.map(r=>[r.round,r.padLoss]),color:colors.muted}],{empty:'Prediction-loss curves appear after training starts',minimum:8,unit:'bits / byte ↓'});
  renderDiagnostics();document.dispatchEvent(new CustomEvent('training-view',{detail:state}));
}
function renderPrograms(){
  const rows=state?.latest?.rows;if(!rows){$('programs').innerHTML=initialPrograms;if(state?.round){$('programs').replaceChildren();const e=document.createElement('div');e.className='empty';e.textContent='Checkpoint restored. Resume to generate the next batch.';$('programs').append(e);}return;}
  const fragment=document.createDocumentFragment();
  const order=rows.map((r,i)=>({r,i}));if($('curriculum-sort').value==='reward')order.sort((a,b)=>b.r.reward-a.r.reward);
  order.forEach(({r,i})=>{
    const button=document.createElement('button');button.className='program-row';button.classList.toggle('active',!follow&&selected===r);button.dataset.rowIndex=i;
    const code=document.createElement('span');code.className='program-code';code.dataset.concept='program';code.textContent=textOf(r.tokens);
    const label=document.createElement('span');label.className='source-label';label.dataset.concept=r.source==='fresh'?'generator':r.source==='replay'?'replay':'mutation';label.textContent=`${r.source} · ${r.emitted}/${r.output.length} emitted`;code.append(label);
    const tape=document.createElement('span');tape.className='tape';tape.dataset.concept='emissions';tape.setAttribute('aria-label',`${r.emitted} bytes emitted, ${r.output.length-r.emitted} padding`);
    r.output.slice(0,64).forEach((value,j)=>{const byte=document.createElement('span');byte.className='byte'+(j>=r.emitted?' pad':'');byte.style.background=`hsl(${85+value*.83} 42% ${24+value/255*40}%)`;byte.dataset.byteIndex=j;byte.dataset.concept=j>=r.emitted?'padding':'teacher-forcing';tape.append(byte);});
    const reward=document.createElement('span');reward.className='reward';reward.dataset.concept='gradient-reward';reward.textContent=r.reward===0?'0':r.reward<.001?r.reward.toExponential(1):r.reward.toFixed(3);
    button.append(code,tape,reward);button.onclick=()=>{selected=r;selectedState=state;follow=false;renderPrograms();renderInspector();};fragment.append(button);
  });
  const scroll=$('programs').scrollTop;$('programs').replaceChildren(fragment);$('programs').scrollTop=scroll;
}
function renderInspector(){
  if(follow&&state?.latest){selected=[...state.latest.rows].sort((a,b)=>b.reward-a.reward)[0];selectedState=state;}
  const r=selected;if(r!==simRow){simRow=r;simulator.setRow(r);}document.dispatchEvent(new CustomEvent('inspect-view',{detail:{row:r,state:selectedState}}));$('follow').disabled=follow;$('inspect-source').textContent=follow?'LIVE · HIGHEST REWARD':r?.source==='sandbox'?'EVALUATION ONLY':'FROZEN SELECTION';
  if(!r){$('selected-reward').textContent='';$('selected-code').textContent='Start training to inspect a generated program.';$('program-meta').textContent='Every displayed byte comes from running the displayed code.';$('byte-values').textContent='—';chart($('prediction-chart'),[],{empty:'Actual output & next-byte predictions',minimum:255,byte:true});return;}
  $('selected-reward').textContent=r.reward===null?'Evaluation only · no training reward':`Reward ${r.reward.toPrecision(4)} · hover or click for the full calculation`;
  $('selected-code').textContent=textOf(r.tokens);
  $('program-meta').textContent=`${r.source} · ${r.emitted} emitted + ${r.output.length-r.emitted} padding · ${r.steps.toLocaleString()} steps · ${r.reason} · ${format(r.loss)} bits/byte${r.reward===null?'':` · reward ${r.reward.toPrecision(3)}`}`;
  const fragment=document.createDocumentFragment();r.output.forEach((b,i)=>{const s=document.createElement('span');s.textContent=b;if(i>=r.emitted)s.className='pad';s.title=`Position ${i+1}${i>=r.emitted?' · padding':''}`;fragment.append(s);});$('byte-values').replaceChildren(fragment);
  chart($('prediction-chart'),[{data:r.output.map((b,i)=>[i+1,b]),color:colors.green},{data:r.predictions.map((b,i)=>[i+1,b]),color:colors.blue,dots:true}],{minimum:255,byte:true,padding:r.emitted+1});
}
function renderProbes(){
  if(!state)return;const e=state.evaluation,p=e?.self?.[probe];
  $('probe-start').textContent=format(state.initial?.[probe]?.loss);$('probe-now').textContent=format(p?.loss);$('probe-prior').textContent=format(e?.prior?.[probe]?.loss);
  $('probe-caption').textContent=describe[probe]+(e?` Evaluated at round ${e.round}.`:'');
  document.querySelector('.context-caption').textContent=`First ${Math.min(16,Math.floor(state.config.context/4))} → last ${Math.min(16,Math.floor(state.config.context/4))} bytes · same fixed sequences; position/content also differ.`;
  $('context-gain').textContent=p?`${format(p.early)} → ${format(p.late)} bits/byte`:'—';
  const primary=state.evaluations.map(e=>[e.round,e.self[probe].loss]),baseline=state.evaluations.filter(e=>e.prior).map(e=>[e.round,e.prior[probe].loss]);
  chart($('probe-chart'),[{data:primary,color:colors.green},{data:baseline,color:colors.purple}],{minimum:8,reference:8,empty:'Fixed probes · evaluated every 20 rounds',unit:'bits / byte ↓'});
}
function chart(canvas,series,{minimum=8,reference=null,empty='',byte=false,padding=null,unit=byte?'byte value':'bits / byte ↓',xLabel=byte?'byte position':'training round'}={}){
  const rect=canvas.getBoundingClientRect(),w=Math.max(180,rect.width),h=rect.height||140,dpr=devicePixelRatio||1;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const left=35,right=14,top=29,bottom=36,cw=w-left-right,ch=h-top-bottom;
  const points=series.flatMap(s=>s.data??[]).filter(([x,y])=>Number.isFinite(x)&&Number.isFinite(y));
  let maxX=Math.max(1,...points.map(p=>p[0])),maxY=Math.max(minimum,...points.map(p=>p[1]));if(!byte){const scale=10**Math.floor(Math.log10(Math.max(maxY,1e-12)));maxY=Math.ceil(maxY/scale)*scale;}
  canvas._plotData={series,maxX,maxY,left,right,unit,xLabel};
  const px=x=>left+x/maxX*cw,py=y=>top+ch-y/maxY*ch;
  if(padding!==null&&padding<=maxX){ctx.fillStyle='#27313e55';ctx.fillRect(px(padding-.5),top,px(maxX)-px(padding-.5),ch);}
  ctx.font='10px ui-monospace,monospace';ctx.textAlign='right';
  for(let i=0;i<3;i++){const value=maxY*i/2,y=py(value);ctx.strokeStyle='#293542';ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-right,y);ctx.stroke();ctx.fillStyle='#8c9cac';ctx.fillText(Number.isInteger(value)?value.toString():maxY<.01?value.toExponential(1):value.toFixed(maxY<1?3:1),left-7,y+3);}
  if(reference!==null){ctx.setLineDash([3,4]);ctx.strokeStyle='#627183';ctx.beginPath();ctx.moveTo(left,py(reference));ctx.lineTo(w-right,py(reference));ctx.stroke();ctx.setLineDash([]);}
  for(const s of series){ctx.strokeStyle=s.color;ctx.fillStyle=s.color;ctx.lineWidth=1.5;ctx.beginPath();let started=false;for(const [x,y] of s.data??[]){if(!Number.isFinite(y)){started=false;continue;}if(s.dots){ctx.fillRect(px(x)-1.5,py(y)-1.5,3,3);continue;}if(!started){ctx.moveTo(px(x),py(y));started=true;}else ctx.lineTo(px(x),py(y));}if(!s.dots)ctx.stroke();if(s.data?.length===1){ctx.beginPath();ctx.arc(px(s.data[0][0]),py(s.data[0][1]),2,0,Math.PI*2);ctx.fill();}}
  ctx.fillStyle='#8c9cac';ctx.textAlign='left';ctx.fillText('0',left,h-19);ctx.textAlign='right';ctx.fillText(maxX.toLocaleString(),w-right,h-19);ctx.textAlign='center';ctx.fillText(xLabel,left+cw/2,h-4);ctx.textAlign='left';ctx.fillText(unit,left,13);
  if(!points.length){ctx.fillStyle='#7e8f9f';ctx.font='11px -apple-system,sans-serif';ctx.textAlign='center';ctx.fillText(empty,left+cw/2,top+ch*.48);}
}
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(state)render();},100);});
function configOptions(){return {compare:$('compare').checked,deviceLimits:$('backend').value==='webgpu'?state?.gpuLimits:null};}
function writeConfig(c){for(const [key] of FIELDS)$('cfg-'+key).value=c[key];$('model-shape').innerHTML=shapeHTML(c,configOptions());$('config-status').textContent='';}
function readConfig(){return validateConfig({...runConfig,...Object.fromEntries(FIELDS.map(([key])=>[key,Number($('cfg-'+key).value)]))},configOptions().deviceLimits??{});}
$('config-fields').innerHTML=FIELDS.map(([key,label,min,max,step,concept])=>`<label><span data-concept="${concept}">${label}</span><input id="cfg-${key}" type="number" min="${min}" max="${max}" step="${step}"></label>`).join('');
$('config-fields').addEventListener('input',()=>{$('preset').value='custom';try{$('model-shape').innerHTML=shapeHTML(readConfig(),configOptions());$('config-status').textContent='Custom draft · current run unchanged. Apply to create a new run.';}catch(e){$('config-status').textContent=`Custom draft is invalid: ${e.message} Current run unchanged.`;}});
$('apply-config').onclick=()=>{try{runConfig=readConfig();$('preset').value='custom';newRun(false);$('config-status').textContent='Applied · new models ready when initialization finishes.';}catch(e){$('config-status').textContent=e.message;}};
$('curriculum-sort').onchange=renderPrograms;
$('extra-diagnostics').ontoggle=renderDiagnostics;
function renderDiagnostics(){if(!state||!$('extra-diagnostics').open)return;const h=state.history,series=(key,color)=>({data:h.map(x=>[x.round,x[key]]),color});
 const band=(mean,sd)=>[{data:h.map(x=>[x.round,x[mean]]),color:colors.green},{data:h.map(x=>[x.round,x[mean]+x[sd]]),color:colors.muted},{data:h.map(x=>[x.round,Math.max(0,x[mean]-x[sd])]),color:colors.muted}];
 chart($('reward-chart'),band('reward','rewardStd'),{minimum:.001,unit:'reward (mean ± stddev)'});
 chart($('emission-chart'),[{data:h.map(x=>[x.round,100*x.emittedFraction]),color:colors.blue}],{minimum:100,unit:'emitted output (%)'});
 chart($('length-chart'),band('lengthMean','lengthStd'),{minimum:8,unit:'instruction tokens'});
 chart($('entropy-chart'),[series('generatorEntropy',colors.green)],{minimum:Math.log2(19),reference:Math.log2(19),unit:'bits / next instruction'});
 $('archive-coverage').textContent=state.latest?`${state.latest.niches}/36 archive niches occupied; ${state.latest.uniqueBytes}/256 distinct emitted byte values in this batch.`:'Archive coverage appears after training starts.';
}
$('eval-format').onchange=()=>{$('eval-help').textContent=$('eval-format').value==='text'?'Text is encoded as UTF-8. Each byte is predicted using its preceding bytes.':'Enter bytes separated by spaces or commas: 65 66 67, or 0x41 0x42 0x43. Values must be 0–255.';};
$('evaluate-custom').onclick=()=>{try{const raw=$('eval-input').value;let bytes;if($('eval-format').value==='text')bytes=Array.from(new TextEncoder().encode(raw));else{const values=raw.trim().split(/[\s,]+/);if(values.some(x=>!(/^(?:\d+|0x[0-9a-f]+)$/i.test(x))))throw Error('Use decimal integers or 0x-prefixed hex bytes, separated by spaces or commas.');bytes=values.map(Number);}if(!bytes.length||bytes.length>16384||bytes.some(x=>!Number.isInteger(x)||x<0||x>255))throw Error('Enter 1–16384 bytes, with each value from 0 to 255.');if(!ready)throw Error('Wait for the models to initialize.');$('eval-result').textContent='Evaluating after the current round finishes…';$('evaluate-custom').disabled=true;worker.postMessage({type:'evaluate-custom',bytes});}catch(e){$('eval-result').textContent=e.message;}};
function renderCustom(d){$('eval-result').textContent=`Round ${d.round} · ${d.bytes.length} bytes · Learner: ${format(d.self.loss)} bits/byte${d.prior?` · Random-program control: ${format(d.prior.loss)} bits/byte`:''}. Lower is better. ${d.bytes.length>d.context?`Scored in non-overlapping chunks of ${d.context} bytes, resetting context at each chunk; no padding is scored.`:'No training or weight updates are performed.'}`;$('custom-chart').hidden=false;chart($('custom-chart'),[{data:d.self.surprise.map((x,i)=>[i+1,x]),color:colors.green},...(d.prior?[{data:d.prior.surprise.map((x,i)=>[i+1,x]),color:colors.purple}]:[])],{minimum:8,unit:'surprise (bits / byte) ↓',xLabel:'byte position'});}
writeConfig(runConfig);initKnowledge();newRun();findSaved();
