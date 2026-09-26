import {textOf} from './machine.mjs';
const $=id=>document.getElementById(id);
const initialPrograms=$('programs').innerHTML;
const colors={green:'#b2f576',blue:'#78b9ff',purple:'#d3a0ef',muted:'#738395'};
let worker,state,selected=null,follow=true,probe='Repeat',ready=false,lastSave=0,saved=null,startAfterInit=false;
const format=x=>x===null||x===undefined?'—':x.toFixed(2);
const duration=ms=>ms<60000?`${Math.floor(ms/1000)}s`:`${Math.floor(ms/60000)}m ${Math.floor(ms/1000)%60}s`;
const describe={Repeat:'Four fixed, held-out constant-byte sequences. All 256 bytes remain possible predictions.',Cycle:'Four fixed cycles of four random bytes. These sequences never enter training.',Count:'Four fixed arithmetic sequences, modulo 256. Seeing a counting program is not the same as learning to predict it.',Text:'A fixed, short English passage, evaluated as raw UTF-8 bytes. A sanity check, not a language benchmark.',Noise:'Fixed independent random bytes. A well-calibrated predictor costs about 8 bits per byte; memorizing these is not allowed.'};
function attach(){
  worker?.terminate();worker=new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});ready=false;$('run').disabled=true;
  worker.onerror=e=>showError(e.message||'Training worker could not start. Serve this page over HTTP or HTTPS.');
  worker.onmessage=({data})=>{
    if(data.type==='state'){
      const wasRunning=state?.running;state=data;ready=true;$('run').disabled=false;render();
      if(startAfterInit){startAfterInit=false;worker.postMessage({type:'run'});}
      if((data.running&&data.round>0&&Date.now()-lastSave>60000)||(!data.running&&wasRunning&&data.round>0)){lastSave=Date.now();worker.postMessage({type:'export',purpose:'local'});}
    }else if(data.type==='error'){showError(data.message);$('run').disabled=false;}
    else if(data.type==='inspect'){selected=data.row;follow=false;renderInspector();}
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
  worker.postMessage({type:'init',preset:$('preset').value,seed,compare:$('compare').checked,mode:$('mode').value});
  worker.postMessage({type:'speed',value:$('speed').value});
}
function showError(message){$('error').hidden=false;$('error').textContent=message;document.body.classList.remove('running');$('status').textContent='Stopped · see error above';}
$('run').onclick=()=>{if(!ready)return;worker.postMessage({type:state.running?'pause':'run'});};
$('reset').onclick=()=>newRun(false);
for(const id of ['preset','seed','compare','mode'])$(id).onchange=()=>newRun(false);
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
  $('preset').value=c.dim===48?'deep':'quick';$('seed').value=c.seed;$('mode').value=c.mode;$('compare').checked=!!data.control;
  worker.postMessage({type:'restore',data});worker.postMessage({type:'speed',value:$('speed').value});
}
function download(data){const url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`self-play-seed-${data.trainer.config.seed}-round-${data.trainer.round}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('saved-status').textContent=`Checkpoint downloaded · round ${data.trainer.round}`;}
async function database(){return await new Promise((resolve,reject)=>{const request=indexedDB.open('self-play-v1',1);request.onupgradeneeded=()=>request.result.createObjectStore('runs');request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});}
async function saveLocal(data){try{const db=await database();await new Promise((resolve,reject)=>{const tx=db.transaction('runs','readwrite');tx.objectStore('runs').put(data,'latest');tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);});db.close();saved=data;$('restore-local').hidden=false;$('saved-status').textContent=`Autosaved locally · round ${data.trainer.round}`;}catch{$('saved-status').textContent='Autosave unavailable · use Download checkpoint';}}
async function findSaved(){try{const db=await database();saved=await new Promise((resolve,reject)=>{const r=db.transaction('runs').objectStore('runs').get('latest');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});db.close();if(saved){$('restore-local').hidden=false;$('saved-status').textContent=`Saved run available · round ${saved.trainer.round} · Run options`;}}catch{}}
document.querySelectorAll('[data-probe]').forEach(b=>b.onclick=()=>{probe=b.dataset.probe;document.querySelectorAll('[data-probe]').forEach(x=>{x.classList.toggle('selected',x===b);x.setAttribute('aria-pressed',String(x===b));});renderProbes();});
$('follow').onclick=()=>{follow=true;renderInspector();renderPrograms();};
$('execute').onclick=()=>{if(ready)worker.postMessage({type:'inspect',code:$('sandbox-code').value,seed:Number($('input-seed').value)||42});};

function render(){
  document.body.classList.toggle('running',state.running);$('run').textContent=state.running?'Pause training':state.round?'Resume training':'Start from zero';
  const s=state.latest;
  $('clock').textContent=`${state.round.toLocaleString()} ${state.round===1?'round':'rounds'} · ${duration(state.elapsed)}`;
  const phase=!state.round?'Random weights. Start a run to see the first generated programs.':state.config.mode==='difficulty'?'Difficulty reward ablation · same engine, different objective.':state.round<50?'Early lessons: watch the learner pick up zero padding.':'Learning from generated programs · inspect a row or switch probes.';
  $('status').textContent=(state.running?'● Running · ':state.round?'Paused · ':'')+phase;
  $('model-info').textContent=`${(state.parameters/1000).toFixed(1)}k parameters · 2 causal transformers · CPU worker`;
  $('loss').textContent=format(s?.loss);$('emission').textContent=s?`${Math.round(s.emittedFraction*100)}%`:'—';
  $('batch-info').textContent=s?`${s.rows.length} PROGRAMS · ${s.niches} ARCHIVE NICHES`:'WAITING FOR FIRST BATCH';
  renderPrograms();renderInspector();renderProbes();
  const smoothed=state.history.map((r,i,a)=>{const window=a.slice(Math.max(0,i-9),i+1),avg=key=>{const nums=window.map(x=>x[key]).filter(x=>x!==null);return nums.length?nums.reduce((s,x)=>s+x,0)/nums.length:null;};return {round:r.round,emittedLoss:avg('emittedLoss'),padLoss:avg('padLoss')};});
  chart($('training-chart'),[{data:smoothed.map(r=>[r.round,r.emittedLoss]),color:colors.blue},{data:smoothed.map(r=>[r.round,r.padLoss]),color:colors.muted}],{empty:'Your learning curves will appear here',minimum:8});
}
function renderPrograms(){
  const rows=state?.latest?.rows;if(!rows){$('programs').innerHTML=initialPrograms;if(state?.round){$('programs').replaceChildren();const e=document.createElement('div');e.className='empty';e.textContent='Checkpoint restored. Resume to generate the next batch.';$('programs').append(e);}return;}
  const fragment=document.createDocumentFragment();
  rows.forEach((r,i)=>{
    const button=document.createElement('button');button.className='program-row';button.classList.toggle('active',!follow&&selected===r);button.title=`${textOf(r.tokens)} · ${r.emitted}/${r.output.length} bytes emitted · ${r.source}`;
    const code=document.createElement('span');code.className='program-code';code.textContent=textOf(r.tokens);
    const label=document.createElement('span');label.className='source-label';label.textContent=`${r.source} · ${r.emitted}/${r.output.length} emitted`;code.append(label);
    const tape=document.createElement('span');tape.className='tape';tape.setAttribute('aria-label',`${r.emitted} bytes emitted, ${r.output.length-r.emitted} padding`);
    r.output.slice(0,64).forEach((value,j)=>{const byte=document.createElement('span');byte.className='byte'+(j>=r.emitted?' pad':'');byte.style.background=`hsl(${85+value*.83} 42% ${24+value/255*40}%)`;byte.title=`Byte ${j+1}: ${value}${j>=r.emitted?' (padding)':''}`;tape.append(byte);});
    const reward=document.createElement('span');reward.className='reward';reward.textContent=r.reward===0?'0':r.reward<.001?r.reward.toExponential(1):r.reward.toFixed(3);
    button.append(code,tape,reward);button.onclick=()=>{selected=r;follow=false;renderPrograms();renderInspector();};fragment.append(button);
  });
  const scroll=$('programs').scrollTop;$('programs').replaceChildren(fragment);$('programs').scrollTop=scroll;
}
function renderInspector(){
  if(follow&&state?.latest)selected=[...state.latest.rows].sort((a,b)=>b.reward-a.reward)[0];
  const r=selected;$('follow').disabled=follow;$('inspect-source').textContent=follow?'LIVE · HIGHEST REWARD':r?.source==='sandbox'?'EVALUATION ONLY':'FROZEN SELECTION';
  if(!r){$('selected-code').textContent='Start training to inspect a generated program.';$('program-meta').textContent='Every displayed byte comes from running the displayed code.';$('byte-values').textContent='—';chart($('prediction-chart'),[],{empty:'Actual output & next-byte predictions',minimum:255,byte:true});return;}
  $('selected-code').textContent=textOf(r.tokens);
  $('program-meta').textContent=`${r.source} · ${r.emitted} emitted + ${r.output.length-r.emitted} padding · ${r.steps.toLocaleString()} steps · ${r.reason} · ${format(r.loss)} bits/byte${r.reward===null?'':` · reward ${r.reward.toPrecision(3)}`}`;
  const fragment=document.createDocumentFragment();r.output.forEach((b,i)=>{const s=document.createElement('span');s.textContent=b;if(i>=r.emitted)s.className='pad';s.title=`Position ${i+1}${i>=r.emitted?' · padding':''}`;fragment.append(s);});$('byte-values').replaceChildren(fragment);
  chart($('prediction-chart'),[{data:r.output.map((b,i)=>[i+1,b]),color:colors.green},{data:r.predictions.map((b,i)=>[i+1,b]),color:colors.blue,dots:true}],{minimum:255,byte:true,padding:r.emitted+1});
}
function renderProbes(){
  if(!state)return;const e=state.evaluation,p=e?.self?.[probe];
  $('probe-start').textContent=format(state.initial?.[probe]?.loss);$('probe-now').textContent=format(p?.loss);$('probe-prior').textContent=format(e?.prior?.[probe]?.loss);
  $('probe-caption').textContent=describe[probe]+(e?` Evaluated at round ${e.round}.`:'');
  $('context-gain').textContent=p?`${format(p.early)} → ${format(p.late)} bits`:'—';
  const primary=state.evaluations.map(e=>[e.round,e.self[probe].loss]),baseline=state.evaluations.filter(e=>e.prior).map(e=>[e.round,e.prior[probe].loss]);
  chart($('probe-chart'),[{data:primary,color:colors.green},{data:baseline,color:colors.purple}],{minimum:8,reference:8,empty:'Fixed probes · evaluated every 20 rounds'});
}
function chart(canvas,series,{minimum=8,reference=null,empty='',byte=false,padding=null}={}){
  const rect=canvas.getBoundingClientRect(),w=Math.max(180,rect.width),h=rect.height||140,dpr=devicePixelRatio||1;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const left=35,right=14,top=14,bottom=24,cw=w-left-right,ch=h-top-bottom;
  const points=series.flatMap(s=>s.data??[]).filter(([x,y])=>Number.isFinite(x)&&Number.isFinite(y));
  let maxX=Math.max(1,...points.map(p=>p[0])),maxY=Math.max(minimum,...points.map(p=>p[1]));if(!byte)maxY=Math.ceil(maxY/2)*2;
  const px=x=>left+x/maxX*cw,py=y=>top+ch-y/maxY*ch;
  if(padding!==null&&padding<=maxX){ctx.fillStyle='#27313e55';ctx.fillRect(px(padding-.5),top,px(maxX)-px(padding-.5),ch);}
  ctx.font='10px ui-monospace,monospace';ctx.textAlign='right';
  for(let i=0;i<3;i++){const value=maxY*i/2,y=py(value);ctx.strokeStyle='#293542';ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-right,y);ctx.stroke();ctx.fillStyle='#8c9cac';ctx.fillText(Number.isInteger(value)?value.toString():value.toFixed(1),left-7,y+3);}
  if(reference!==null){ctx.setLineDash([3,4]);ctx.strokeStyle='#627183';ctx.beginPath();ctx.moveTo(left,py(reference));ctx.lineTo(w-right,py(reference));ctx.stroke();ctx.setLineDash([]);}
  for(const s of series){ctx.strokeStyle=s.color;ctx.fillStyle=s.color;ctx.lineWidth=1.5;ctx.beginPath();let started=false;for(const [x,y] of s.data??[]){if(!Number.isFinite(y)){started=false;continue;}if(s.dots){ctx.fillRect(px(x)-1.5,py(y)-1.5,3,3);continue;}if(!started){ctx.moveTo(px(x),py(y));started=true;}else ctx.lineTo(px(x),py(y));}if(!s.dots)ctx.stroke();if(s.data?.length===1){ctx.beginPath();ctx.arc(px(s.data[0][0]),py(s.data[0][1]),2,0,Math.PI*2);ctx.fill();}}
  ctx.fillStyle='#8c9cac';ctx.textAlign='left';ctx.fillText(byte?'byte 1':'round 0',left,h-7);ctx.textAlign='right';ctx.fillText(maxX.toLocaleString(),w-right,h-7);
  if(!points.length){ctx.fillStyle='#7e8f9f';ctx.font='11px -apple-system,sans-serif';ctx.textAlign='center';ctx.fillText(empty,left+cw/2,top+ch*.48);}
}
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(state)render();},100);});
newRun();findSaved();
