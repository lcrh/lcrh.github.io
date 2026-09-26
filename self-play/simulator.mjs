import {createMachine,Random,textOf} from './machine.mjs';
export function mountSimulator(host,onFreeze){
 host.innerHTML=`<details><summary>Step through this program’s execution</summary><div class="sim-body"><p>The instruction pointer marks the next primitive operation. Macros expand into ordinary instructions. Replay uses this row’s captured random input state.</p><div class="sim-controls"><button data-action="step">Step</button><button data-action="play">Play</button><button data-action="reset">Reset execution</button><label>Speed <select aria-label="Execution speed"><option value="2">2 steps/s</option><option value="8" selected>8 steps/s</option><option value="64">64 steps/s</option><option value="1024">1024 steps/s</option></select></label></div><p class="sim-status" role="status"></p><div class="sim-label">Source token (highlighted while its expansion runs)</div><code class="sim-source"></code><div class="sim-label">Expanded instructions · highlighted instruction executes next</div><code class="sim-code"></code><div class="sim-label">Memory tape · highlighted cell is under the data pointer</div><div class="sim-tape"></div><div class="sim-label">Output so far · decimal byte values</div><div class="sim-output"></div></div></details>`;
 const $=q=>host.querySelector(q);let row=null,machine=null,timer=null;
 function stop(){clearInterval(timer);timer=null;$('[data-action="play"]').textContent='Play';}
 function reset(){stop();if(row?.inputState!==undefined)machine=createMachine(row.tokens,new Random(row.inputState),row.limits);else machine=null;draw();}
 function draw(){const available=!!machine;for(const b of host.querySelectorAll('button'))b.disabled=!available;
 if(!machine){$('.sim-status').textContent='Select a generated program or run your own program to replay its execution.';for(const q of ['.sim-source','.sim-code','.sim-tape','.sim-output'])$(q).replaceChildren();return;}
 const s=machine.snapshot();$('.sim-status').textContent=`Captured round ${row.round} · step ${s.steps}/${s.stepBudget} · next instruction ${s.done?'none':s.pc+1} · data pointer ${s.pointer} · ${s.emitted}/${s.outputLimit} bytes emitted${s.done?' · '+s.reason:''}`;
 const mark=(target,chars,current)=>{const f=document.createDocumentFragment();Array.from(chars).forEach((c,i)=>{const e=document.createElement(i===current?'mark':'span');e.textContent=c;f.append(e);});target.replaceChildren(f);};
 mark($('.sim-source'),textOf(row.tokens),s.done?-1:s.origins[s.pc]);mark($('.sim-code'),s.code,s.done?-1:s.pc);
 const active=$('.sim-code mark');if(active){const box=$('.sim-code');box.scrollTop=Math.max(0,active.offsetTop-box.offsetTop-box.clientHeight/2);}
 const f=document.createDocumentFragment();for(let offset=-8;offset<=8;offset++){const index=(s.pointer+offset+s.tape.length)%s.tape.length,e=document.createElement('span');e.className=offset===0?'current':'';e.innerHTML=`<small>${index}</small><b>${s.tape[index]}</b>`;f.append(e);}$('.sim-tape').replaceChildren(f);
 $('.sim-output').textContent=Array.from(s.output.slice(0,s.emitted)).join(' ')||(s.done?'No bytes emitted.':'No output yet.');
 if(s.done){const equal=s.output.every((x,i)=>x===row.output[i])&&s.emitted===row.emitted;$('.sim-output').textContent+=`\n${s.outputLimit-s.emitted} zero padding bytes added after execution. ${equal?'Replay matches the selected row exactly.':'Replay differs from selected row.'}`;stop();$('[data-action="step"]').disabled=true;$('[data-action="play"]').disabled=true;}
 }
 $('[data-action="step"]').onclick=()=>{onFreeze();stop();machine?.step();draw();};
 $('[data-action="reset"]').onclick=()=>{onFreeze();reset();};
 $('[data-action="play"]').onclick=()=>{if(timer){stop();return;}onFreeze();const rate=Number($('select').value);$('[data-action="play"]').textContent='Pause';timer=setInterval(()=>{machine.run(Math.max(1,Math.round(rate/8)));draw();},rate<8?1000/rate:125);};
 $('select').onchange=stop;$('details').ontoggle=()=>{if(!$('details').open)stop();};
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
 return {setRow(r){row=r;reset();},stop};
}
