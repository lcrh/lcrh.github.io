import {createGPUBackend,GPUBackend} from './gpu.mjs';
import {Transformer} from './model.mjs';
import {Trainer} from './trainer.mjs';
import {recordedGradientFixture} from './test-gradient-fixture.mjs';
const out=document.querySelector('#result'),log=s=>out.textContent+='\n'+s;
const finite=(values,label)=>{for(const value of values)if(!Number.isFinite(value))throw Error(label+' contains nonfinite '+value);};
const equal=(a,b,label)=>{if(JSON.stringify(a)!==JSON.stringify(b))throw Error(label);};
try{
 const gpu=await createGPUBackend();log('Backend: '+gpu.name);
 for(const config of [{vocab:19,dim:8,context:8,heads:2},{vocab:256,dim:16,context:8,heads:4,layers:2,ffMultiplier:3}]){
  const m=new Transformer(config),targets=[1,2,3,4,5,2,3,1],f=m.forward(targets);m.backward(f);const expected=m.blocks.map(b=>b.g.slice());m.zero();const g=await gpu.operation({kind:'learner',model:m,targets});finite([g.loss,f.loss], 'Learner losses');finite(g.p,'Learner probabilities');finite(g.tokenLoss,'Learner token losses');m.blocks.forEach((b,i)=>{finite(b.g,b.name+' GPU gradient');finite(expected[i],b.name+' CPU gradient');});let worst=0;for(let b=0;b<m.blocks.length;b++)for(let i=0;i<m.blocks[b].g.length;i++)worst=Math.max(worst,Math.abs(m.blocks[b].g[i]-expected[b][i]));if(worst>1e-4||Math.abs(g.loss-f.loss)>1e-4)throw Error('GPU mismatch '+worst);log('PASS full gradients, '+config.vocab+' vocab / '+(config.layers??1)+' layers; max error '+worst);
 }
 const generator=new Transformer({vocab:19,dim:8,context:16,layers:2}),rows=[{tokens:[1,2,3,18]},{tokens:[7,7,18]},{tokens:[1,1,2,2,18]}],weights=[{weight:-.6},{weight:.35},{weight:.04}];
 rows.forEach((r,i)=>generator.backward(generator.forward(r.tokens),weights[i].weight,false));const expected=generator.blocks.map(b=>b.g.slice());generator.zero();await gpu.operation({kind:'generator',model:generator,rows,weights});generator.blocks.forEach((b,i)=>{finite(b.g,b.name+' GPU generator gradient');finite(expected[i],b.name+' CPU generator gradient');});let worst=0;generator.blocks.forEach((b,i)=>b.g.forEach((g,j)=>worst=Math.max(worst,Math.abs(g-expected[i][j]))));if(worst>1e-4)throw Error('Generator gradient mismatch '+worst);log('PASS mixed-sign generator gradients; max error '+worst);
 // Deliberately corrupt only the LAST downloaded gradient. Earlier blocks
 // must remain untouched and tolerance checks must never accept NaN.
 let caught=false;try{finite([NaN],'Injected numerical error');}catch{caught=true;}if(!caught)throw Error('Finite checker accepted NaN');
 let gradientCalls=0;const tensorsBeforeFailure=gpu.tf.memory().numTensors;
 const fakeTF=Object.assign({},gpu.tf,{grads:fn=>(...args)=>{const result=gpu.tf.grads(fn)(...args);if(++gradientCalls===2){const last=result.length-1,bad=gpu.tf.fill(result[last].shape,NaN);result[last].dispose();result[last]=bad;}return result;}}),guardedBackend=new GPUBackend(fakeTF);
 generator.blocks.forEach(b=>b.g.fill(.125));const unchanged=generator.blocks.map(b=>Array.from(b.g));caught=false;
 try{await guardedBackend.operation({kind:'generator',model:generator,rows,weights});}catch(error){if(!error.nonfinite)throw error;caught=true;}
 if(!caught||gradientCalls!==2)throw Error('Second-row nonfinite downloaded gradient was accepted');equal(generator.blocks.map(b=>Array.from(b.g)),unchanged,'Partial gradient copy after numerical failure');generator.zero();
 if(gpu.tf.memory().numTensors!==tensorsBeforeFailure)throw Error('Tensor leak after second-row gradient failure');
 log('PASS nonfinite tolerance regression and atomic GPU gradient rejection');
 const fixture=recordedGradientFixture(await(await fetch('./fixtures/gpu-gradient-round378.json')).json());
 fixture.rows.forEach((r,i)=>fixture.model.backward(fixture.model.forward(r.tokens),fixture.weights[i].weight,false));
 const fixtureExpected=fixture.model.blocks.map(b=>b.g.slice());fixture.model.zero();
 let firstFixtureGradient;
 for(let repeat=0;repeat<3;repeat++){
  await gpu.operation({kind:'generator',...fixture});let error=0;
  fixture.model.blocks.forEach((b,i)=>{finite(b.g,'Recorded GPU gradient');finite(fixtureExpected[i],'Recorded CPU gradient');b.g.forEach((g,j)=>error=Math.max(error,Math.abs(g-fixtureExpected[i][j])));});
  if(error>1e-4)throw Error('Recorded round378 gradient mismatch '+error);
  const current=fixture.model.blocks.map(b=>Array.from(b.g));if(repeat===0)firstFixtureGradient=current;else equal(current,firstFixtureGradient,'Repeated recorded gradient is not bit-identical');
  log('PASS exact round378 generator regression, repeat '+repeat+'; max error '+error);fixture.model.zero();
 }
 const config={dim:8,context:32,programLength:16,pool:4},t=new Trainer(config),reference=new Trainer(config,{retainHistory:true}),start=performance.now();
 for(let i=0;i<10;i++){
  await t.advanceWith(gpu);const it=reference.iteration(reference.learner.direction(reference.history[Math.floor(reference.round/2)],reference.config.lr));let next=it.next();while(!next.done)next=it.next(await gpu.operation(next.value));equal(t.latest,reference.latest,'Historical reward differs');equal(t.learner.serialize(),reference.learner.serialize(),'Historical learner differs');equal(t.generator.serialize(),reference.generator.serialize(),'Historical generator differs');
 }
 log('PASS 10 GPU rounds: historical reconstruction exactly matches retained checkpoints; '+((performance.now()-start)/1000).toFixed(2)+'s');
 const restored=Trainer.restore(JSON.parse(JSON.stringify(t.serialize())));await t.advanceWith(gpu);await restored.advanceWith(gpu);equal(t.serialize(),restored.serialize(),'GPU checkpoint continuation differs');log('PASS same-device GPU checkpoint continuation');
 if(gpu.tf.memory().numTensors!==0)throw Error('Leaked tensors '+gpu.tf.memory().numTensors);log('PASS zero retained tensors');log('ALL GPU CHECKS PASSED');
}catch(e){log('FAIL '+e.stack);}
