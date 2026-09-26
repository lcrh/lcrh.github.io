// Full transformer forward passes and reverse-mode parameter gradients on WebGPU.
// CPU owns the interpreter, sampling cache, reward reduction, and Adam state.
export async function createGPUBackend(){
 await import('./vendor/tf-4.22.0.min.js');await import('./vendor/tf-webgpu-4.22.0.min.js');
 const tf=globalThis.tf;if(!navigator.gpu)throw Error('WebGPU is unavailable in this browser. Choose CPU explicitly, or open this page in a browser with WebGPU enabled.');
 tf.env().set('WEBGPU_CPU_FORWARD',false);await tf.setBackend('webgpu');await tf.ready();
 if(tf.getBackend()!=='webgpu')throw Error('Could not initialize WebGPU.');
 return new GPUBackend(tf);
}
export class GPUBackend{
 constructor(tf){this.tf=tf;this.name='WebGPU';this.passes=0;const limits=tf.backend()?.device?.limits;this.limits=limits?{maxStorageBufferBindingSize:limits.maxStorageBufferBindingSize,maxBufferSize:limits.maxBufferSize}:null;}
 tensors(model){const tf=this.tf,{dim:d,ffMultiplier}=model.config,f=d*ffMultiplier;return model.blocks.map(b=>{const name=b.name.replace(/^layer\d+_/,''),shape=name==='embedding'?[model.config.vocab+1,d]:name==='position'?[model.config.context,d]:['query','key','value','attention'].includes(name)?[d,d]:name==='up'?[f,d]:name==='down'?[d,f]:name==='output'?[model.config.vocab,d]:[b.w.length];return tf.tensor(b.w,shape);});}
 forward(model,targets,weights){
 const tf=this.tf,{dim:d,heads:h,vocab:v,layers}=model.config,n=targets.length,k=d/h,W=Object.fromEntries(model.blocks.map((b,i)=>[b.name,weights[i]]));
 const norm=(x,w)=>x.mul(x.square().mean(-1,true).add(1e-5).rsqrt()).mul(w);
 const linear=(x,w)=>tf.matMul(x,w,false,true);
 // One-hot projection avoids backend-specific Gather gradient behavior.
 const input=tf.oneHot(tf.tensor1d([v,...Array.from(targets).slice(0,-1)],'int32'),v+1);
 let x=tf.matMul(input,W.embedding).add(W.position.slice([0,0],[n,d]));
 const mask=tf.tensor2d(Array.from({length:n*n},(_,i)=>i%n>Math.floor(i/n)?-1e9:0),[n,n]);
 for(let layer=0;layer<layers;layer++){
  const get=name=>W[layer?'layer'+layer+'_'+name:name],a=norm(x,get('norm1'));
  const heads=y=>y.reshape([n,h,k]).transpose([1,0,2]);
  const q=heads(linear(a,get('query'))),key=heads(linear(a,get('key'))),val=heads(linear(a,get('value')));
  const attention=tf.softmax(tf.matMul(q,key,false,true).mul(1/Math.sqrt(k)).add(mask),-1);
  const mix=tf.matMul(attention,val).transpose([1,0,2]).reshape([n,d]);
  const x2=x.add(linear(mix,get('attention'))),b=norm(x2,get('norm2'));
  x=x2.add(linear(tf.relu(linear(b,get('up'))),get('down')));
 }
 const logits=linear(norm(x,W.norm3),W.output).add(W.bias),logp=tf.logSoftmax(logits),target=tf.oneHot(tf.tensor1d(Array.from(targets),'int32'),v);
 const tokenLoss=logp.mul(target).sum(-1).neg();return {p:tf.softmax(logits),tokenLoss,loss:tokenLoss.mean()};
 }
 async operation(op){
  const tf=this.tf,weights=this.tensors(op.model);let output,gradients;
  try{
   if(op.kind==='generator'){
    gradients=tf.tidy(()=>tf.grads((...w)=>tf.addN(op.rows.map((r,i)=>this.forward(op.model,r.tokens,w).tokenLoss.sum().mul(op.weights[i].weight))))(weights));
    const values=await Promise.all(gradients.map(t=>t.data()));op.model.blocks.forEach((b,i)=>b.g.set(values[i]));return;
   }
   if(op.kind==='learner'){
    gradients=tf.tidy(()=>tf.grads((...w)=>{const f=this.forward(op.model,op.targets,w);output={p:tf.keep(f.p),tokenLoss:tf.keep(f.tokenLoss),loss:tf.keep(f.loss)};return f.loss;})(weights));
   }else output=tf.tidy(()=>this.forward(op.model,op.targets,weights));
   const [p,tokenLoss,loss,...g]=await Promise.all([output.p.data(),output.tokenLoss.data(),output.loss.data(),...(gradients??[]).map(t=>t.data())]);
   if(gradients)op.model.blocks.forEach((b,i)=>b.g.set(g[i]));this.passes++;
   return {p,tokenLoss,loss:loss[0],n:op.targets.length,targets:op.targets};
  }finally{tf.dispose(weights);tf.dispose(output);tf.dispose(gradients);}
 }
}
