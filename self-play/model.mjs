// A real small causal transformer, with explicit reverse-mode gradients.
// Pre-RMSNorm, multihead softmax attention, residual ReLU MLP, learned positions.
// All embeddings, attention/MLP weights, norms and output weights are trainable.
import {Random} from './machine.mjs';
function block(name,n,random,scale=0,constant=0) {
  const w=new Float32Array(n);for(let i=0;i<n;i++)w[i]=constant+(random.next()*2-1)*scale;
  return {name,w,g:new Float32Array(n),m:new Float32Array(n),v:new Float32Array(n)};
}
function rms(x,w,n,d) {
  const y=new Float32Array(x.length),inv=new Float32Array(n);
  for(let t=0;t<n;t++){let ss=0;for(let j=0;j<d;j++)ss+=x[t*d+j]**2;inv[t]=1/Math.sqrt(ss/d+1e-5);for(let j=0;j<d;j++)y[t*d+j]=x[t*d+j]*inv[t]*w[j];}
  return {y,inv};
}
function rmsBackward(x,dy,b,inv,n,d) {
  const dx=new Float32Array(x.length);
  for(let t=0;t<n;t++){let dot=0;for(let j=0;j<d;j++){let k=t*d+j;dot+=dy[k]*b.w[j]*x[k];b.g[j]+=dy[k]*x[k]*inv[t];}for(let j=0;j<d;j++){let k=t*d+j;dx[k]=inv[t]*dy[k]*b.w[j]-x[k]*inv[t]**3*dot/d;}}
  return dx;
}
function linear(x,w,n,ins,outs) {
  const y=new Float32Array(n*outs);
  for(let t=0;t<n;t++)for(let o=0;o<outs;o++){let s=0;for(let j=0;j<ins;j++)s+=x[t*ins+j]*w[o*ins+j];y[t*outs+o]=s;}
  return y;
}
function linearBackward(x,dy,b,n,ins,outs) {
  const dx=new Float32Array(n*ins);
  for(let t=0;t<n;t++)for(let o=0;o<outs;o++){const g=dy[t*outs+o],off=o*ins;for(let j=0;j<ins;j++){b.g[off+j]+=g*x[t*ins+j];dx[t*ins+j]+=g*b.w[off+j];}}
  return dx;
}
export class Transformer {
  constructor({vocab=256,dim=24,heads=2,context=64,seed=1}={}) {
    this.config={vocab,dim,heads,context,seed};this.step=0;this.lastLR=0;
    const r=new Random(seed),d=dim,f=d*2;this.blocks=[];
    const add=(name,n,scale=0,c=0)=>{const b=block(name,n,r,scale,c);this.blocks.push(b);this[name]=b;};
    add('embedding',(vocab+1)*d,.18);add('position',context*d,.025);
    add('norm1',d,0,1);add('query',d*d,Math.sqrt(1/d));add('key',d*d,Math.sqrt(1/d));add('value',d*d,Math.sqrt(1/d));add('attention',d*d,.1/Math.sqrt(d));
    add('norm2',d,0,1);add('up',f*d,Math.sqrt(2/d));add('down',d*f,.1/Math.sqrt(f));
    add('norm3',d,0,1);add('output',vocab*d,.02);add('bias',vocab);
    this.size=this.blocks.reduce((s,b)=>s+b.w.length,0);
  }
  forward(targets) {
    const {dim:d,heads:h,vocab:v}=this.config,n=targets.length,k=d/h,f=2*d;
    if(!n||n>this.config.context)throw Error('Sequence length outside model context');
    const x=new Float32Array(n*d);
    for(let t=0;t<n;t++)for(let j=0;j<d;j++)x[t*d+j]=this.embedding.w[(t?targets[t-1]:v)*d+j]+this.position.w[t*d+j];
    const a=rms(x,this.norm1.w,n,d),q=linear(a.y,this.query.w,n,d,d),key=linear(a.y,this.key.w,n,d,d),val=linear(a.y,this.value.w,n,d,d);
    const probs=new Float32Array(h*n*n),mix=new Float32Array(n*d),scale=1/Math.sqrt(k);
    for(let head=0;head<h;head++)for(let t=0;t<n;t++){
      const offset=(head*n+t)*n;let max=-Infinity;
      for(let s=0;s<=t;s++){let z=0;for(let j=0;j<k;j++)z+=q[t*d+head*k+j]*key[s*d+head*k+j];probs[offset+s]=z*scale;max=Math.max(max,z*scale);}
      let sum=0;for(let s=0;s<=t;s++){probs[offset+s]=Math.exp(probs[offset+s]-max);sum+=probs[offset+s];}
      for(let s=0;s<=t;s++){const p=probs[offset+s]/=sum;for(let j=0;j<k;j++)mix[t*d+head*k+j]+=p*val[s*d+head*k+j];}
    }
    const attended=linear(mix,this.attention.w,n,d,d),x2=new Float32Array(n*d);
    for(let i=0;i<x.length;i++)x2[i]=x[i]+attended[i];
    const b=rms(x2,this.norm2.w,n,d),u=linear(b.y,this.up.w,n,d,f),relu=Float32Array.from(u,z=>Math.max(0,z));
    const down=linear(relu,this.down.w,n,f,d),x3=new Float32Array(n*d);
    for(let i=0;i<x.length;i++)x3[i]=x2[i]+down[i];
    const c=rms(x3,this.norm3.w,n,d),p=linear(c.y,this.output.w,n,d,v);let loss=0;
    const tokenLoss=new Float32Array(n);
    for(let t=0;t<n;t++){let max=-Infinity;for(let j=0;j<v;j++){p[t*v+j]+=this.bias.w[j];max=Math.max(max,p[t*v+j]);}let sum=0;for(let j=0;j<v;j++){p[t*v+j]=Math.exp(p[t*v+j]-max);sum+=p[t*v+j];}for(let j=0;j<v;j++)p[t*v+j]/=sum;tokenLoss[t]=-Math.log(Math.max(p[t*v+targets[t]],1e-30));loss+=tokenLoss[t];}
    return {targets,n,x,a,q,key,val,probs,mix,x2,b,u,relu,x3,c,p,loss:loss/n,tokenLoss};
  }
  backward(c,weight=1,mean=true) {
    const {dim:d,heads:h,vocab:v}=this.config,{n}=c,k=d/h,f=2*d,scale=1/Math.sqrt(k);
    const dp=Float32Array.from(c.p,z=>z*weight/(mean?n:1));
    for(let t=0;t<n;t++){dp[t*v+c.targets[t]]-=weight/(mean?n:1);for(let j=0;j<v;j++)this.bias.g[j]+=dp[t*v+j];}
    const dc=linearBackward(c.c.y,dp,this.output,n,d,v);
    const dx3=rmsBackward(c.x3,dc,this.norm3,c.c.inv,n,d);
    const dr=linearBackward(c.relu,dx3,this.down,n,f,d);
    for(let i=0;i<dr.length;i++)if(c.u[i]<=0)dr[i]=0;
    const db=linearBackward(c.b.y,dr,this.up,n,d,f),dx2=rmsBackward(c.x2,db,this.norm2,c.b.inv,n,d);
    for(let i=0;i<dx2.length;i++)dx2[i]+=dx3[i];
    const dm=linearBackward(c.mix,dx2,this.attention,n,d,d),dq=new Float32Array(n*d),dk=new Float32Array(n*d),dv=new Float32Array(n*d);
    const da=new Float32Array(n);
    for(let head=0;head<h;head++)for(let t=0;t<n;t++){
      const offset=(head*n+t)*n;let weighted=0;
      for(let s=0;s<=t;s++){let z=0;for(let j=0;j<k;j++){const i=t*d+head*k+j,si=s*d+head*k+j;z+=dm[i]*c.val[si];dv[si]+=c.probs[offset+s]*dm[i];}da[s]=z;weighted+=z*c.probs[offset+s];}
      for(let s=0;s<=t;s++){const dz=c.probs[offset+s]*(da[s]-weighted)*scale;for(let j=0;j<k;j++){dq[t*d+head*k+j]+=dz*c.key[s*d+head*k+j];dk[s*d+head*k+j]+=dz*c.q[t*d+head*k+j];}}
    }
    const aq=linearBackward(c.a.y,dq,this.query,n,d,d),ak=linearBackward(c.a.y,dk,this.key,n,d,d),av=linearBackward(c.a.y,dv,this.value,n,d,d);
    for(let i=0;i<aq.length;i++)aq[i]+=ak[i]+av[i];
    const dx=rmsBackward(c.x,aq,this.norm1,c.a.inv,n,d);
    for(let t=0;t<n;t++)for(let j=0;j<d;j++){const g=dx[t*d+j]+dx2[t*d+j];this.embedding.g[(t?c.targets[t-1]:v)*d+j]+=g;this.position.g[t*d+j]+=g;}
  }
  zero() {for(const b of this.blocks)b.g.fill(0);}
  snapshot() {return this.blocks.map(b=>b.w.slice());}
  direction(past,lr) {
    const correction=1-.999**this.step;
    return this.blocks.map((b,bi)=>Float32Array.from(b.w,(w,i)=>this.step?lr/(Math.sqrt(b.v[i]/correction)+1e-8)*(past[bi][i]-w):0));
  }
  alignment(direction) {let s=0;for(let b=0;b<this.blocks.length;b++)for(let i=0;i<direction[b].length;i++)s+=this.blocks[b].g[i]*direction[b][i];return Math.abs(s);}
  update(lr,{clip=1,decay=.01}={}) {
    let ss=0;for(const b of this.blocks)for(const g of b.g)ss+=g*g;
    const norm=Math.sqrt(ss),factor=Math.min(1,clip/(norm+1e-12));this.step++;this.lastLR=lr;
    const c1=1-.9**this.step,c2=1-.999**this.step;
    for(const b of this.blocks)for(let i=0;i<b.w.length;i++){const g=b.g[i]*factor;b.m[i]=.9*b.m[i]+.1*g;b.v[i]=.999*b.v[i]+.001*g*g;b.w[i]-=lr*((b.m[i]/c1)/(Math.sqrt(b.v[i]/c2)+1e-8)+decay*b.w[i]);}
    this.zero();return norm;
  }
  // Incremental autoregressive decoding: cache keys/values for the causal prefix.
  sample(random,end,maxLength) {
    const {dim:d,heads:h,vocab:v}=this.config,k=d/h,f=d*2;
    const keys=[],values=[],tokens=[];let previous=v,logp=0;
    for(let t=0;t<maxLength;t++){
      const x=new Float32Array(d);for(let j=0;j<d;j++)x[j]=this.embedding.w[previous*d+j]+this.position.w[t*d+j];
      const a=rms(x,this.norm1.w,1,d),q=linear(a.y,this.query.w,1,d,d);keys.push(linear(a.y,this.key.w,1,d,d));values.push(linear(a.y,this.value.w,1,d,d));
      const mix=new Float32Array(d);
      for(let head=0;head<h;head++){const scores=new Float32Array(t+1);let max=-Infinity;for(let s=0;s<=t;s++){let z=0;for(let j=0;j<k;j++)z+=q[head*k+j]*keys[s][head*k+j];scores[s]=z/Math.sqrt(k);max=Math.max(max,scores[s]);}let sum=0;for(let s=0;s<=t;s++){scores[s]=Math.exp(scores[s]-max);sum+=scores[s];}for(let s=0;s<=t;s++)for(let j=0;j<k;j++)mix[head*k+j]+=scores[s]/sum*values[s][head*k+j];}
      const x2=linear(mix,this.attention.w,1,d,d);for(let j=0;j<d;j++)x2[j]+=x[j];
      const b=rms(x2,this.norm2.w,1,d),u=linear(b.y,this.up.w,1,d,f);for(let j=0;j<f;j++)u[j]=Math.max(0,u[j]);
      const x3=linear(u,this.down.w,1,f,d);for(let j=0;j<d;j++)x3[j]+=x2[j];
      const c=rms(x3,this.norm3.w,1,d),p=linear(c.y,this.output.w,1,d,v);let max=-Infinity;for(let j=0;j<v;j++){p[j]+=this.bias.w[j];max=Math.max(max,p[j]);}let sum=0;for(let j=0;j<v;j++){p[j]=Math.exp(p[j]-max);sum+=p[j];}
      let u0=random.next()*sum,chosen=v-1;for(let j=0;j<v;j++){u0-=p[j];if(u0<=0){chosen=j;break;}}
      logp+=Math.log(p[chosen]/sum);tokens.push(chosen);previous=chosen;if(chosen===end)break;
    }
    return {tokens,logp};
  }
  serialize() {return {config:this.config,step:this.step,lastLR:this.lastLR,blocks:this.blocks.map(b=>({w:Array.from(b.w),m:Array.from(b.m),v:Array.from(b.v)}))};}
  static restore(data) {const m=new Transformer(data.config);m.step=data.step;m.lastLR=data.lastLR;for(let i=0;i<m.blocks.length;i++)for(const field of ['w','m','v'])m.blocks[i][field].set(data.blocks[i][field]);return m;}
}
