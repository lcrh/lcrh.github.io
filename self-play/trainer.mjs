import {Random,ALPHABET,END,execute,mutate,textOf} from './machine.mjs';
import {Transformer} from './model.mjs';

export const DEFAULTS={seed:7,dim:24,context:64,programLength:48,pool:12,steps:4096,memory:256,lr:.0015,generatorLR:.0004,beta:.02,expert:1,bankSize:256,mode:'selfplay'};
export const PRESETS={quick:{...DEFAULTS},deep:{...DEFAULTS,dim:48,context:128,programLength:96,pool:24,steps:16384,bankSize:512,lr:.001,generatorLR:.0002}};
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
export function generatorWeights(rows,beta,expert) {
  const average=mean(rows.map(r=>r.reward)),sd=Math.sqrt(mean(rows.map(r=>(r.reward-average)**2))),total=rows.reduce((s,r)=>s+r.reward,0);
  const eligible=rows.filter(r=>r.oldLogp!==null&&r.source!=='mutation').length;
  return rows.map(r=>{
    const advantage=(r.reward-average)/(sd+1e-8)-beta*(r.logp+r.tokens.length*Math.log(ALPHABET.length));
    const ratio=r.oldLogp===null?0:Math.exp(Math.max(-20,Math.min(20,r.logp-r.oldLogp)));
    const pg=r.oldLogp!==null&&r.source!=='mutation'?ratio*advantage/eligible:0;
    const ei=total>0?expert*r.reward/total:0;
    return {pg,ei,ratio,advantage,weight:pg+ei};
  });
}
export class Trainer {
  constructor(config={}, {retainHistory=false}={}) {
    this.config={...DEFAULTS,...config};const c=this.config;
    this.random=new Random(c.seed+1009);
    this.learner=new Transformer({vocab:256,dim:c.dim,context:c.context,seed:c.seed});
    this.generator=new Transformer({vocab:ALPHABET.length,dim:c.dim,context:c.programLength,seed:c.seed+17});
    this.round=0;this.bank=[];this.archive=new Map();this.shadow=null;this.latest=null;this.totalEmitted=0;this.totalTokens=0;this.retainHistory=retainHistory;this.history=retainHistory?[this.learner.snapshot()]:null;
  }
  past() {
    const target=Math.floor(this.round/2);
    if(this.retainHistory)return this.history[target];
    if(!this.shadow)this.shadow=new Trainer(this.config);
    while(this.shadow.round<target)this.shadow.advance();
    return this.shadow.learner.snapshot();
  }
  prior() {const tokens=[];for(let j=0;j<this.config.programLength;j++){const t=this.random.int(ALPHABET.length);tokens.push(t);if(t===END)break;}return {tokens,logp:-tokens.length*Math.log(ALPHABET.length)};}
  pool() {
    const c=this.config,rows=[],available=this.bank.map((_,i)=>i);
    for(let i=0;i<c.pool;i++) {
      let sample,source='fresh';
      if(c.mode!=='prior'&&i>=Math.floor(c.pool*.75)&&available.length) {
        const at=this.random.int(available.length),entry=this.bank[available.splice(at,1)[0]];
        sample={tokens:[...entry.tokens],logp:entry.oldLogp};source='replay';
      } else if(c.mode!=='prior'&&i>=Math.floor(c.pool*.5)&&i<Math.floor(c.pool*.75)&&this.archive.size) {
        const niches=[...this.archive.values()],niche=niches[this.random.int(niches.length)],parent=niche[this.random.int(niche.length)];
        sample={tokens:mutate(parent.tokens,this.random,c.programLength),logp:null};source='mutation';
      } else sample=c.mode==='prior'?this.prior():this.generator.sample(this.random,END,c.programLength);
      const result=execute(sample.tokens,this.random,{length:c.context,steps:c.steps,memory:c.memory});
      rows.push({tokens:sample.tokens,oldLogp:sample.logp,source,...result});
    }
    return rows;
  }
  advance() {
    const c=this.config,direction=c.mode==='prior'?null:this.learner.direction(this.past(),c.lr),rows=this.pool();
    const gradient=this.learner.blocks.map(b=>new Float32Array(b.w.length));
    let emittedLoss=0,emittedCount=0,padLoss=0,padCount=0;
    for(const r of rows) {
      this.learner.zero();const cache=this.learner.forward(r.output);this.learner.backward(cache);r.loss=cache.loss/Math.LN2;
      r.reward=direction?this.learner.alignment(direction):0;
      if(c.mode==='difficulty')r.reward=cache.loss;
      for(let b=0;b<gradient.length;b++)for(let i=0;i<gradient[b].length;i++)gradient[b][i]+=this.learner.blocks[b].g[i]/rows.length;
      for(let t=0;t<c.context;t++)if(t<r.emitted){emittedLoss+=cache.tokenLoss[t]/Math.LN2;emittedCount++;}else{padLoss+=cache.tokenLoss[t]/Math.LN2;padCount++;}
      // Display forecasts are teacher-forced; the learner never sees program tokens.
      r.predictions=Array.from({length:c.context},(_,t)=>{let best=0;for(let b=1;b<256;b++)if(cache.p[t*256+b]>cache.p[t*256+best])best=b;return best;});
      r.surprise=Array.from(cache.tokenLoss).map(x=>x/Math.LN2);
    }
    if(c.mode!=='prior') {
      const caches=rows.map(r=>{const cache=this.generator.forward(r.tokens);r.logp=-cache.loss*r.tokens.length;return cache;});
      const weights=generatorWeights(rows,c.beta,c.expert);this.generator.zero();
      rows.forEach((r,i)=>{r.policy=weights[i];this.generator.backward(caches[i],weights[i].weight,false);});
      this.generator.update(c.generatorLR);
    }
    this.learner.zero();gradient.forEach((g,i)=>this.learner.blocks[i].g.set(g));this.learner.update(c.lr);
    for(const niche of this.archive.values())for(const entry of niche)entry.reward*=.97;
    for(const r of rows) {
      if(r.source!=='replay'){
        this.bank.push({tokens:[...r.tokens],oldLogp:r.oldLogp});
        if(this.bank.length>c.bankSize)this.bank.shift();
      }
      if(r.reward>0){const bucket=r.bodyLength<=8?0:r.bodyLength<=16?1:r.bodyLength<=32?2:3,key=Math.min(8,r.depth)*4+bucket;
        const niche=this.archive.get(key)??[],program=textOf(r.tokens),existing=niche.find(e=>textOf(e.tokens)===program);
        if(existing)existing.reward=r.reward;else niche.push({tokens:[...r.tokens],reward:r.reward});
        niche.sort((a,b)=>b.reward-a.reward);if(niche.length>8)niche.length=8;this.archive.set(key,niche);
      }
    }
    this.round++;this.totalTokens+=c.pool*c.context;this.totalEmitted+=emittedCount;
    if(this.retainHistory)this.history.push(this.learner.snapshot());
    const bytes=new Set(rows.flatMap(r=>Array.from(r.output.subarray(0,r.emitted))));
    this.latest={round:this.round,loss:mean(rows.map(r=>r.loss)),emittedLoss:emittedCount?emittedLoss/emittedCount:null,padLoss:padCount?padLoss/padCount:null,emittedFraction:emittedCount/(c.pool*c.context),reward:mean(rows.map(r=>r.reward)),lookback:Math.floor((this.round-1)/2),uniqueBytes:bytes.size,niches:this.archive.size,totalTokens:this.totalTokens,totalEmitted:this.totalEmitted,rows:rows.map(r=>({...r,output:Array.from(r.output)}))};
    return this.latest;
  }
  serialize() {return {version:1,config:this.config,random:this.random.state,round:this.round,learner:this.learner.serialize(),generator:this.generator.serialize(),bank:this.bank,archive:[...this.archive],totalTokens:this.totalTokens,totalEmitted:this.totalEmitted,shadow:this.shadow?.serialize()??null};}
  static restore(data) {if(data.version!==1)throw Error('Unsupported checkpoint version');const t=new Trainer(data.config);t.random.state=data.random;t.round=data.round;t.learner=Transformer.restore(data.learner);t.generator=Transformer.restore(data.generator);t.bank=data.bank;t.archive=new Map(data.archive);t.totalTokens=data.totalTokens;t.totalEmitted=data.totalEmitted;t.shadow=data.shadow?Trainer.restore(data.shadow):null;return t;}
}

// A fixed evaluation set; not passed to the generator, learner updates, or archives.
export function probes(length=64) {
  const r=new Random(90173),sets=[];
  for(let i=0;i<4;i++){
    const a=r.int(256),b=r.int(256),pattern=Array.from({length:4},()=>r.int(256));
    sets.push({name:'Repeat',bytes:Uint8Array.from({length},()=>a)});
    sets.push({name:'Cycle',bytes:Uint8Array.from({length},(_,j)=>pattern[j%4])});
    sets.push({name:'Count',bytes:Uint8Array.from({length},(_,j)=>(a+j*(2*i+1))%256)});
    sets.push({name:'Noise',bytes:Uint8Array.from({length},()=>r.int(256))});
  }
  const text=new TextEncoder().encode('Learning from programs. A small machine can print a sequence; a small model can learn to predict what comes next. ');
  sets.push({name:'Text',bytes:Uint8Array.from({length},(_,i)=>text[i%text.length])});
  return sets;
}
export function evaluate(model) {
  const groups={};for(const p of probes(model.config.context)){
    const f=model.forward(p.bytes),g=groups[p.name]??{sum:0,early:0,late:0,count:0};g.sum+=f.loss/Math.LN2;
    const width=Math.min(16,Math.floor(p.bytes.length/4));
    g.early+=mean(Array.from(f.tokenLoss.slice(0,width)))/Math.LN2;g.late+=mean(Array.from(f.tokenLoss.slice(-width)))/Math.LN2;g.count++;groups[p.name]=g;
  }
  return Object.fromEntries(Object.entries(groups).map(([key,g])=>[key,{loss:g.sum/g.count,early:g.early/g.count,late:g.late/g.count}]));
}
