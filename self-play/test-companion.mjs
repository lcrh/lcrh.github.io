import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {ARTICLES,CHAPTERS,outgoing} from './knowledge-data.mjs';
import {ENTRYPOINTS,DYNAMIC_ENTRYPOINTS} from './entrypoints.mjs';
import {pathsFromEntries} from './knowledge.mjs';
import {Transformer} from './model.mjs';
import {Trainer} from './trainer.mjs';
import {createMachine,Random} from './machine.mjs';
import {validateConfig,parameterCount,resourceEstimate,MAX_TYPED_ARRAY_BYTES} from './config.mjs';
const paths=pathsFromEntries();assert.equal(paths.size,Object.keys(ARTICLES).length);
for(const [id,a] of Object.entries(ARTICLES)){for(const to of outgoing(id))assert.ok(ARTICLES[to],`${id} → ${to}`);assert.ok(CHAPTERS.some(c=>c.id===a.chapter));}
for(const e of [...ENTRYPOINTS,...DYNAMIC_ENTRYPOINTS])assert.ok(ARTICLES[e.topic]);
console.log(`Companion: ${paths.size} articles, all reachable from ${ENTRYPOINTS.length+DYNAMIC_ENTRYPOINTS.length} UI entrypoints; longest shortest path ${Math.max(...[...paths.values()].map(p=>p.length-1))} links`);
// Validate large shapes without allocating their models or claiming available RAM.
const large=validateConfig({dim:512,heads:8,layers:12,ffMultiplier:4,context:1024,programLength:512,pool:128,bankSize:8192});
assert.equal(large.dim,512);assert.equal(large.layers,12);assert.equal(large.pool,128);
validateConfig({dim:4096,heads:32,layers:64,context:512,programLength:512,pool:64});
const needs256MiB={dim:64,heads:1,layers:1,context:8192,programLength:64,pool:4};
assert.throws(()=>validateConfig(needs256MiB,{maxStorageBufferBindingSize:128*1024**2,maxBufferSize:512*1024**2}),/selected GPU device/);
assert.equal(validateConfig(needs256MiB,{maxStorageBufferBindingSize:512*1024**2,maxBufferSize:1024**3}).context,8192);
assert.throws(()=>validateConfig({context:32768,heads:1}),/single CPU array/);
for(const bad of [{dim:NaN},{layers:1.5},{context:Infinity},{dim:17,heads:2},{pool:0},{lr:0},{bankSize:Number.MAX_SAFE_INTEGER}])assert.throws(()=>validateConfig(bad));
const costs=resourceEstimate(large,{compare:true}),alone=resourceEstimate(large,{compare:false});
assert.ok(costs.longRunStateBytes>costs.initialStateBytes);assert.equal(costs.pairStateBytes,16*costs.parameters);assert.equal(costs.initialStateBytes-alone.initialStateBytes,costs.pairStateBytes);assert.ok(costs.workingBytes>0);assert.ok(MAX_TYPED_ARRAY_BYTES>128*1024**2);
assert.ok(!readFileSync(new URL('./index.html',import.meta.url),'utf8').includes('value="custom" disabled'));
console.log('Expanded config ranges, actual-device tensor limits, invalid-shape checks, and CPU/history/control resource estimates pass');
const c=validateConfig({dim:16,heads:4,layers:2,ffMultiplier:3,context:32,programLength:24,pool:4});
const t=new Trainer(c);assert.equal(t.learner.size,parameterCount(c,256,c.context));
for(let i=0;i<10;i++)t.advance();for(const r of t.latest.rows){const m=createMachine(r.tokens,new Random(r.inputState),r.limits);while(!m.done())m.step();assert.deepEqual(Array.from(m.result().output),r.output);assert.equal(m.result().emitted,r.emitted);assert.ok(Math.abs(Math.abs(r.alignment.signed)-r.reward)<1e-12);assert.ok(Math.abs(r.alignment.blocks.reduce((s,b)=>s+b.contribution,0)-r.alignment.signed)<1e-10);}
const mean=t.latest.rows.reduce((s,r)=>s+r.reward,0)/c.pool;assert.equal(mean,t.latest.reward);const sd=Math.sqrt(t.latest.rows.reduce((s,r)=>s+(r.reward-mean)**2,0)/c.pool);assert.equal(sd,t.latest.rewardStd);
const restored=Trainer.restore(JSON.parse(JSON.stringify(t.serialize())));t.advance();restored.advance();assert.deepEqual(t.serialize(),restored.serialize());
console.log('Configurable multilayer checkpoint, exact stepped replay, signed reward decomposition and mean/std diagnostics pass');
const beforeDir='/tmp/selfplay-before-diagnostics/';try{const {Trainer:Original}=await import(beforeDir+'trainer.mjs');const original=new Original(),updated=new Trainer();for(let i=0;i<12;i++){original.advance();updated.advance();}assert.ok(JSON.stringify(updated.learner.snapshot())===JSON.stringify(original.learner.snapshot()),'Original learner differs');assert.ok(JSON.stringify(updated.generator.snapshot())===JSON.stringify(original.generator.snapshot()),'Original generator differs');assert.equal(updated.random.state,original.random.state);assert.deepEqual(updated.bank,original.bank);console.log('Original CPU preset: weights, randomness and program bank remain bit-exact after added diagnostics');}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e;}
const map=`# Paper companion map\n\n${Object.keys(ARTICLES).length} articles; ${ENTRYPOINTS.length+DYNAMIC_ENTRYPOINTS.length} entrypoints. All articles are reachable; maximum shortest path is ${Math.max(...[...paths.values()].map(p=>p.length-1))} links.\n\n## Interface entrypoints\n\n| Location | Selector | First concept |\n|---|---|---|\n${[...ENTRYPOINTS,...DYNAMIC_ENTRYPOINTS].map(e=>`| ${e.label} | \`${e.selector}\` | [${e.topic}](wiki.html#article=${e.topic}) |`).join('\n')}\n\n## Paper and article map\n\n${CHAPTERS.map(c=>`### ${c.title}\n\n${Object.values(ARTICLES).filter(a=>a.chapter===c.id).map(a=>`- **${a.title}** (${a.id}); paper ${a.source}; links: ${outgoing(a.id).join(', ')}; shortest UI route: ${paths.get(a.id).join(' → ')}`).join('\n')}`).join('\n\n')}\n`;
writeFileSync(new URL('./KNOWLEDGE-MAP.md',import.meta.url),map);
