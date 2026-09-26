import {Transformer} from './model.mjs';
export function recordedGradientFixture(data){
 if(data.fixtureVersion!==1)throw Error('Unknown gradient fixture version');
 const model=new Transformer(data.config);
 if(data.weights.length!==model.blocks.length)throw Error('Gradient fixture block count mismatch');
 model.blocks.forEach((b,i)=>{if(data.weights[i].length!==b.w.length)throw Error('Gradient fixture shape mismatch');b.w.set(data.weights[i]);});
 return {model,rows:data.rows.map(r=>({tokens:r.tokens})),weights:data.rows.map(r=>({weight:r.weight}))};
}
