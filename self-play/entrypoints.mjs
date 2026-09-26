import {FIELDS} from './config.mjs';
// The interface map is independent from the article graph, and tested against it.
export const ENTRYPOINTS=[
 ['Overview','header h1','overview'],['Generator in introduction','header [data-concept="generator"]','generator'],['Predictor in introduction','header [data-concept="learner"]','learner'],['Reward in introduction','header [data-concept="gradient-reward"]','gradient-reward'],
 ['Model settings','#model-settings summary','transformer'],['Seed label','label[for="seed"]','seed'],['Fixed-prior help','.help-button','prior-control'],['Round counter','#clock','round'],['Reward objective','.options label:first-child','reward-ablations'],['Checkpoint download','#save','checkpoints'],
 ['Generator in flow','.flow .green','generator'],['Execution in flow','.flow>span:nth-of-type(2) b','machine'],['Learner in flow','.flow .blue','learner'],['Program column','.table-head>span:nth-child(1)','program'],['Output column','.table-head>span:nth-child(2)','emissions'],['Reward column','.table-head>span:nth-child(3)','gradient-reward'],['Occupied archive niches','#batch-info','archive'],['Training loss','#loss','cross-entropy'],['Emitted output percentage','#emission','emissions'],['Training-loss graph','#training-chart','diagnostics'],
 ['Held-out test heading','.probes h2','transfer'],['Probe selector instructions','.probe-instructions a','probes'],['Repeat / Cycle / Count / Text / Noise buttons','#probe-tabs','probes'],['Transfer plot','#probe-chart','transfer'],['Within-sequence comparison','.context-stat','icl'],['Selected program','#selected-code','program'],['Selected program reward','#selected-reward','gradient-reward'],['Selected execution statistics','#program-meta','budgets'],['Selected byte values','#byte-values','teacher-forcing'],['Prediction plot','#prediction-chart','teacher-forcing'],['Execution viewer','#execution summary','machine'],
 ['Reward statistics','#reward-chart','reward-statistics'],['Output percentage graph','#emission-chart','emissions'],['Program-length graph','#length-chart','program-length'],['Generator entropy graph','#entropy-chart','exploration'],['Archive coverage','#archive-coverage','archive'],['Custom evaluation heading','.custom-eval h2','probes'],['Custom sequence loss','#custom-chart','bpb'],['Program sandbox','.sandbox summary','sandbox'],['Paper differences','.under-lab>details:nth-child(2) summary','demo-differences']
].map(([label,selector,topic])=>({label,selector,topic}));
ENTRYPOINTS.push(...FIELDS.map(([key,label,,, ,topic])=>({label:'Settings: '+label,selector:'#cfg-'+key,topic})));
ENTRYPOINTS.push({label:'Device training backend',selector:'label[for="backend"]',topic:'gpu'});
export const DYNAMIC_ENTRYPOINTS=[
 {label:'Each generated source program',selector:'.program-code',topic:'program'},
 {label:'Each program’s reward',selector:'.reward',topic:'gradient-reward'},
 {label:'Each emitted byte',selector:'.byte:not(.pad)',topic:'teacher-forcing'},
 {label:'Each padded byte',selector:'.byte.pad',topic:'padding'},
 {label:'Fresh / replay / mutation source',selector:'.source-label',topic:'pool'}
];
