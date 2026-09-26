// Appendix E: every token string executes; macros expand before bracket matching.
export const ALPHABET = '<>+-[].,ZRLNCGHWVXF';
export const END = ALPHABET.indexOf('F');
export const MACROS = {Z:'[-]',R:'[->+<]',L:'[->+++<]',N:'[-<->]',C:'[->+>+<<]',G:'[>]',H:'[<]',W:'[[-]>+<]',V:'[.>]',X:'[-]++++++++++++++++'};
export class Random {
  constructor(seed=1) {this.state=seed>>>0;}
  next() {let t=this.state=(this.state+0x6D2B79F5)>>>0;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;}
  int(n) {return Math.floor(this.next()*n);}
}
export function textOf(tokens) {return Array.from(tokens,x=>ALPHABET[x]).join('');}
export function tokensOf(text) {return Array.from(text).filter(x=>ALPHABET.includes(x)).map(x=>ALPHABET.indexOf(x));}
export function createMachine(tokens, random, {length=64, steps=4096, memory=256}={}) {
  const raw = typeof tokens==='string'? tokens : textOf(tokens);
  const body=raw.split('F')[0];
  const origins=[];const code=Array.from(body,(x,i)=>{const expanded=MACROS[x]??x;for(let j=0;j<expanded.length;j++)origins.push(i);return expanded;}).join('');
  const jumps=new Int32Array(code.length).fill(-1),stack=[];
  for(let i=0;i<code.length;i++) {if(code[i]==='[')stack.push(i);else if(code[i]===']'&&stack.length){const j=stack.pop();jumps[i]=j;jumps[j]=i;}}
  const tape=new Uint8Array(memory),output=new Uint8Array(length);
  let pointer=0,pc=0,used=0,emitted=0,depth=0,maxDepth=0;
  // Count dynamically active matched loops, including loops in macro expansions.
  const active=new Set();
  function done(){return pc>=code.length||used>=steps||emitted>=length;}
  function step() {
    if(done())return false;
    const op=code[pc];used++;
    switch(op) {
      case '>':pointer=(pointer+1)%memory;break;
      case '<':pointer=(pointer+memory-1)%memory;break;
      case '+':tape[pointer]++;break;
      case '-':tape[pointer]--;break;
      case ',':tape[pointer]=random.int(256);break;
      case '.':output[emitted++]=tape[pointer];break;
      case '[':if(jumps[pc]>=0){if(tape[pointer]===0)pc=jumps[pc];else if(!active.has(pc)){active.add(pc);depth++;maxDepth=Math.max(maxDepth,depth);}}break;
      case ']':if(jumps[pc]>=0){if(tape[pointer]!==0)pc=jumps[pc];else if(active.delete(jumps[pc]))depth--;}break;
    }
    pc++;
    return true;
  }
  function result(){return {output,emitted,steps:used,depth:maxDepth,reason:emitted===length?'output limit':used===steps?'step limit':'halted',bodyLength:body.length};}
  return {step,done,run(n=Infinity){let count=0;while(count<n&&step())count++;return result();},result,snapshot(){return {...result(),code,body,origins,tape,pointer,pc,done:done(),stepBudget:steps,outputLimit:length};}};
}
export function execute(tokens,random,limits){return createMachine(tokens,random,limits).run();}
export function mutate(tokens,random,maxLength) {
  let a=Array.from(tokens);if(a.at(-1)===END)a.pop();
  const kind=random.int(3),i=random.int(a.length+1);
  if(kind===0&&a.length)a[Math.min(i,a.length-1)]=random.int(END);
  else if(kind===1&&a.length<maxLength)a.splice(i,0,random.int(END));
  else if(a.length)a.splice(Math.min(i,a.length-1),1);
  if(a.length<maxLength)a.push(END);
  return a;
}
