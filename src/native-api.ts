import {createHash,timingSafeEqual} from 'node:crypto';
import {createServer} from 'node:http';
import type {IncomingMessage,ServerResponse} from 'node:http';

export interface NativeApiOptions {
  apiKey:string;
  localBearerToken:string;
  /** Atomically reserve spend on every dispatch; throw when the run budget cannot admit it. Never use a no-op in production. */
  authorizeDispatch:(request:Readonly<Record<string,unknown>>,signal:AbortSignal)=>Promise<void>;
  workspaceId:string;
  fetch?:typeof fetch;
  timeoutMs?:number;
}
const REQUEST_BYTES=8*1024*1024;
const RESPONSE_BYTES=32*1024*1024;
const UPSTREAM='https://api.anthropic.com/v1/messages';
const loopback=new Set(['127.0.0.1','::1','::ffff:127.0.0.1']);
function error(response:ServerResponse,status:number,message:string):void {
  if(response.destroyed)return;
  if(response.headersSent){response.destroy();return;}
  response.writeHead(status,{'content-type':'application/json'}).end(JSON.stringify({error:{type:'proxy_error',message}}));
}
/** Standalone loopback-only factory. Does not load credentials, listen, retry, or inject prompts. */
export function createNativeApiProxy(options:NativeApiOptions) {
  const timeout=options.timeoutMs??300000;
  if(!options.apiKey||!options.workspaceId||!options.localBearerToken||typeof options.authorizeDispatch!=="function"||/[\r\n]/.test(options.localBearerToken)||/[\r\n]/.test(options.apiKey+options.workspaceId)||!Number.isInteger(timeout)||timeout<1||timeout>300000)throw new Error('invalid proxy configuration');
  const send=options.fetch??fetch;
  const tokenHash=createHash("sha256").update(options.localBearerToken).digest();
  return createServer((request,response)=>{void handle(request,response);});
  async function handle(request:IncomingMessage,response:ServerResponse):Promise<void>{
    if(!loopback.has(request.socket.remoteAddress??'')||request.headers.origin!==undefined){error(response,403,'local non-browser clients only');return;}
    const auth=request.headers.authorization;
    const provided=typeof auth==="string"&&auth.startsWith("Bearer ")?auth.slice(7):"";
    if(!provided||!timingSafeEqual(tokenHash,createHash("sha256").update(provided).digest())){error(response,401,"local bearer token required");return;}
    if(request.method!=='POST'||request.url!=='/v1/messages'){error(response,404,'route not found');return;}
    if(request.headers['content-type']?.split(';')[0]?.trim()!=='application/json'){error(response,415,'application/json required');return;}
    const abort=new AbortController();let timedOut=false;
    const timer=setTimeout(()=>{timedOut=true;abort.abort();error(response,504,'upstream deadline exceeded');request.destroy();},timeout);
    const disconnect=()=>{if(!response.writableEnded)abort.abort();};response.on('close',disconnect);request.on('aborted',disconnect);
    try{
      const chunks:Buffer[]=[];let size=0;
      for await(const chunk of request){size+=chunk.length;if(size>REQUEST_BYTES){error(response,413,'request too large');request.destroy();return;}chunks.push(Buffer.from(chunk));}
      const body=Buffer.concat(chunks);
      let parsed:unknown;
      try{parsed=JSON.parse(body.toString('utf8'));if(!parsed||typeof parsed!=='object'||Array.isArray(parsed))throw Error();}catch{error(response,400,'invalid JSON object');return;}
      const headers=new Headers({'content-type':'application/json','x-api-key':options.apiKey,'anthropic-workspace-id':options.workspaceId,'anthropic-version':'2023-06-01'});
      for(const name of ['anthropic-version','anthropic-beta']){const value=request.headers[name];if(value!==undefined){if(typeof value!=='string'){error(response,400,'invalid native header');return;}headers.set(name,value);}}
      try{await options.authorizeDispatch(parsed as Record<string,unknown>,abort.signal);}catch{error(response,429,"dispatch budget reservation denied");return;}
      if(abort.signal.aborted)return;
      const upstream=await send(UPSTREAM,{method:'POST',headers,body,signal:abort.signal,redirect:'error'});
      const reader=upstream.body?.getReader();const parts:Uint8Array[]=[];let received=0;
      if(reader){try{for(;;){const item=await reader.read();if(item.done)break;received+=item.value.length;if(received>RESPONSE_BYTES){await reader.cancel();abort.abort();error(response,502,'upstream response too large');return;}parts.push(item.value);}}finally{reader.releaseLock();}}
      if(abort.signal.aborted)return;
      // Fetch decodes transfer/content encoding; do not forward stale length/encoding headers.
      for(const name of ['content-type','request-id','retry-after','anthropic-ratelimit-requests-limit','anthropic-ratelimit-requests-remaining','anthropic-ratelimit-requests-reset','anthropic-ratelimit-tokens-limit','anthropic-ratelimit-tokens-remaining','anthropic-ratelimit-tokens-reset']){const value=upstream.headers.get(name);if(value!==null)response.setHeader(name,value);}
      response.writeHead(upstream.status).end(Buffer.concat(parts));
    }catch{if(!response.destroyed)error(response,timedOut?504:502,timedOut?'upstream deadline exceeded':'upstream transport failed');}
    finally{clearTimeout(timer);response.off('close',disconnect);request.off('aborted',disconnect);}
  }
}
