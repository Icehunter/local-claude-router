import {expect,it} from 'vitest';
import {translateRequest,toChatCompletion} from '../src/openai-http.js';
import {AnthropicClient} from '../src/anthropic/client.js';
const defaults={defaultModel:'claude-sonnet-5-5',defaultMaxTokens:100};
const req=(extra:Record<string,unknown>={})=>translateRequest({messages:[{role:'user',content:'test'}],...extra},defaults);
it.each([{max_tokens:10,max_completion_tokens:11},{temperature:0},{top_p:1},{stop:['x']},{seed:1},{response_format:{type:'json_schema'}},{logprobs:true}])('rejects unsupported requested treatment %j',extra=>expect(()=>req(extra)).toThrow());
it('passes cache markers through actual upstream serialization',async()=>{
 const marker={type:'ephemeral',ttl:'5m'};const r=req({cache_control:marker,messages:[{role:'system',content:[{type:'text',text:'system',cache_control:marker}]},{role:'user',content:[{type:'text',text:'test',cache_control:marker}]}]});let body:any;
 const client=new AnthropicClient({credentials:{get:async()=>({accessToken:'test-token',accountKind:'claude-pro'} as never)} as never,fetch:async(_url,init)=>{body=JSON.parse(String(init?.body));return new Response(JSON.stringify({id:'x',type:'message',role:'assistant',model:'claude-sonnet-5-5',content:[{type:'text',text:'ok'}],stop_reason:'end_turn'}));}});
 await client.createMessage(r,undefined,{singleAttempt:true});expect(body.cache_control).toEqual(marker);expect(body.messages[0].content[0].cache_control).toEqual(marker);expect(body.system.at(-1).cache_control).toEqual(marker);
});
it('forwards the requested model through actual upstream serialization',async()=>{
 let body:any;
 const client=new AnthropicClient({credentials:{get:async()=>({accessToken:'test-token',accountKind:'claude-pro'} as never)} as never,fetch:async(_url,init)=>{body=JSON.parse(String(init?.body));return new Response(JSON.stringify({id:'x',type:'message',role:'assistant',model:'claude-haiku-4-5',content:[{type:'text',text:'ok'}],stop_reason:'end_turn'}));}});
 await client.createMessage(req({model:'claude-haiku-4-5'}),undefined,{singleAttempt:true});
 expect(body.model).toBe('claude-haiku-4-5');
});

it.each([{type:'ephemeral',ttl:'1h'},{type:'invalid'},{type:'ephemeral',extra:true}])('rejects unsupported cache shape %j',cache_control=>expect(()=>req({cache_control})).toThrow());
const result=(stop_reason:string|null='end_turn',usage:any={input_tokens:10,output_tokens:3,cache_creation_input_tokens:20,cache_read_input_tokens:30},content:any[]=[{type:'text',text:'ok'}])=>({requestId:'rid',response:{id:'id',type:'message',role:'assistant',model:'claude-sonnet-5-5',stop_reason,usage,content}});
it('preserves raw metadata and accounts for all input tokens',()=>{const raw=result();const c=toChatCompletion(raw,1) as any;expect(c.usage).toMatchObject({prompt_tokens:60,completion_tokens:3,total_tokens:63});expect(c.upstream).toEqual({request_id:'rid',stop_reason:'end_turn',block_types:['text'],usage:raw.response.usage});});
it.each(['refusal','pause_turn','tool_use',null,'other'])('does not report %s as normal stop',reason=>expect(toChatCompletion(result(reason),1).choices[0].finish_reason).not.toBe('stop'));
it('preserves missing usage as unknown',()=>{const raw=result();delete (raw.response as any).usage;expect(toChatCompletion(raw,1).usage).toBeNull();});
it('marks unexpected tool blocks even with end_turn',()=>{expect(toChatCompletion(result('end_turn',{},[{type:'tool_use',id:'x',name:'foo',input:{}}]),1).choices[0].finish_reason).not.toBe('stop');});
it('maps nonstrict response schema to one native respond tool without changing schema',()=>{
 const schema={oneOf:[{type:'object',properties:{x:{type:'number',minimum:2}},required:['x']}]};const r=req({response_format:{type:'json_schema',json_schema:{name:'out',strict:false,schema}}});
 expect(r.tools).toEqual([{name:'respond',description:'Return the requested JSON result.',input_schema:{type:'object',properties:{result:schema},required:['result'],additionalProperties:false}}]);expect(r.toolChoice).toEqual({type:'auto'});expect(r.responseTool).toBe('respond');expect(r.system).toContain('respond');
});
it.each([null,{}, {type:'json_object'},{type:'json_schema',json_schema:{name:'x',strict:true,schema:{}}},{type:'json_schema',json_schema:{name:'x',schema:null}}])('rejects malformed or strict format %j',response_format=>expect(()=>req({response_format})).toThrow());
it('maps only the expected tool input to JSON, retaining raw metadata',()=>{
 const c=toChatCompletion(result('tool_use',undefined,[{type:'tool_use',name:'respond',input:{action:'inspect'},id:'t'}]),1,'respond');expect(c.choices[0].message.content).toBe('{"action":"inspect"}');expect(c.choices[0].finish_reason).toBe('stop');expect(c.upstream?.stop_reason).toBe('tool_use');
});
it.each([[{type:'text',text:'{"action":"inspect"}'}],[{type:'tool_use',name:'other',input:{}}],[{type:'tool_use',name:'respond',input:{}},{type:'tool_use',name:'respond',input:{}}]])('does not silently fallback when expected response tool is missing or ambiguous',(...content)=>expect(toChatCompletion(result('end_turn',undefined,content),1,'respond').choices[0].finish_reason).toBe('unsupported_content'));

it('serializes native response tool schema verbatim without strict grammar or forced choice',async()=>{
 const schema={type:'object',properties:{n:{type:'number',minimum:2}}};let body:any;
 const r=req({response_format:{type:'json_schema',json_schema:{name:'out',strict:false,schema}}});
 const client=new AnthropicClient({credentials:{get:async()=>({accessToken:'test-token',accountKind:'claude-pro'} as never)} as never,fetch:async(_url,init)=>{body=JSON.parse(String(init?.body));return new Response(JSON.stringify({id:'x',type:'message',role:'assistant',model:'claude-sonnet-5-5',content:[],stop_reason:'end_turn'}));}});
 await client.createMessage({...r,effort:'medium'},undefined,{singleAttempt:true});expect(body.tools[0].input_schema).toEqual(schema);expect(body.tool_choice).toEqual({type:'auto'});expect(body.output_config).toEqual({effort:'medium'});expect(body.tools[0]).not.toHaveProperty('strict');
});
it('bounds response schema depth and bytes without loosening constraints',()=>{let schema:any={type:'string'};for(let i=0;i<70;i++)schema={items:schema};expect(()=>req({response_format:{type:'json_schema',json_schema:{name:'out',schema}}})).toThrow();expect(()=>req({response_format:{type:'json_schema',json_schema:{name:'out',schema:{description:'x'.repeat(256001)}}}})).toThrow();});
it('adds root object type only for unions proven object without modifying branches or source',()=>{
 const schema={oneOf:[{type:'object',properties:{n:{type:'number',minimum:2}}},{type:'object',properties:{text:{type:'string',maxLength:8}}}]};const before=JSON.stringify(schema);
 const r=req({response_format:{type:'json_schema',json_schema:{name:'out',strict:false,schema}}});expect(r.tools![0]!.input_schema).toEqual({type:'object',properties:{result:schema},required:['result'],additionalProperties:false});expect(JSON.stringify(schema)).toBe(before);
});
it.each([{oneOf:[{type:'object'},{type:'string'}]},{anyOf:[{type:'object'},{}]},{oneOf:[]},{properties:{x:{type:'string'}}},{type:'array',items:{type:'string'}}])('rejects root schema not proven object %j',schema=>expect(()=>req({response_format:{type:'json_schema',json_schema:{name:'out',strict:false,schema}}})).toThrow());
it('envelopes root unions and unwraps result before caller validation',()=>{
 const schema={oneOf:[{type:'object',properties:{action:{const:'inspect'}}},{type:'object',properties:{action:{const:'finish'}}}]};
 const r=req({response_format:{type:'json_schema',json_schema:{name:'out',strict:false,schema}}});expect(r.responseEnvelope).toBe(true);expect(r.tools![0]!.input_schema).toEqual({type:'object',properties:{result:schema},required:['result'],additionalProperties:false});
 const c=toChatCompletion(result('tool_use',undefined,[{type:'tool_use',name:'respond',input:{result:{action:'inspect'}},id:'t'}]),1,'respond',true);expect(c.choices[0].message.content).toBe('{"action":"inspect"}');
});
it('rejects envelope-invalid tool input without fallback',()=>{const c=toChatCompletion(result('tool_use',undefined,[{type:'tool_use',name:'respond',input:{action:'inspect'},id:'t'}]),1,'respond',true);expect(c.choices[0].finish_reason).toBe('unsupported_content');expect(c.choices[0].message.content).toBe('');});
it('rejects root union references rather than changing their resolution',()=>{expect(()=>req({response_format:{type:'json_schema',json_schema:{name:'out',schema:{type:'object',oneOf:[{$ref:'#/$defs/branch'}],$defs:{branch:{type:'object'}}}}}})).toThrow();});
it('instructs envelope placement explicitly',()=>{const r=req({response_format:{type:'json_schema',json_schema:{name:'out',schema:{oneOf:[{type:'object'}]}}}});expect(r.system).toContain('required result property');expect(r.system).not.toContain('Its input is the result.');});
it.each([{value:null},{value:[]},{value:3},{value:'text'}])('rejects nonobject envelope result %j',({value})=>{const c=toChatCompletion(result('tool_use',undefined,[{type:'tool_use',name:'respond',input:{result:value},id:'t'}]),1,'respond',true);expect(c.choices[0].finish_reason).toBe('unsupported_content');expect(c.choices[0].message.content).toBe('');});
