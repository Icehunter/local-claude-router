import {HttpError} from './http-support.js';
const object=(value:unknown):value is Record<string,unknown>=>typeof value==='object'&&value!==null&&!Array.isArray(value);
/** Conservative proof: each alternative must itself require an object. */
function provesObject(schema:unknown):boolean {
 if(!object(schema))return false;
 if(schema.type==='object')return true;
 if(schema.type!==undefined)return false;
 const unions=['oneOf','anyOf'].filter(key=>schema[key]!==undefined);
 return unions.length>0&&unions.every(key=>Array.isArray(schema[key])&&schema[key].length>0&&schema[key].every(provesObject));
}
/** Tool input remains unchanged except a redundant root object type proven by its unions. */
export function responseTool(value:unknown):{schema:Record<string,unknown>;envelope:boolean}|undefined {
 if(value===undefined)return undefined;
 const fail=():never=>{throw new HttpError(400,'unsupported response_format; nonstrict json_schema required');};
 if(!object(value)||value.type!=='json_schema'||Object.keys(value).some(k=>!['type','json_schema'].includes(k)))return fail();
 const d=value.json_schema;
 if(!object(d)||typeof d.name!=='string'||!d.name.length||d.name.length>64||d.strict!==undefined&&d.strict!==false||!object(d.schema)||Object.keys(d).some(k=>!['name','description','strict','schema'].includes(k))||d.description!==undefined&&typeof d.description!=='string')return fail();
 let hasReference=false;
 let count=0;const walk=(v:unknown,depth:number):void=>{if(++count>10000||depth>64)fail();if(v&&typeof v==='object'){if(Object.hasOwn(v,'$ref')||Object.hasOwn(v,'$dynamicRef')||Object.hasOwn(v,'$recursiveRef'))hasReference=true;for(const child of Object.values(v))walk(child,depth+1);}};
 walk(d.schema,0);if(JSON.stringify(d.schema).length>256000)return fail();
 if(d.schema.type!=="object"&&!provesObject(d.schema))return fail();
 const schema=structuredClone(d.schema);
 const envelope=['oneOf','anyOf','allOf'].some(key=>Object.hasOwn(schema,key));
 if(envelope&&hasReference)return fail();
 return {schema:envelope?{type:'object',properties:{result:schema},required:['result'],additionalProperties:false}:schema,envelope};
}
