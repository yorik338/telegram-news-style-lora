"""Re-run the existing NeMo evidence rail with a larger server model."""
import argparse, asyncio, importlib.metadata, inspect, json, shutil, time
from pathlib import Path
import requests
from nemoguardrails.types import LLMResponse, UsageInfo
from nemoguardrails.library.self_check.facts.actions import self_check_facts
from check_grounding import GroundingChecker, ROOT, digest, read_rows

WORK=ROOT.parent.parent/'work'
class ApiJudge:
    provider_name='local-llama.cpp'
    provider_url='http://127.0.0.1:8091'
    model_name='Qwen3-14B-Q4_K_M'
    def __init__(self, temperature=0.6, max_tokens=512, grammar=None):
        self.last_call=None
        self.temperature=temperature
        self.max_tokens=max_tokens
        self.grammar=grammar
    async def generate_async(self,prompt,*,stop=None,**kwargs):
        messages=[{'role':'user','content':prompt}] if isinstance(prompt,str) else [m.to_dict() for m in prompt]
        # The upstream rail's 32-token answer allowance excludes the reasoning
        # budget. Do not apply its newline stop to the internal reasoning field.
        payload={'model':'strong-fact-judge','messages':messages,'temperature':self.temperature,
                 'top_p':0.95,'top_k':20,'min_p':0,'seed':42,'max_tokens':self.max_tokens,
                 'reasoning_effort':'none','chat_template_kwargs':{'enable_thinking':False}}
        if self.grammar is not None: payload['grammar']=self.grammar
        started=time.perf_counter()
        reply=await asyncio.to_thread(requests.post,self.provider_url+'/v1/chat/completions',json=payload,timeout=600)
        reply.raise_for_status();data=reply.json();choice=data['choices'][0]
        raw=choice['message']['content'].strip(); usage=data.get('usage',{})
        self.last_call={'raw_verdict':raw,'finish_reason':choice['finish_reason'],
                        'messages':messages,'request':payload,'api_response':data,
                        'seconds':time.perf_counter()-started,
                        'prompt_tokens':usage.get('prompt_tokens'), 'output_tokens':usage.get('completion_tokens')}
        if choice['message'].get('reasoning_content'):
            raise ValueError('Unexpected reasoning: this calibrated configuration requires non-thinking judgement')
        return LLMResponse(content=raw,model=self.model_name,finish_reason=choice['finish_reason'],
             usage=UsageInfo(input_tokens=usage.get('prompt_tokens',0),output_tokens=usage.get('completion_tokens',0),total_tokens=usage.get('total_tokens',0)))
    async def stream_async(self,*args,**kwargs):
        raise NotImplementedError
        yield

async def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['controls','benchmark'])
    p.add_argument('--name',required=True);p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true')
    args=p.parse_args(); inputs=[];paths=[]
    if args.stage=='controls':
        paths=[ROOT/'checker_checks.jsonl',ROOT/'checker_challenge.jsonl']
        for path in paths: inputs.extend({**r,'source_set':path.stem,'case_id':r['id']} for r in read_rows(path))
    else:
        paths=sorted((ROOT/'generations').glob('qwen*_grounded_v1/posts.jsonl'))
        for path in paths:
            inputs.extend({**r,'case_id':r['generator']+'_'+r['id']+'_'+str(r['style'])} for r in read_rows(path))
    if args.limit: inputs=inputs[:args.limit]
    out=ROOT/'runs'/args.name
    if out.exists() and not args.resume: raise RuntimeError('Refuse overwriting existing run')
    manifest={'tool':'NeMo Guardrails self_check_facts','version':2,'stage':args.stage,'cases':len(inputs),
        'judge':json.loads((WORK/'strong-judge/lock.json').read_text()),
        'server':json.loads((WORK/'strong-judge/server.json').read_text(encoding='utf-8-sig')),
        'script_sha256':digest(__file__),'upstream_action_sha256':digest(inspect.getfile(self_check_facts)),
        'config_hashes':{p.name:digest(p) for p in (ROOT/'config-v3').glob('*.yml')},
        'dataset_hashes':{str(p.relative_to(ROOT)):digest(p) for p in paths},
        'decoding':{'temperature':0.6,'top_p':0.95,'top_k':20,'min_p':0,'seed':42,'max_tokens':512,'reasoning_effort':'none','enable_thinking':False},
        'nemoguardrails':importlib.metadata.version('nemoguardrails'),
        'same_generator_family':True,'judge_in_generator_pool':False,'human_validation':False,
        'score_semantics':'Binary support from attached sources, not real-world truth probability',
        'diagnostic_limit':args.limit}
    out.mkdir(parents=True,exist_ok=True)
    if not args.resume:
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        shutil.copyfile(__file__,out/Path(__file__).name)
        shutil.copytree(ROOT/'config-v3',out/'config-snapshot')
    else:
        prior=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if prior!=manifest: raise RuntimeError('Resume configuration mismatch')
    done={r['case_id'] for r in read_rows(out/'results.jsonl')} if (out/'results.jsonl').exists() else set()
    checker=GroundingChecker(ApiJudge())
    for row in inputs:
        if row['case_id'] in done: continue
        verdict=await checker.check(row['sources'],row['response'])
        record={**row,**verdict}
        with (out/'results.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps({k:record.get(k) for k in ('case_id','supported','expected_supported','status','seconds','error')},ensure_ascii=False),flush=True)

if __name__=='__main__': asyncio.run(main())
