"""Raw first responses, with immutable manifests and safe exact-case resume."""
import argparse,contextlib,time
from datetime import datetime,timezone
from common import ROOT,PROJECT,read,rows,sha,dump,append,messages,load_base,generate

def main():
 import torch
 from peft import PeftModel
 ap=argparse.ArgumentParser();ap.add_argument('--base-path',required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 dest=ROOT/'evaluation';out=dest/'raw_comparison.jsonl';manifest_path=dest/'generation_manifest.json'
 artifact=read(ROOT/'training/artifact.json');plan=read(ROOT/'evaluation_plan.json')
 assert sha(ROOT/'adapter/adapter_model.safetensors')==artifact['sha256']
 assert sha(ROOT/'data/fresh_test.jsonl')==plan['fresh_sha256']
 identity=dict(adapter_sha256=artifact['sha256'],common_sha256=sha(ROOT/'common.py'),
   chat_format_sha256=sha(ROOT/'chat_format.py'),demo_sha256=sha(ROOT/'data/demos.json'),
   script_sha256=sha(__file__),fresh_sha256=plan['fresh_sha256'],provided_sha256=sha(PROJECT/'data/test_inputs.jsonl'))
 if a.resume:
  prior=read(manifest_path)
  assert all(prior[k]==v for k,v in identity.items()),'Resume requires exactly the same frozen code, weights and inputs'
 else:
  assert not manifest_path.exists() and not out.exists(),'Never overwrite a generation attempt'
  dump(manifest_path,dict(**identity,started_utc=datetime.now(timezone.utc).isoformat(),
   first_attempts=True,development_sources_previously_seen=True,fresh_sources_first_attempts=True,
   output_editorial_edits=False,feedback_or_retuning_from_evaluation=False,
   max_new_tokens=384,decoding='greedy',inference_batch=1,export_loaded_from_disk=True,
   inference_adapter_dtype='FP16 (autocast_adapter_dtype=False)',
   timing_scope='single generation; shared user GPU, not a controlled speed benchmark'))
 cases=[dict(id=r['id'],source=r['text'],split='development') for r in rows(PROJECT/'data/test_inputs.jsonl')]
 cases += [dict(id=r['id'],source=r['source'],split='fresh_test') for r in rows(ROOT/'data/fresh_test.jsonl')]
 existing=rows(out) if out.exists() else []
 done={(r['split'],r['id'],r['style'],r['variant']) for r in existing}
 assert len(done)==len(existing),'Duplicate first responses'
 expected={(r['split'],r['id'],s,v) for r in cases for s in (1,2) for v in ('base','lora_v3')}
 assert done<=expected
 model,tok=load_base(a.base_path)
 model=PeftModel.from_pretrained(model,ROOT/'adapter',autocast_adapter_dtype=False).eval()
 torch.cuda.empty_cache()
 start=time.perf_counter()
 for i,r in enumerate(cases):
  for style in (1,2):
   for variant in ('base','lora_v3'):
    k=(r['split'],r['id'],style,variant)
    if k in done:continue
    torch.cuda.empty_cache()
    ctx=model.disable_adapter() if variant=='base' else contextlib.nullcontext()
    with ctx:rec=generate(model,tok,messages(r['source'],style))
    append(out,dict(**r,style=style,variant=variant,**rec));done.add(k)
    print('GENERATED',i+1,len(cases),r['id'],style,variant,rec['output_tokens'],round(rec['seconds'],2),flush=True)
 assert done==expected
 records=rows(out)
 for variant in ('base','lora_v3'):
  folder=dest/variant;folder.mkdir(exist_ok=True)
  for split in ('development','fresh_test'):
   for style in (1,2):
    rs=[r for r in records if r['variant']==variant and r['style']==style and r['split']==split]
    (folder/f'{split}_type{style}.txt').write_text('\n\n---\n\n'.join(f"Источник {r['id']}\n\n"+r['response'] for r in rs)+'\n',encoding='utf-8')
 dump(dest/'duration.json',dict(this_session_seconds=time.perf_counter()-start,records=len(records),
   sum_generation_seconds=sum(r['seconds'] for r in records),resumed=a.resume))
 print('EVALUATION_COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
