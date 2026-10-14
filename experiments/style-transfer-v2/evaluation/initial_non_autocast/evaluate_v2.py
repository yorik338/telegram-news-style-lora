"""First raw responses only. Reload the exported adapter; freeze before new sources."""
import argparse,contextlib,json,time
from datetime import datetime,timezone
from pathlib import Path
from common import ROOT,PROJECT,read,rows,sha,dump,append,messages,load_base,generate

def main():
    from peft import PeftModel
    ap=argparse.ArgumentParser();ap.add_argument('--base-path',required=True);a=ap.parse_args()
    dest=ROOT/'evaluation';dest.mkdir(exist_ok=True);out=dest/'raw_comparison.jsonl'
    assert not out.exists(),'Never replace first attempts'
    artifact=read(ROOT/'training/artifact.json');plan=read(ROOT/'evaluation_plan.json')
    assert sha(ROOT/'adapter/adapter_model.safetensors')==artifact['sha256']
    assert sha(ROOT/'data/fresh_regression.jsonl')==plan['fresh_regression_sha256']
    dump(dest/'generation_manifest.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
        adapter_sha256=artifact['sha256'],common_sha256=sha(ROOT/'common.py'),
        demo_sha256=sha(ROOT/'data/demos.json'),provided_test_sha256=sha(PROJECT/'data/test_inputs.jsonl'),
        fresh_regression_sha256=plan['fresh_regression_sha256'],script_sha256=sha(__file__),
        decoding='greedy, repetition_penalty=1, max_new_tokens=384',
        first_attempts_only=True,output_editorial_edits=False,feedback_or_retuning_from_evaluation=False,
        adapter_loaded_from_saved_fp16_export=True))
    model,tok=load_base(a.base_path);model=PeftModel.from_pretrained(model,ROOT/'adapter').eval()
    generate(model,tok,messages('Городская библиотека открыла читальный зал. Вход свободный.',2),64)
    cases=[dict(id=r['id'],source=r['text'],split='provided_test') for r in rows(PROJECT/'data/test_inputs.jsonl')]
    # The generated synthetic sources are read only after the trained checkpoint is frozen.
    cases += [dict(id=r['id'],source=r['source'],split='fresh_regression') for r in rows(ROOT/'data/fresh_regression.jsonl')]
    start=time.perf_counter()
    for i,r in enumerate(cases):
        for style in (1,2):
            msg=messages(r['source'],style)
            for variant in ('base','lora_v2'):
                ctx=model.disable_adapter() if variant=='base' else contextlib.nullcontext()
                with ctx:rec=generate(model,tok,msg)
                rec=dict(**r,style=style,variant=variant,**rec)
                append(out,rec)
                print('GENERATED',i+1,len(cases),r['id'],style,variant,rec['output_tokens'],round(rec['seconds'],2),flush=True)
    records=rows(out)
    for variant in ('base','lora_v2'):
        folder=dest/variant;folder.mkdir()
        for style in (1,2):
            rs=[r for r in records if r['variant']==variant and r['style']==style and r['split']=='provided_test']
            (folder/f'outputs_type{style}.txt').write_text('\n\n---\n\n'.join(f"Новость {r['id']}\n\n"+r['response'] for r in rs)+'\n',encoding='utf-8')
    dump(dest/'duration.json',dict(generation_seconds=time.perf_counter()-start,records=len(records)))
    print('EVALUATION_GENERATION_COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
