"""Freeze the checkpoint, generate ten posts and three naive comparisons, then holdout."""
import argparse,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
from pipeline import ROOT,Rewriter
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-path');ap.add_argument('--holdout',action='store_true');a=ap.parse_args()
    dest=ROOT/'evaluation';audit=dest/'generation_first_attempts.jsonl'
    assert not audit.exists(),'Never replace first responses'
    started=time.perf_counter()
    config=json.loads((ROOT/'training/config.json').read_text(encoding='utf-8'))
    artifact=json.loads((ROOT/'training/artifact.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'adapter/adapter_model.safetensors')==artifact['sha256']
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),base=config['base'],revision=config['revision'],
        adapter_sha256=artifact['sha256'],prompt_sha256=sha(ROOT/'prompt_template.json'),
        script_sha256=sha(Path(__file__)),pipeline_sha256=sha(ROOT/'pipeline.py'),test_inputs_sha256=sha(ROOT/'data/test_inputs.jsonl'),
        decoding=dict(temperature=.6,top_p=.95,top_k=20,repetition_penalty=1,seed=42,max_new_tokens=512),
        attempts_per_primary_case=1,naive_comparison_cases=[[1,1],[3,2],[4,1]],
        holdout_checkpoint_and_prompt_frozen_before_read=True,holdout_used_for_tuning=False)
    (dest/'generation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    model=Rewriter(a.base_path)
    model.generate('Библиотека открыла читальный зал. Вход свободный.',2)
    records=[]
    for r in rows(ROOT/'data/test_inputs.jsonl'):
        for style in (1,2):
            rec=dict(id=r['id'],split='provided_test',**model.generate(r['text'],style))
            records.append(rec)
            with audit.open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
            print(json.dumps({k:rec.get(k) for k in ('id','style','seconds','truncated')}),flush=True)
    for style in (1,2):
        rs=[r for r in records if r['style']==style]
        (ROOT/f'outputs_type{style}.txt').write_text('\n\n---\n\n'.join(f"Новость {r['id']}\n\n"+r['response'] for r in rs)+'\n',encoding='utf-8')
    for identity,style in manifest['naive_comparison_cases']:
        r=next(r for r in rows(ROOT/'data/test_inputs.jsonl') if r['id']==identity)
        before=model.generate(r['text'],style,variant='naive')
        after=next(r for r in records if r['id']==identity and r['style']==style)
        with (dest/'naive_before_after.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(id=identity,style=style,source=r['text'],before=before,after=after),ensure_ascii=False)+'\n')
        print('NAIVE',identity,style,flush=True)
    if a.holdout:
        path=ROOT/'data/final_holdout.jsonl'
        assert sha(path)=='2cebbb9aafe37818578c58ff6817e59405c105aed5c2064754f1d2e8c1e4f9e0'
        for r in rows(path):
            for style in (1,2):
                for variant in ('base','solution'):
                    rec=dict(id=r['id'],split='final_holdout',**model.generate(r['source'],style,variant=variant))
                    with (dest/'final_holdout.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                    print('HOLDOUT',r['id'],style,variant,flush=True)
    (dest/'generation_duration.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started),indent=2),encoding='utf-8')
    print('GENERATION_COMPLETE',flush=True)
if __name__=='__main__':main()
