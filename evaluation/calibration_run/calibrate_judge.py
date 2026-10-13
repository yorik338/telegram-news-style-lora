"""One bounded calibration pass over existing controls; no benchmark repairs."""
import asyncio,json,time,shutil,argparse
from pathlib import Path
from check_grounding import ROOT,GroundingChecker,read_rows,digest
from check_strong_nemo import ApiJudge

async def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='nemo-qwen14b-calibrated-v2');a=ap.parse_args()
    out=ROOT/'runs'/a.name
    out.mkdir(exist_ok=False)
    config=ROOT/'config-calibrated'
    shutil.copytree(config,out/'config-snapshot')
    shutil.copyfile(__file__,out/'calibrate_judge.py')
    inputs=read_rows(ROOT/'checker_checks.jsonl')+read_rows(ROOT/'checker_challenge.jsonl')
    manifest=dict(cases=len(inputs),temperature=0.6,max_tokens=8,thinking=False,grammar='root ::= "yes" | "no"',
        tool='NeMo self_check_facts',model='Qwen3-14B-Q4_K_M',script_sha256=digest(__file__),
        config_hashes={p.name:digest(p) for p in config.glob('*.yml')},
        controls_previously_inspected=True,independent_human_labels=False,
        note='Prompt and decoding calibration on reused controls; not a fresh generalization test')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    checker=GroundingChecker(ApiJudge(temperature=0.6,max_tokens=8,grammar='root ::= "yes" | "no"'),config_path=config)
    results=[]
    for r in inputs:
        result={**r,**await checker.check(r['sources'],r['response'])}
        results.append(result)
        with (out/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(json.dumps({k:result.get(k) for k in ('id','supported','expected_supported','status','seconds')},ensure_ascii=False),flush=True)
    summary={}
    for group,rs in [('all',results)]+[(s,[r for r in results if r.get('split')==s]) for s in sorted({r.get('split','challenge') for r in results})]:
        if not rs:continue
        summary[group]=dict(cases=len(rs),matches=sum(r['supported']==r['expected_supported'] for r in rs),
            false_accepts=sum(r['supported'] is True and r['expected_supported'] is False for r in rs),
            false_rejects=sum(r['supported'] is False and r['expected_supported'] is True for r in rs),
            unknown=sum(r['supported'] is None for r in rs))
    summary['automatic_acceptance_enabled']=False
    summary['policy']='Advisory verdicts only; supported=True is not proof. Source audit remains authoritative.'
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CALIBRATION',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':asyncio.run(main())
