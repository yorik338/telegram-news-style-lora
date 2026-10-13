"""Advisory calibrated NeMo checks; does not replace source editorial review."""
import asyncio,json,sys,time,requests
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'evaluation/judge_runtime'))
from check_grounding import GroundingChecker,read_rows,digest
from check_strong_nemo import ApiJudge

async def main():
    out=ROOT/'evaluation/calibrated_checks.jsonl';assert not out.exists()
    deadline=time.monotonic()+60
    while True:
        try:requests.get('http://127.0.0.1:8091/health',timeout=2).raise_for_status();break
        except requests.RequestException:
            if time.monotonic()>deadline:raise
            await asyncio.sleep(1)
    checker=GroundingChecker(ApiJudge(temperature=.6,max_tokens=8,grammar='root ::= "yes" | "no"'),config_path=ROOT/'evaluation/judge_config')
    inputs=[]
    for name in ('generation_first_attempts','final_outputs','final_holdout'):
        for r in read_rows(ROOT/f'evaluation/{name}.jsonl'):inputs.append(dict(dataset=name,**r))
    for r in read_rows(ROOT/'evaluation/naive_before_after.jsonl'):
        inputs.append(dict(dataset='naive',id=r['id'],**r['before']))
    for i,r in enumerate(inputs,1):
        record=dict(dataset=r['dataset'],id=r['id'],style=r['style'],variant=r['variant'],
            **await checker.check([r['source']],r['response']))
        with out.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps(dict(i=i,total=len(inputs),dataset=r['dataset'],supported=record['supported'])),flush=True)
    result={}
    for dataset,variant in sorted({(r['dataset'],r['variant']) for r in inputs}):
        rs=[r for r in read_rows(out) if r['dataset']==dataset and r['variant']==variant]
        result[dataset+'/'+variant]=dict(cases=len(rs),approved=sum(r['supported'] is True for r in rs),
            rejected=sum(r['supported'] is False for r in rs),unknown=sum(r['supported'] is None for r in rs))
    (ROOT/'evaluation/calibrated_check_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('REVIEW_COMPLETE',result,flush=True)
if __name__=='__main__':asyncio.run(main())
