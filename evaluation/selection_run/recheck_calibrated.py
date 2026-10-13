import asyncio,json,shutil
from collections import Counter
from check_grounding import ROOT,GroundingChecker,read_rows,digest
from check_strong_nemo import ApiJudge

async def main():
    out=ROOT/'runs/nemo-qwen14b-calibrated-benchmark-v1';out.mkdir(exist_ok=False)
    paths=sorted((ROOT/'generations').glob('*_grounded_v4/posts.jsonl'))
    inputs=[r for p in paths for r in read_rows(p)]
    assert len(inputs)==106
    config=ROOT/'config-calibrated'
    manifest=dict(cases=106,calibration='nemo-qwen14b-calibrated-v2',temperature=.6,max_tokens=8,
        grammar='root ::= "yes" | "no"',source_files={str(p.relative_to(ROOT)):digest(p) for p in paths},
        config_hashes={p.name:digest(p) for p in config.glob('*.yml')},script_sha256=digest(__file__),
        semantic_accuracy_claimed=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    shutil.copytree(config,out/'config-snapshot');shutil.copyfile(__file__,out/'recheck_calibrated.py')
    checker=GroundingChecker(ApiJudge(temperature=.6,max_tokens=8,grammar=manifest['grammar']),config_path=config)
    results=[]
    for i,row in enumerate(inputs,1):
        result={**row,**await checker.check(row['sources'],row['response'])};results.append(result)
        with (out/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(json.dumps(dict(i=i,generator=row['generator'],id=row['id'],style=row['style'],supported=result['supported'])),flush=True)
    summary={}
    for name in sorted({r['generator'] for r in results}):
        rs=[r for r in results if r['generator']==name]
        summary[name]=dict(cases=len(rs),approved=sum(r['supported'] is True for r in rs),
            rejected=sum(r['supported'] is False for r in rs),unknown=sum(r['supported'] is None for r in rs))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('SUMMARY',summary,flush=True)
if __name__=='__main__':asyncio.run(main())
