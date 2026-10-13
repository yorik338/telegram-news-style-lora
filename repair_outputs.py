"""One transparent source-based editorial retry; first attempts remain intact."""
import argparse,hashlib,json
from pipeline import ROOT,Rewriter
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def main():
    p=argparse.ArgumentParser();p.add_argument('--base-path');a=p.parse_args()
    original=ROOT/'evaluation/generation_first_attempts.jsonl'
    feedback=ROOT/'evaluation/editor_feedback.json'
    out=ROOT/'evaluation/editorial_retry.jsonl'
    assert not out.exists(),'Do not hide repeated retries'
    first=rows(original);cases=json.loads(feedback.read_text(encoding='utf-8'))['cases']
    model=Rewriter(a.base_path);final=[]
    for f in cases:
        previous=next(r for r in first if r['id']==f['id'] and r['style']==f['style'])
        r=dict(id=f['id'],split='provided_test',editorial_attempt=2,feedback=f['feedback'],
            **model.generate(previous['source'],f['style'],feedback=f['feedback']))
        final.append(r)
        with out.open('a',encoding='utf-8') as stream:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
        print(json.dumps({k:r.get(k) for k in ('id','style','seconds','truncated')}),flush=True)
    assert len(final)==10
    (ROOT/'evaluation/final_outputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in final),encoding='utf-8')
    for style in (1,2):
        rs=[r for r in final if r['style']==style]
        (ROOT/f'outputs_type{style}.txt').write_text('\n\n---\n\n'.join(f"Новость {r['id']}\n\n"+r['response'] for r in rs)+'\n',encoding='utf-8')
    (ROOT/'evaluation/editorial_manifest.json').write_text(json.dumps(dict(attempts_per_post=2,
        feedback_origin='Codex semantic editorial review',all_posts_regenerated=True,manual_text_edits=False,
        first_attempts_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        feedback_sha256=hashlib.sha256(feedback.read_bytes()).hexdigest(),holdout_untouched=True),indent=2),encoding='utf-8')
if __name__=='__main__':main()
