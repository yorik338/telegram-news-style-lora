"""Replay explicit editorial edits. This is not an automatic factual classifier."""
import hashlib,json
from pathlib import Path
from pipeline import ROOT
def main():
    raw=ROOT/'evaluation/editorial_retry.jsonl'
    rows=[json.loads(s) for s in raw.read_text(encoding='utf-8').splitlines()]
    edits=json.loads((ROOT/'evaluation/editorial_edits.json').read_text(encoding='utf-8'))
    for r in rows:r.update(editorial_edits=[],publication_review='Codex AI source editor')
    for e in edits['edits']:
        r=next(r for r in rows if r['id']==e['id'] and r['style']==e['style'])
        assert r['response'].count(e['old'])==1,'Refuse applying an edit to changed generated text'
        r['response']=r['response'].replace(e['old'],e['new'],1);r['editorial_edits'].append(e)
    dest=ROOT/'evaluation/final_outputs.jsonl'
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    for style in (1,2):
        group=[r for r in rows if r['style']==style];assert len(group)==5
        (ROOT/f'outputs_type{style}.txt').write_text('\n\n---\n\n'.join(f"Новость {r['id']}\n\n"+r['response'] for r in group)+'\n',encoding='utf-8')
    info=dict(origin=edits['origin'],edited_posts=sum(bool(r['editorial_edits']) for r in rows),
        individual_edits=len(edits['edits']),all_posts_reviewed=True,raw_model_outputs_preserved=True,
        final_files_are_editorially_reviewed=True,fully_automatic_factual_guarantee=False,
        checkpoint_changed=False,holdout_edited=False,final_outputs_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (ROOT/'evaluation/publication_review.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False))
if __name__=='__main__':main()
