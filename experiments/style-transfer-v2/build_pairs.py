"""Build reverse pairs only after explicit source/neutral semantic review."""
from common import ROOT,read,rows,dump,sha
import json,re

def clean_neutral(text):
    # Channel signatures are known corpus metadata, not news evidence. This is
    # preparation-time cleanup; raw model outputs are preserved verbatim.
    return re.sub(r'\s*(?:(?:Авторы? сообщения\s*(?:—|:)|Информация предоставлена)\s*)?@banki_oil\.?\s*$','',text).strip()

def main():
    corpus=read(ROOT/'data/corpus.json');raw={r['id']:r for r in rows(ROOT/'data/neutralization_raw.jsonl')}
    audit=read(ROOT/'data/neutralization_review.json');decisions={r['id']:r for r in audit['decisions']}
    assert len(raw)==len(corpus)==len(decisions)==48
    output={'train':[],'validation':[]};accepted=0
    for r in corpus:
        # Anchor on the supplied neutral source to avoid training only on short synthetic inputs.
        output[r['split']].append(dict(id=r['id']+'_provided',group_id=r['group_id'],style=r['style'],
            source=r['original_source'],target=r['target'],origin='provided_neutral_source'))
        decision=decisions[r['id']]
        assert decision['raw_sha256']==__import__('hashlib').sha256(raw[r['id']]['response'].encode()).hexdigest()
        if decision['accepted']:
            assert not raw[r['id']]['truncated']
            output[r['split']].append(dict(id=r['id']+'_reverse',group_id=r['group_id'],style=r['style'],
                source=clean_neutral(raw[r['id']]['response']),target=r['target'],origin='style_lora_backtranslation',
                raw_neutral=raw[r['id']]['response'],signature_metadata_removed=clean_neutral(raw[r['id']]['response'])!=raw[r['id']]['response']))
            accepted+=1
    # The 10 original validation inputs are the criterion; synthetic validation variants are diagnostic only.
    val_original=[r for r in output['validation'] if r['origin']=='provided_neutral_source']
    assert not ({r['group_id'] for r in output['train']}&{r['group_id'] for r in val_original})
    for name,rs in [('train',output['train']),('validation',val_original),('validation_reverse',output['validation'])]:
        path=ROOT/f'data/{name}.jsonl'
        assert not path.exists(),'Never overwrite frozen pairs'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8')
    dump(ROOT/'data/pairs_manifest.json',dict(reverse_accepted=accepted,reverse_total=48,
        review_origin=audit['origin'],prompt_demos=2,unique_train_groups=len({r['group_id'] for r in output['train']}),
        validation_used_for_checkpoint_selection='10 original held-out source inputs, not synthetic validation variants',
        files={n:dict(rows=len(rows(ROOT/'data'/n)),sha256=sha(ROOT/'data'/n)) for n in ['train.jsonl','validation.jsonl','validation_reverse.jsonl']},
        synthetic_rows_are_not_independent_examples=True))
    print('PAIRS_COMPLETE',len(output['train']),len(val_original),accepted,flush=True)
if __name__=='__main__':main()
