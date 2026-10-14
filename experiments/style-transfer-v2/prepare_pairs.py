"""Style target -> neutral input, then learn the reverse direction; resume raw calls."""
import argparse,re,torch
from common import ROOT,PROJECT,read,rows,dump,append,sha,load_base,generate,NEUTRAL_SYSTEM

def target(row):
    obj=row['target']
    if row['style']==1:return ' '.join(obj['paragraphs'])+' @banki_oil'
    emoji=row['original_target'].split()[0]
    return emoji+' **'+obj['headline']+'**\n\n'+'\n\n'.join(obj['paragraphs'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-path',required=True);a=ap.parse_args()
    data=ROOT/'data';data.mkdir(exist_ok=True)
    corpus=[]
    for split,file in [('train','train.jsonl'),('validation','validation.jsonl')]:
        for row in rows(PROJECT/'data'/file):
            corpus.append(dict(id=row['id'],group_id=row['group_id'],style=row['style'],split=split,
                original_source=row['source'],target=target(row),original_target=row['original_target'],
                target_origin='Previously AI-curated factual references; financial separate headline omitted, city original emoji restored'))
    # These two style demos remain excluded from train/validation.
    demos={}
    for s,d in read(PROJECT/'data/prompt_examples.json').items():
        demos[s]=dict(source=d['source'],target=d['response'],id=d['original_id'])
    dump(data/'demos.json',demos)
    dump(data/'corpus.json',corpus)
    raw=data/'neutralization_raw.jsonl';done={r['id'] for r in rows(raw)} if raw.exists() else set()
    dump(ROOT/'preparation_manifest.json',dict(upstream=read(ROOT/'upstream_manifest.json'),
        base=read(PROJECT/'model_lock.json'),original_train_sha256=sha(PROJECT/'data/train.jsonl'),
        original_validation_sha256=sha(PROJECT/'data/validation.jsonl'),corpus_sha256=sha(data/'corpus.json'),
        neutralization_system=NEUTRAL_SYSTEM,decoding='greedy',semantic_review_required=True,
        source_of_corpus='38 training and 10 validation rows; original 2 prompt demos excluded',
        original_test_and_previous_holdout_used_for_training=False))
    model,tok=load_base(a.base_path)
    for i,row in enumerate(corpus,1):
        if row['id'] in done:continue
        rec=generate(model,tok,[dict(role='system',content=NEUTRAL_SYSTEM),dict(role='user',content=row['target'])],512)
        append(raw,dict(id=row['id'],style=row['style'],split=row['split'],target=row['target'],**rec))
        print('NEUTRAL',i,len(corpus),row['id'],rec['output_tokens'],round(rec['seconds'],2),flush=True)
    print('PREPARATION_RAW_COMPLETE; review semantic coverage before building pairs',flush=True)
if __name__=='__main__':main()
