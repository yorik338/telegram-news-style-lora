"""Adapt style-lora's 8-gram overlap and blind A/B export; no invented human votes."""
import html,json,random,re
from common import ROOT,rows,dump,sha

def grams(text):
    text=text.replace('@banki_oil','').replace('**','')
    words=re.findall(r'\w+',text.lower())
    return {tuple(words[i:i+8]) for i in range(max(0,len(words)-7))}

def format_ok(r):
    text=r['response'].strip();parts=[p for p in text.split('\n\n') if p.strip()]
    emoji=lambda s:any(0x1f000<=ord(c)<=0x1faff or 0x2600<=ord(c)<=0x27bf for c in s)
    if r['style']==1:return len(parts)==1 and text.endswith('@banki_oil') and '**' not in text and not emoji(text)
    lead=parts[0].split('**',1)[0] if parts else ''
    return len(parts)==3 and emoji(lead) and bool(re.match(r'^.*\*\*.+\*\*$',parts[0])) and '@banki_oil' not in text

def main():
    rs=rows(ROOT/'evaluation/raw_comparison.jsonl');train=rows(ROOT/'data/train.jsonl')
    known={s:set().union(*(grams(r['target']) for r in train if r['style']==s)) for s in (1,2)}
    summaries={};rng=random.Random(107);key=[];items=[]
    by={(r['split'],r['id'],r['style'],r['variant']):r for r in rs}
    for variant in ('base','lora_v3'):
        selected=[r for r in rs if r['variant']==variant]
        fractions=[]
        for r in selected:
            g=grams(r['response']);fractions.append(len(g&known[r['style']])/max(1,len(g)))
        summaries[variant]=dict(posts=len(selected),truncated=sum(r['truncated'] for r in selected),
            mean_source_8gram_overlap=sum(len(grams(r['response'])&grams(r['source']))/max(1,len(grams(r['response']))) for r in selected)/len(selected),
            format_passes=sum(format_ok(r) for r in selected),
            mean_words=sum(len(r['response'].split()) for r in selected)/len(selected),
            mean_training_8gram_overlap=sum(fractions)/len(fractions),
            fact_or_style_quality_inferred_from_these_metrics=False)
    for identity in sorted({(r['split'],str(r['id']),r['style']) for r in rs}):
        split,k,style=identity
        base=next(r for r in rs if r['split']==split and str(r['id'])==k and r['style']==style and r['variant']=='base')
        new=next(r for r in rs if r['split']==split and str(r['id'])==k and r['style']==style and r['variant']=='lora_v3')
        left_new=rng.random()<.5;left,right=(new,base) if left_new else (base,new)
        item_id=f'{split}:{k}:{style}';key.append(dict(item=item_id,left='lora_v3' if left_new else 'base'))
        items.append(f'''<section><h2>{html.escape(item_id)} — стиль {style}</h2>
<details><summary>Источник</summary><pre>{html.escape(base['source'])}</pre></details>
<div class="pair"><article><h3>A</h3><pre>{html.escape(left['response'])}</pre></article><article><h3>B</h3><pre>{html.escape(right['response'])}</pre></article></div>
<label>Предпочтение по стилю: <select data-item="{item_id}"><option value="">Не оценено</option><option>A</option><option>B</option><option>Одинаково</option><option>Оба плохие</option></select></label>
</section>''')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Слепое сравнение стиля</title>
<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#fafafa;color:#222}pre{white-space:pre-wrap;font:inherit;line-height:1.5}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{padding:18px;background:white;border:1px solid #ddd}section{margin:32px 0;border-top:1px solid #ccc}button,select{font:inherit;padding:8px}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style>
<h1>Какой вариант лучше передаёт стиль?</h1><p>Оцените все пары до просмотра ключа. Для стиля 1 оценивайте короткий живой аналитический абзац; для стиля 2 — естественную городскую новость с коротким заголовком и двумя абзацами. Факты проверяются отдельно по источнику. Краткость сама по себе не означает хороший стиль. Ответы не редактировались. Ваш выбор сохраняется только в этом браузере.</p>
<button id="save">Скачать оценки</button>'''+''.join(items)+'''
<script>const storageKey='style-transfer-v3-votes';const selects=[...document.querySelectorAll('select[data-item]')];
let votes={};try{votes=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}
selects.forEach(s=>{s.value=votes[s.dataset.item]||'';s.onchange=()=>{votes[s.dataset.item]=s.value;localStorage.setItem(storageKey,JSON.stringify(votes))}});
document.getElementById('save').onclick=()=>{const blob=new Blob([JSON.stringify({origin:'human browser selections',votes},null,2)],{type:'application/json'});const u=URL.createObjectURL(blob);const a=document.createElement('a');a.href=u;a.download='style_votes.json';a.click();URL.revokeObjectURL(u)};</script></html>'''
    (ROOT/'evaluation/blind_ab.html').write_text(page,encoding='utf-8')
    dump(ROOT/'evaluation/blind_ab_key.json',key)
    dump(ROOT/'evaluation/style_diagnostics.json',dict(metrics=summaries,human_votes_received=0,
        blind_pairs=len(key),raw_responses_sha256=sha(ROOT/'evaluation/raw_comparison.jsonl'),
        embedding_auc_omitted='Upstream English semantic encoder is unsuitable as an unquestioned Russian style metric; human A/B remains primary'))
    print('BLIND_AB_COMPLETE',len(key),flush=True)
if __name__=='__main__':main()
