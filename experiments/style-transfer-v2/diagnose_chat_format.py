"""Verify the fixed-format prefix on viewed provided/validation cases; no training updates."""
import argparse,torch
from common import ROOT,PROJECT,read,rows,dump,messages,load_base,generate
from chat_format import prompt_ids

def main():
    from peft import PeftModel
    ap=argparse.ArgumentParser();ap.add_argument('--base-path',required=True);a=ap.parse_args()
    model,tok=load_base(a.base_path);model=PeftModel.from_pretrained(model,ROOT/'adapter').eval()
    cases=[dict(id=r['id'],source=r['text']) for r in rows(PROJECT/'data/test_inputs.jsonl')[:2]]
    cases += [dict(id=r['id'],source=r['source']) for r in rows(ROOT/'data/validation.jsonl') if r['id']=='type1_22_provided']
    results=[]
    for row in cases:
        msg=messages(row['source'],1);ids,fixed=prompt_ids(tok,msg)
        x=torch.tensor([ids],device='cuda')
        with torch.inference_mode():logits=model(input_ids=x,attention_mask=torch.ones_like(x),logits_to_keep=1,use_cache=False).logits[0,-1].float()
        values,indices=torch.topk(logits,5)
        record=dict(id=row['id'],fixed_template_prefix=tok.decode(fixed),fixed_template_token_ids=fixed,
            top=[dict(id=int(i),text=tok.decode([int(i)]),logit=float(v)) for v,i in zip(values,indices)])
        results.append(record);print(record,flush=True)
    dump(ROOT/'evaluation/chat_format_diagnostic.json',results)
if __name__=='__main__':main()
