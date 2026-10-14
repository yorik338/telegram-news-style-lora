"""First-token diagnostics on already-viewed cases, before any fresh regression generation."""
import argparse,torch
from common import ROOT,PROJECT,read,rows,dump,messages,load_base

def main():
    from peft import PeftModel,prepare_model_for_kbit_training
    ap=argparse.ArgumentParser();ap.add_argument('--base-path',required=True);a=ap.parse_args()
    model,tok=load_base(a.base_path);model=PeftModel.from_pretrained(model,ROOT/'adapter').eval()
    cases=[dict(id=r['id'],source=r['text']) for r in rows(PROJECT/'data/test_inputs.jsonl')[:2]]
    cases += [dict(id=r['id'],source=r['source']) for r in rows(ROOT/'data/validation.jsonl') if r['id']=='type1_22_provided']
    results=[]
    for prepared in (False,True):
        if prepared:
            # Match frozen non-quantized parameters' training precision; do not change learned LoRA values.
            for name,p in model.named_parameters():
                if 'lora_' not in name and p.__class__.__name__!='Params4bit' and p.dtype in (torch.float16,torch.bfloat16):p.data=p.data.float()
        for autocast in (False,True):
            for row in cases:
                inputs=tok.apply_chat_template(messages(row['source'],1),add_generation_prompt=True,tokenize=True,
                    return_dict=True,return_tensors='pt').to('cuda')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=autocast):
                    logits=model(**inputs,logits_to_keep=1,use_cache=False).logits[0,-1].float()
                values,indices=torch.topk(logits,5)
                eos=model.generation_config.eos_token_id
                eos=[eos] if isinstance(eos,int) else eos
                record=dict(id=row['id'],prepared_fp32_frozen=prepared,autocast_fp16=autocast,
                    top=[dict(id=int(i),text=tok.decode([int(i)]),logit=float(v)) for v,i in zip(values,indices)],
                    eos_logits={str(i):float(logits[i]) for i in eos})
                results.append(record);print(record,flush=True)
    dump(ROOT/'evaluation/precision_diagnostic.json',results)
if __name__=='__main__':main()
