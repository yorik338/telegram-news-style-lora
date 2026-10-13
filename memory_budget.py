"""Analytic KV budget; actual vLLM activation/workspace must be profiled."""
import argparse,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def budget(total_gib=24,other_gib=0,reserve_gib=2,model_runtime_gib=5.5,util=.85,seqs=15,length=4096):
    c=json.loads((ROOT/'model_config.json').read_text(encoding='utf-8'))
    head=c.get('head_dim',c['hidden_size']//c['num_attention_heads'])
    token=2*c['num_hidden_layers']*c['num_key_value_heads']*head*2
    safe_util=min(util,(total_gib-other_gib-reserve_gib)/total_gib)
    if safe_util<=0:raise ValueError('No memory budget')
    one=token*length/2**30;kv=one*seqs
    available=safe_util*total_gib-model_runtime_gib
    return dict(status='analytic estimate, not measured 24GiB deployment',layers=c['num_hidden_layers'],kv_heads=c['num_key_value_heads'],
        head_dim=head,kv_dtype='FP16',kv_bytes_per_token=token,kv_gib_per_request=one,kv_gib_total=kv,
        physical_gib=total_gib,other_process_gib=other_gib,external_reserve_gib=reserve_gib,
        gpu_memory_utilization=safe_util,model_executor_budget_gib=safe_util*total_gib,
        assumed_model_runtime_gib=model_runtime_gib,required_runtime_profiling=True,
        available_kv_gib=available,estimated_kv_capacity_tokens=max(0,int(available*2**30//token)),
        required_kv_capacity_tokens=seqs*length,estimated_fits=kv<=available,
        note='model_runtime_gib is a placeholder for measured weights+LoRA+activation/workspace; replace it with actual profiling')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--total-gib',type=float,default=24);p.add_argument('--other-gib',type=float,default=0)
    p.add_argument('--reserve-gib',type=float,default=2);p.add_argument('--model-runtime-gib',type=float,default=5.5)
    p.add_argument('--util',type=float,default=.85);p.add_argument('--seqs',type=int,default=15);p.add_argument('--length',type=int,default=4096)
    a=p.parse_args();print(json.dumps(budget(**vars(a)),ensure_ascii=False,indent=2))
