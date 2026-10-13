"""Deadline-bounded QLoRA; user explicitly permits the relative benchmark winner."""
import argparse,hashlib,json,math,random,sys,time,contextlib,importlib.metadata
from pathlib import Path
import torch
import torch.nn.functional as F
PROJECT=Path(__file__).resolve().parent
ROOT=PROJECT/'training/reproduction'
BENCH=PROJECT
EVIDENCE=PROJECT
sys.path.insert(0,str(PROJECT))
from pipeline import load_base as load_model,render,messages

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]

def main():
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from transformers import get_linear_schedule_with_warmup
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=1);ap.add_argument('--base-path');a=ap.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    selection=json.loads((EVIDENCE/'relative_selection.json').read_text(encoding='utf-8'))
    assert selection['authorization']=='user_explicit_relative_winner_then_finetune'
    alias=selection['winner']
    info=next(r for r in json.loads((BENCH/'candidates.lock.json').read_text(encoding='utf-8'))['models'] if r['alias']==alias)
    if a.base_path:info['local_model_path']=a.base_path
    else:
        from huggingface_hub import snapshot_download
        info['local_model_path']=snapshot_download(info['model'],revision=info['revision'],allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
    for f in info['weight_files']:assert sha(Path(info['local_model_path'])/f['name'])==f['sha256']
    assert not (ROOT/'adapter').exists() and not (ROOT/'epochs.jsonl').exists(),'Never overwrite training'
    train=rows(BENCH/'data/train.jsonl');validation=rows(BENCH/'data/validation.jsonl')
    manifest=json.loads((BENCH/'training_data_manifest.json').read_text(encoding='utf-8'))
    for name,item in manifest['files'].items():assert sha(BENCH/'data'/name)==item['sha256']
    assert len(train)==38 and len(validation)==10
    assert not ({r['group_id'] for r in train}&{r['group_id'] for r in validation})
    torch.manual_seed(42);random.seed(42)
    model,tokenizer=load_model(info['local_model_path'],info['revision'])
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    model.peft_config['default'].base_model_name_or_path=info['model']
    model.peft_config['default'].revision=info['revision']
    params=[p for p in model.parameters() if p.requires_grad]
    def encode(row):
        msg=messages(row['source'],row['style'])
        prompt=tokenizer.apply_chat_template(msg,add_generation_prompt=True,tokenize=True)
        full=tokenizer.apply_chat_template(msg+[dict(role='assistant',content=render(row['target'],row['style']))],tokenize=True)
        assert full[:len(prompt)]==prompt and len(full)>len(prompt)
        assert len(full)<=4096,'No source truncation allowed'
        return full,len(prompt)
    encoded={r['id']:encode(r) for r in train+validation}
    longest=max(train+validation,key=lambda r:len(encoded[r['id']][0]))
    def batch(row):
        full,n=encoded[row['id']]
        x=torch.tensor([full],device='cuda')
        return x,n
    def loss_for(row):
        x,n=batch(row)
        # Position n-1 predicts the first assistant token. Exclude final token's
        # unused logit, retain every assistant token including EOS.
        keep=torch.arange(n-1,x.shape[1]-1,device='cuda')
        logits=model(input_ids=x,attention_mask=torch.ones_like(x),logits_to_keep=keep,use_cache=False).logits
        return F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),x[0,n:]),x.shape[1]-n
    model.eval()
    shortest=min(train+validation,key=lambda r:len(encoded[r['id']][0]))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        x,n=batch(shortest);labels=x.clone();labels[:,:n]=-100
        full_loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=labels,use_cache=False).loss
        small_loss,_=loss_for(shortest)
        difference=abs(float(full_loss)-float(small_loss))
        assert difference<1e-4,'Assistant-only loss alignment mismatch'
    del full_loss,small_loss,x,labels
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=128)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    model.train();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.float16):loss,_=loss_for(longest)
    assert torch.isfinite(loss)
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    norm=torch.nn.utils.clip_grad_norm_(params,1)
    assert torch.isfinite(norm)
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
    assert optimizer.state,'Real optimizer step required'
    probe=dict(seconds=time.perf_counter()-start,loss=float(loss.detach()),tokens=len(encoded[longest['id']][0]),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        loss_equivalence_abs_difference=difference,probe_update_discarded=True)
    (ROOT/'memory_probe.json').write_text(json.dumps(probe,indent=2),encoding='utf-8');print('PROBE',probe,flush=True)
    with torch.no_grad():
        for n,p in model.named_parameters():
            if p.requires_grad:p.copy_(initial[n].to(p.device))
    del initial,loss,optimizer
    torch.manual_seed(42);random.seed(42)
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=128)
    updates=a.epochs*math.ceil(len(train)/4)
    scheduler=get_linear_schedule_with_warmup(optimizer,1,updates)
    smoke_rows=[next(r for r in validation if r['style']==1),next(r for r in validation if r['style']==2),[r for r in validation if r['style']==2][-1]]
    settings=dict(winner=alias,base=info['model'],revision=info['revision'],authorization=selection['authorization'],
        seed=42,epochs=a.epochs,micro_batch=1,gradient_accumulation=4,lr=1e-4,weight_decay=.01,warmup_steps=1,
        rank=8,alpha=16,dropout=.05,gradient_checkpointing=True,compute='FP16',loss='every assistant token only',
        logits_memory_optimization='Qwen3 logits_to_keep; verified equivalent to masked full-token loss',
        train_rows=38,validation_rows=10,max_actual_tokens=len(encoded[longest['id']][0]),
        trainable_parameters=sum(p.numel() for p in params),checkpoint_selection='minimum validation token loss',
        final_holdout_opened=False,smoke_ids=[r['id'] for r in smoke_rows],
        script_sha256=sha(__file__),prompt_script_sha256=sha(PROJECT/'pipeline.py'),
        data_hashes=manifest['files'])
    settings.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
        transformers=importlib.metadata.version('transformers'),peft=importlib.metadata.version('peft'),
        bitsandbytes=importlib.metadata.version('bitsandbytes'),base_weight_files=info['weight_files'])
    (ROOT/'config.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    def validate():
        model.eval();total=tokens=0
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            for r in validation:
                loss,n=loss_for(r);total+=float(loss)*n;tokens+=n
        return total/tokens
    base_validation=validate();print('BASE_VALIDATION',base_validation,flush=True)
    best=float('inf');steps=0;torch.cuda.reset_peak_memory_stats()
    for epoch in range(a.epochs):
        model.train();order=list(train);random.shuffle(order);total=0;start=time.perf_counter()
        for i,row in enumerate(order):
            accumulation=min(4,len(order)-(i//4)*4)
            with torch.autocast('cuda',dtype=torch.float16):raw,_=loss_for(row);loss=raw/accumulation
            assert torch.isfinite(loss)
            total+=float(raw.detach());scaler.scale(loss).backward()
            if (i+1)%4==0 or i+1==len(order):
                scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(params,1)
                assert torch.isfinite(norm)
                scaler.step(optimizer);scaler.update();scheduler.step();optimizer.zero_grad(set_to_none=True);steps+=1
                print('STEP',json.dumps(dict(epoch=epoch+1,step=steps,examples=i+1,loss=float(raw.detach()))),flush=True)
        value=validate()
        record=dict(epoch=epoch+1,train_loss=total/len(train),validation_token_loss=value,base_validation_token_loss=base_validation,
            optimizer_steps=steps,seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        with (ROOT/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        print('EPOCH',record,flush=True)
        if value<best:
            best=value;model.save_pretrained(ROOT/'adapter',safe_serialization=True);tokenizer.save_pretrained(ROOT/'adapter')
            (ROOT/'best_checkpoint.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    # One epoch is the default; loading best is necessary for a multi-epoch request.
    if a.epochs!=1:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model,load_file(str(ROOT/'adapter/adapter_model.safetensors')))
    model.eval();model.config.use_cache=True
    for row in smoke_rows:
        for variant in ('base','lora'):
            msg=messages(row['source'],row['style'])
            inputs=tokenizer.apply_chat_template(msg,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to('cuda')
            torch.manual_seed(42);start=time.perf_counter()
            ctx=model.disable_adapter() if variant=='base' else contextlib.nullcontext()
            with ctx,torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                output=model.generate(**inputs,max_new_tokens=512,do_sample=True,temperature=.6,top_p=.95,top_k=20,
                    repetition_penalty=1,pad_token_id=tokenizer.eos_token_id,use_cache=True)
            tok=output[0,inputs['input_ids'].shape[1]:]
            record=dict(id=row['id'],style=row['style'],variant=variant,source=row['source'],
                response=tokenizer.decode(tok,skip_special_tokens=True),tokens=len(tok),truncated=len(tok)==512,seconds=time.perf_counter()-start)
            with (ROOT/'smoke_before_after.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            print('SMOKE',json.dumps({k:record[k] for k in ('id','variant','tokens','truncated','seconds')}),flush=True)
    weight=ROOT/'adapter/adapter_model.safetensors'
    b_max=max(float(p.detach().abs().max()) for n,p in model.named_parameters() if 'lora_B' in n)
    assert b_max>0,'Saved adapter must contain learned nonzero B weights'
    (ROOT/'artifact.json').write_text(json.dumps(dict(path='adapter/adapter_model.safetensors',size_bytes=weight.stat().st_size,sha256=sha(weight),
        trained=True,base_validation_token_loss=base_validation,best_validation_token_loss=best,optimizer_steps=steps,
        lora_b_max_abs=b_max,independent_quality_improvement_proven=False,final_holdout_opened=False),indent=2),encoding='utf-8')
    print('TRAINING_COMPLETE',weight,flush=True)

if __name__=='__main__':
    import contextlib
    main()
