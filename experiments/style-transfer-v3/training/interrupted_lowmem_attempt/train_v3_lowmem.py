"""Native CUDA/PEFT backend; method and hyperparameter starting point from style-lora (MIT)."""
import argparse,contextlib,hashlib,json,math,random,time
from pathlib import Path
import torch
import torch.nn.functional as F
from common import ROOT,PROJECT,rows,read,sha,dump,append,messages,load_base

def main():
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,set_peft_model_state_dict
    from transformers import get_cosine_schedule_with_warmup
    from safetensors.torch import load_file,save_file
    ap=argparse.ArgumentParser();ap.add_argument('--base-path',required=True);a=ap.parse_args()
    output=ROOT/'training';adapter=ROOT/'adapter'
    assert not adapter.exists() and not (output/'lowmem_completed.json').exists(),'Never overwrite a trained experiment'
    output.mkdir(exist_ok=True);scratch=ROOT/'_checkpoints';scratch.mkdir(exist_ok=True)
    train=rows(ROOT/'data/train.jsonl');val=rows(ROOT/'data/validation.jsonl');manifest=read(ROOT/'data/pairs_manifest.json')
    for n,item in manifest['files'].items():assert sha(ROOT/'data'/n)==item['sha256']
    assert len(val)==10 and len(train)>=38
    assert not ({r['group_id'] for r in train}&{r['group_id'] for r in val})
    lock=read(PROJECT/'model_lock.json')
    for f in lock['weight_files']:assert sha(Path(a.base_path)/f['name'])==f['sha256']
    torch.manual_seed(42);random.seed(42)
    model,tok=load_base(a.base_path)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    # These embeddings are frozen. Restore their inference precision, saving ~0.6 GiB.
    for module in (model.get_input_embeddings(),model.get_output_embeddings()):
        assert not module.weight.requires_grad
        module.weight.data=module.weight.data.to(torch.float16)
    torch.cuda.empty_cache()
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    model.peft_config['default'].base_model_name_or_path=lock['model'];model.peft_config['default'].revision=lock['revision']
    initial_config=read(output/'initial_attempt/config.json')
    initial_epoch=rows(output/'initial_attempt/epochs.jsonl')[0]
    initial_weights=scratch/'best/adapter_model.safetensors'
    assert initial_epoch['epoch']==1 and initial_epoch['optimizer_steps']==21
    initial_sha=sha(initial_weights)
    set_peft_model_state_dict(model,load_file(str(initial_weights)))
    params=[p for p in model.parameters() if p.requires_grad]
    encoded={}
    for r in train+val:
        msg=messages(r['source'],r['style'])
        from chat_format import prompt_ids
        prompt,_=prompt_ids(tok,msg)
        full=tok.apply_chat_template(msg+[dict(role='assistant',content=r['target'])],tokenize=True)
        assert full[:len(prompt)]==prompt and len(full)>len(prompt) and len(full)<=4096
        encoded[r['id']]=(full,len(prompt))
    def loss_for(r,full_logits=False):
        ids,n=encoded[r['id']];x=torch.tensor([ids],device='cuda')
        if full_logits:
            labels=x.clone();labels[:,:n]=-100
            return model(input_ids=x,attention_mask=torch.ones_like(x),labels=labels,use_cache=False).loss,len(ids)-n
        keep=torch.arange(n-1,x.shape[1]-1,device='cuda')
        logits=model(input_ids=x,attention_mask=torch.ones_like(x),logits_to_keep=keep,use_cache=False).logits
        return F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),x[0,n:]),len(ids)-n
    model.eval();short=min(train,key=lambda r:len(encoded[r['id']][0]))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        full,_=loss_for(short,True);small,_=loss_for(short)
        difference=abs(float(full)-float(small));assert difference<1e-4
    del full,small
    def validate():
        model.eval();total=count=0
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            for r in val:
                loss,n=loss_for(r);total+=float(loss)*n;count+=n
        return total/count
    optimizer=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01)
    updates=math.ceil(len(train)/4);warmup=max(1,math.ceil(updates*.05))
    scheduler=get_cosine_schedule_with_warmup(optimizer,warmup,updates)
    scaler=torch.amp.GradScaler('cuda',init_scale=128)
    settings=dict(base=lock,upstream=read(ROOT/'upstream_manifest.json'),method='reviewed compact targets, supported reverse pairs, original anchors, and synthetic invariants',
        rank=8,alpha=16,dropout=.05,epochs=2,lr=1e-4,weight_decay=.01,seed=42,
        scheduler='cosine',warmup_steps=warmup,micro_batch=1,gradient_accumulation=4,expected_steps=42,
        optimizer_restart=True,phase2_lr=5e-5,phase2_steps=updates,
        frozen_embeddings='FP16 in second stage; FP32 in initial stage',
        initial_checkpoint_sha256=initial_sha,initial_training=initial_config,
        deviation=read(output/'initial_attempt/interruption.json'),
        train_rows=len(train),unique_train_groups=len({r['group_id'] for r in train}),validation_rows=len(val),
        max_actual_tokens=max(len(x[0]) for x in encoded.values()),compute='FP16',quantization='NF4',
        gradient_checkpointing=True,assistant_only_loss=True,loss_equivalence_abs_difference=difference,
        trainable_parameters=sum(p.numel() for p in params),gpu=torch.cuda.get_device_name(0),
        prompt_sha256=sha(ROOT/'common.py'),data=manifest,script_sha256=sha(__file__),
        inference_export_dtype='float16; export rounding measured separately',
        checkpoint_selection='minimum token-weighted validation loss; factual and style benefit require generation evaluation')
    dump(output/'config.json',settings)
    with model.disable_adapter():baseline=validate()
    print('BASE_VALIDATION_LOWMEM',baseline,flush=True)
    best=validate();steps=21
    first=dict(initial_epoch,validation_token_loss=best,original_validation_token_loss=initial_epoch['validation_token_loss'],base_validation_token_loss=baseline,precision='frozen embeddings FP16 reevaluation')
    (output/'epochs.jsonl').write_text('',encoding='utf-8');append(output/'epochs.jsonl',first)
    dump(output/'best_checkpoint.json',first)
    model.save_pretrained(scratch/'lowmem_best',safe_serialization=True)
    print('RESTORED_EPOCH1_VALIDATION',best,flush=True)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    for epoch in (1,):
        model.train();order=list(train);random.shuffle(order);total=0;ep_start=time.perf_counter()
        for i,r in enumerate(order):
            count=min(4,len(order)-(i//4)*4)
            with torch.autocast('cuda',dtype=torch.float16):raw,_=loss_for(r);loss=raw/count
            assert torch.isfinite(loss)
            total+=float(raw.detach());scaler.scale(loss).backward()
            if (i+1)%4==0 or i+1==len(order):
                scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(params,1);assert torch.isfinite(norm)
                prior_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                assert scaler.get_scale()>=prior_scale,'Overflow skipped an optimizer step'
                scheduler.step();optimizer.zero_grad(set_to_none=True);steps+=1
                append(output/'steps.jsonl',dict(epoch=epoch+1,step=steps,loss=float(raw.detach())))
                print('STEP',epoch+1,steps,round(float(raw.detach()),4),flush=True)
        value=validate()
        record=dict(epoch=epoch+1,train_loss=total/len(train),validation_token_loss=value,base_validation_token_loss=baseline,
            optimizer_steps=steps,seconds=time.perf_counter()-ep_start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        append(output/'epochs.jsonl',record);print('EPOCH',record,flush=True)
        if value<best:
            best=value;model.save_pretrained(scratch/'lowmem_best',safe_serialization=True)
            dump(output/'best_checkpoint.json',record)
    assert steps==42 and optimizer.state
    # Export the best trained checkpoint in FP16, fitting GitHub's file-size limit.
    state=load_file(str(scratch/'best/adapter_model.safetensors'));delta=0
    for k,v in state.items():
        half=v.half();assert torch.isfinite(half).all()
        delta=max(delta,float((v-half.float()).abs().max()));state[k]=half
    adapter.mkdir();save_file(state,str(adapter/'adapter_model.safetensors'),metadata={'format':'pt'})
    model.peft_config['default'].save_pretrained(adapter);tok.save_pretrained(adapter)
    f=adapter/'adapter_model.safetensors'
    dump(output/'artifact.json',dict(trained=True,path='adapter/adapter_model.safetensors',bytes=f.stat().st_size,
        sha256=sha(f),optimizer_steps=steps,best_validation_token_loss=best,base_validation_token_loss=baseline,
        tensor_count=len(state),export_dtype='FP16',max_abs_export_rounding=delta,
        seconds=time.perf_counter()-started+initial_epoch['seconds'],
        lowmem_phase_seconds=time.perf_counter()-started,
        interrupted_extra_steps_not_retained=read(output/'initial_attempt/interruption.json')['discarded_optimizer_steps'],
        quality_improvement_proven=False))
    dump(output/'lowmem_completed.json',dict(completed=True,retained_optimizer_steps=steps))
    print('TRAINING_COMPLETE_LOWMEM',sha(f),flush=True)
if __name__=='__main__':main()
