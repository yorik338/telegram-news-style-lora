"""Verify the chosen low-memory checkpoint, export it, and measure exact rounding.

The executed restart trainer retained the original export path from the first
trainer. This finalizer corrects that path before any evaluation; it never uses
generated answers. The erroneous first-epoch export is preserved in scratch.
"""
from common import ROOT,read,rows,sha,dump

def main():
 import torch
 from safetensors.torch import load_file,save_file
 assert not (ROOT/'evaluation/generation_manifest.json').exists(),'Never change evaluated weights'
 assert not (ROOT/'training/export_verification.json').exists(),'Final export is immutable'
 artifact=read(ROOT/'training/artifact.json');best=read(ROOT/'training/best_checkpoint.json')
 assert best['validation_token_loss']==min(r['validation_token_loss'] for r in rows(ROOT/'training/epochs.jsonl'))
 selected=ROOT/'_checkpoints/lowmem_best/adapter_model.safetensors'
 state=load_file(str(selected));rounded={};delta=0
 assert len(state)==504
 for name,value in state.items():
  half=value.half();assert torch.isfinite(half).all()
  rounded[name]=half;delta=max(delta,float((value-half.float()).abs().max()))
 export=ROOT/'adapter/adapter_model.safetensors';previous_sha=sha(export)
 backup=ROOT/'_checkpoints/first_epoch_export';backup.mkdir(exist_ok=True)
 import shutil
 shutil.copy2(export,backup/export.name)
 dump(ROOT/'training/first_export_artifact.json',artifact)
 save_file(rounded,str(export),metadata={'format':'pt'})
 loaded=load_file(str(export))
 assert loaded.keys()==rounded.keys()
 assert all(torch.equal(loaded[k],rounded[k]) for k in rounded)
 record=dict(selected_epoch=best['epoch'],selected_optimizer_steps=best['optimizer_steps'],
   selected_fp32_checkpoint_sha256=sha(selected),export_sha256=sha(export),
   previous_export_sha256=previous_sha,all_tensors_equal_selected_checkpoint_rounded_to_fp16=True,
   tensor_count=len(rounded),max_abs_export_rounding=delta,
   export_path_correction_before_any_generation=True,evaluation_outputs_used=False)
 dump(ROOT/'training/export_verification.json',record)
 artifact.update(sha256=sha(export),bytes=export.stat().st_size,max_abs_export_rounding=delta,
   selected_epoch=best['epoch'],selected_optimizer_steps=best['optimizer_steps'],export_verified_against_selected_checkpoint=True)
 dump(ROOT/'training/artifact.json',artifact)
 print('SELECTED_EXPORT_VERIFIED',record,flush=True)

if __name__=='__main__':main()
