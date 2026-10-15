"""Reproducibility and integrity tests; no proxy claim of semantic accuracy."""
import argparse,hashlib,unittest
from common import ROOT,PROJECT,read,rows,sha,messages

class PreparationChecks(unittest.TestCase):
 def test_frozen_splits_and_supported_reverse_pairs(self):
  plan=read(ROOT/'evaluation_plan.json');train=rows(ROOT/'data/train.jsonl');val=rows(ROOT/'data/validation.jsonl')
  self.assertEqual(sha(ROOT/'data/train.jsonl'),plan['train_sha256'])
  self.assertEqual(sha(ROOT/'data/validation.jsonl'),plan['validation_sha256'])
  self.assertEqual(sha(ROOT/'data/fresh_test.jsonl'),plan['fresh_sha256'])
  self.assertEqual(len(train),82);self.assertEqual(len(val),10)
  self.assertEqual(len({r['group_id'] for r in train}),47)
  self.assertFalse({r['group_id'] for r in train}&{r['group_id'] for r in val})
  self.assertEqual(len([r for r in train if r['origin']=='style_lora_backtranslation']),20)
  self.assertEqual(len([r for r in train if r['origin']=='AI_synthetic_invariant_training']),24)
  self.assertEqual(len(read(ROOT/'data/curation_review.json')['decisions']),48)
  fresh=rows(ROOT/'data/fresh_test.jsonl')
  self.assertEqual(len(fresh),8)
  self.assertFalse({r['source'] for r in train+val}&{r['source'] for r in fresh})
  demos={r['id'] for r in read(ROOT/'data/demos.json').values()}
  self.assertFalse(demos&{r['id'].rsplit('_',1)[0] for r in train+val})
  bad={'type1_13_reverse','type1_14_reverse','type1_19_reverse'}
  self.assertFalse(bad&{r['id'] for r in train})
  driver=next(r for r in train if r['id']=='type2_15_provided')
  self.assertIn('нарушителя по паспорту',driver['target']);self.assertNotIn('нарушителям',driver['target'])

 def test_upstream_and_previous_experiment_preserved(self):
  m=read(ROOT/'upstream_manifest.json');self.assertEqual(m['license'],'MIT')
  for name,digest in m['files'].items():self.assertEqual(sha(ROOT/'upstream/style-lora'/name),digest)
  old=ROOT.parent/'style-transfer-v2'
  self.assertEqual(sha(old/'adapter/adapter_model.safetensors'),read(ROOT/'evaluation_plan.json')['prior_adapter_sha256'])
  self.assertEqual(sha(old/'evaluation/raw_comparison.jsonl'),read(old/'evaluation/source_audit.json')['raw_sha256'])

 def test_shared_prompt_content_boundary(self):
  from transformers import AutoTokenizer
  from chat_format import prompt_ids
  tok=AutoTokenizer.from_pretrained(ROOT.parent/'style-transfer-v2/adapter')
  template=(ROOT/'inference_chat_template.jinja').read_text(encoding='utf-8')
  for r in rows(ROOT/'data/validation.jsonl'):
   msg=messages(r['source'],r['style']);ids,fixed=prompt_ids(tok,msg)
   full=tok.apply_chat_template(msg+[dict(role='assistant',content=r['target'])],tokenize=True)
   self.assertEqual(ids,full[:len(ids)])
   self.assertEqual(ids,tok.apply_chat_template(msg,add_generation_prompt=True,tokenize=True,chat_template=template))
   self.assertTrue(tok.decode(full[len(ids):],skip_special_tokens=True).startswith(r['target']))
   self.assertEqual(fixed,[151667,271,151668,271])

class CompletedChecks(unittest.TestCase):
 def test_trained_best_export(self):
  import torch
  from safetensors import safe_open
  cfg=read(ROOT/'training/config.json');artifact=read(ROOT/'training/artifact.json')
  self.assertEqual(cfg['rank'],8);self.assertEqual(cfg['lr'],1e-4)
  self.assertEqual(artifact['optimizer_steps'],42);self.assertTrue(artifact['trained'])
  self.assertEqual(sha(ROOT/'adapter/adapter_model.safetensors'),artifact['sha256'])
  self.assertLess(cfg['loss_equivalence_abs_difference'],1e-4)
  self.assertEqual(sha(ROOT/'common.py'),cfg['prompt_sha256'])
  self.assertEqual(sha(ROOT/'train_v3_lowmem.py'),cfg['script_sha256'])
  self.assertTrue(cfg['optimizer_restart']);self.assertEqual(cfg['phase2_lr'],5e-5)
  self.assertEqual([r['step'] for r in rows(ROOT/'training/phase2_steps.jsonl')],list(range(22,43)))
  self.assertEqual(sha(ROOT/'training/initial_attempt/train_v3.py'),cfg['initial_training']['script_sha256'])
  epoch=read(ROOT/'training/best_checkpoint.json')
  self.assertEqual(epoch['validation_token_loss'],min(e['validation_token_loss'] for e in rows(ROOT/'training/epochs.jsonl')))
  export=read(ROOT/'training/export_verification.json')
  self.assertTrue(export['all_tensors_equal_selected_checkpoint_rounded_to_fp16'])
  self.assertEqual(export['selected_epoch'],epoch['epoch'])
  self.assertEqual(export['export_sha256'],artifact['sha256'])
  checkpoint=ROOT/'_checkpoints/lowmem_best/adapter_model.safetensors'
  if checkpoint.exists():
   from safetensors.torch import load_file
   full=load_file(str(checkpoint));half=load_file(str(ROOT/'adapter/adapter_model.safetensors'))
   self.assertTrue(all(torch.equal(full[k].half(),half[k]) for k in full))
  max_b=0
  with safe_open(ROOT/'adapter/adapter_model.safetensors',framework='pt',device='cpu') as f:
   self.assertEqual(len(f.keys()),504)
   for k in f.keys():
    t=f.get_tensor(k);self.assertTrue(torch.isfinite(t).all());self.assertEqual(t.dtype,torch.float16)
    self.assertEqual(t.shape[0] if 'lora_A' in k else t.shape[1],8)
    if 'lora_B' in k:max_b=max(max_b,float(t.abs().max()))
  self.assertGreater(max_b,0)

 def test_same_evidence_and_raw_first_responses(self):
  rs=rows(ROOT/'evaluation/raw_comparison.jsonl');self.assertEqual(len(rs),52)
  by={(r['split'],r['id'],r['style'],r['variant']):r for r in rs};self.assertEqual(len(by),52)
  m=read(ROOT/'evaluation/generation_manifest.json')
  self.assertEqual(m['common_sha256'],sha(ROOT/'common.py'))
  self.assertEqual(m['script_sha256'],sha(ROOT/'evaluate_v3.py'))
  self.assertFalse(m['output_editorial_edits']);self.assertFalse(m['feedback_or_retuning_from_evaluation'])
  for r in rs:
   other=by[r['split'],r['id'],r['style'],'base' if r['variant']=='lora_v3' else 'lora_v3']
   self.assertEqual(r['messages'],other['messages']);self.assertEqual(r['source'],other['source'])
   self.assertTrue(r['response'].strip());self.assertFalse(r['truncated'])
   self.assertLessEqual(r['prompt_tokens']+r['output_tokens'],4096)
   self.assertEqual(r['decoding'],'greedy');self.assertEqual(r['inference_batch'],1)
   text=(ROOT/f"evaluation/{r['variant']}/{r['split']}_type{r['style']}.txt").read_text(encoding='utf-8')
   self.assertIn(r['response'],text)

 def test_audit_and_blind_assessment_integrity(self):
  rs=rows(ROOT/'evaluation/raw_comparison.jsonl');audit=read(ROOT/'evaluation/source_audit.json')
  self.assertEqual(audit['raw_sha256'],sha(ROOT/'evaluation/raw_comparison.jsonl'))
  self.assertFalse(audit['independent_human_review']);self.assertEqual(len(audit['decisions']),52)
  by={(r['split'],r['id'],r['style'],r['variant']):r for r in rs};seen=set()
  for d in audit['decisions']:
   k=(d['split'],d['id'],d['style'],d['variant']);self.assertNotIn(k,seen);seen.add(k);r=by[k]
   self.assertEqual(d['response_sha256'],hashlib.sha256(r['response'].encode()).hexdigest())
   for issue in d['issues']:self.assertIn(issue['quote'],r['response'])
  diag=read(ROOT/'evaluation/style_diagnostics.json')
  self.assertEqual(diag['human_votes_received'],0);self.assertEqual(diag['blind_pairs'],26)
  self.assertEqual(len(read(ROOT/'evaluation/blind_ab_key.json')),26)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepared-only',action='store_true');args=parser.parse_args()
 loader=unittest.TestLoader();suite=loader.loadTestsFromTestCase(PreparationChecks)
 if not args.prepared_only:suite.addTests(loader.loadTestsFromTestCase(CompletedChecks))
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 raise SystemExit(not result.wasSuccessful())
