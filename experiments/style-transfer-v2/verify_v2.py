"""Integrity, leakage and raw-output checks, not an automatic semantic judge."""
import hashlib,json,unittest
from common import ROOT,PROJECT,read,rows,sha,messages

class Checks(unittest.TestCase):
    def test_upstream_license_and_exact_snapshots(self):
        m=read(ROOT/'upstream_manifest.json');self.assertEqual(m['license'],'MIT')
        self.assertFalse(m['source_scripts_executed'])
        for name,digest in m['files'].items():self.assertEqual(sha(ROOT/'upstream/style-lora'/name),digest)

    def test_dataset_group_boundaries_and_synthetic_trace(self):
        train=rows(ROOT/'data/train.jsonl');val=rows(ROOT/'data/validation.jsonl')
        original=rows(PROJECT/'data/train.jsonl')
        self.assertEqual({r['group_id'] for r in train},{r['group_id'] for r in original})
        self.assertEqual({r['id'].removesuffix('_provided') for r in train if r['origin']=='provided_neutral_source'},
                         {r['id'] for r in original})
        self.assertEqual(len(original),38)
        self.assertEqual(len(val),10)
        self.assertFalse({r['group_id'] for r in train}&{r['group_id'] for r in val})
        demos={d['id'] for d in read(ROOT/'data/demos.json').values()}
        self.assertFalse(demos&{r['id'].rsplit('_',1)[0] for r in train+val})
        decisions={r['id']:r for r in read(ROOT/'data/neutralization_review.json')['decisions']}
        self.assertEqual(len(decisions),48)
        for r in train:
            if r['origin']=='style_lora_backtranslation':
                d=decisions[r['id'].removesuffix('_reverse')];self.assertTrue(d['accepted'])
                self.assertEqual(hashlib.sha256(r['raw_neutral'].encode()).hexdigest(),d['raw_sha256'])
        for name,m in read(ROOT/'data/pairs_manifest.json')['files'].items():self.assertEqual(sha(ROOT/'data'/name),m['sha256'])

    def test_trained_export_shape_finiteness_and_previous_adapter(self):
        import torch
        from safetensors import safe_open
        a=read(ROOT/'training/artifact.json');config=read(ROOT/'training/config.json')
        self.assertTrue(a['trained']);self.assertEqual(a['optimizer_steps'],config['expected_steps'])
        self.assertEqual(config['rank'],16);self.assertEqual(sha(ROOT/a['path']),a['sha256'])
        self.assertLess(config['loss_equivalence_abs_difference'],1e-4)
        self.assertLess(a['max_abs_export_rounding'],1e-4)
        with safe_open(ROOT/a['path'],framework='pt',device='cpu') as f:
            self.assertEqual(len(f.keys()),504)
            learned_b=0.0
            for k in f.keys():
                t=f.get_tensor(k);self.assertEqual(t.dtype,torch.float16);self.assertTrue(torch.isfinite(t).all())
                self.assertEqual(t.shape[0] if 'lora_A' in k else t.shape[1],16)
                if 'lora_B' in k:learned_b=max(learned_b,float(t.abs().max()))
            self.assertGreater(learned_b,0,'An initialized zero-B adapter is not a trained result')
        old=read(PROJECT/'training/artifact.json');self.assertEqual(sha(PROJECT/old['path']),old['sha256'])

    def test_chat_format_prefill_matches_training_content_boundary(self):
        from transformers import AutoTokenizer
        from chat_format import prompt_ids
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'adapter')
        override=(ROOT/'inference_chat_template.jinja').read_text(encoding='utf-8')
        for style in (1,2):
            r=next(r for r in rows(ROOT/'data/validation.jsonl') if r['style']==style)
            msg=messages(r['source'],style)
            ids,fixed=prompt_ids(tokenizer,msg)
            full=tokenizer.apply_chat_template(msg+[dict(role='assistant',content=r['target'])],tokenize=True)
            self.assertEqual(ids,full[:len(ids)])
            self.assertTrue(tokenizer.decode(full[len(ids):],skip_special_tokens=True).startswith(r['target']))
            self.assertEqual(ids,tokenizer.apply_chat_template(msg,add_generation_prompt=True,tokenize=True,chat_template=override))
            self.assertEqual(fixed,read(ROOT/'evaluation/runtime_fix.json')['fixed_prefix_ids'])
        config=read(ROOT/'training/config.json')
        self.assertEqual(sha(ROOT/'training/source_snapshots/common.py'),config['prompt_sha256'])
        self.assertEqual(sha(ROOT/'training/source_snapshots/train_v2.py'),config['script_sha256'])

    def test_evaluation_first_attempts_and_same_prompt(self):
        rs=rows(ROOT/'evaluation/raw_comparison.jsonl');self.assertEqual(len(rs),44)
        by={(r['split'],r['id'],r['style'],r['variant']):r for r in rs};self.assertEqual(len(by),44)
        for r in rs:
            other=by[r['split'],r['id'],r['style'],'base' if r['variant']=='lora_v2' else 'lora_v2']
            self.assertEqual(r['messages'],other['messages']);self.assertEqual(r['source'],other['source'])
            self.assertEqual(r['decoding'],'greedy');self.assertTrue(r['response'].strip())
            self.assertFalse(r['truncated']);self.assertLessEqual(r['prompt_tokens']+r['output_tokens'],4096)
            self.assertNotIn('editorial_edits',r)
        m=read(ROOT/'evaluation/generation_manifest.json')
        self.assertFalse(m['output_editorial_edits']);self.assertFalse(m['feedback_or_retuning_from_evaluation'])
        self.assertEqual(sha(ROOT/'common.py'),m['common_sha256'])
        self.assertEqual(sha(ROOT/'chat_format.py'),m['chat_format_sha256'])
        self.assertEqual(sha(ROOT/'adapter/adapter_model.safetensors'),m['adapter_sha256'])
        self.assertEqual(sha(ROOT/'data/fresh_regression.jsonl'),read(ROOT/'evaluation_plan.json')['fresh_regression_sha256'])
        for variant in ('base','lora_v2'):
            for style in (1,2):
                txt=(ROOT/f'evaluation/{variant}/outputs_type{style}.txt').read_text(encoding='utf-8')
                for r in rs:
                    if r['variant']==variant and r['style']==style and r['split']=='provided_test':self.assertIn(r['response'],txt)

    def test_blind_pairs_and_no_fabricated_votes(self):
        k=read(ROOT/'evaluation/blind_ab_key.json');self.assertEqual(len(k),22)
        d=read(ROOT/'evaluation/style_diagnostics.json');self.assertEqual(d['human_votes_received'],0)
        self.assertEqual(d['raw_responses_sha256'],sha(ROOT/'evaluation/raw_comparison.jsonl'))
        self.assertEqual(len({r['item'] for r in k}),22)
        self.assertTrue({r['left'] for r in k}<={'base','lora_v2'})

    def test_semantic_audit_covers_unchanged_raw_outputs(self):
        audit=read(ROOT/'evaluation/source_audit.json')
        self.assertEqual(audit['raw_sha256'],sha(ROOT/'evaluation/raw_comparison.jsonl'))
        self.assertFalse(audit['independent_human_review'])
        rs=rows(ROOT/'evaluation/raw_comparison.jsonl')
        by={(r['split'],r['id'],r['style'],r['variant']):r for r in rs}
        self.assertEqual(len(audit['decisions']),len(by))
        seen=set()
        for d in audit['decisions']:
            k=(d['split'],d['id'],d['style'],d['variant']);self.assertNotIn(k,seen);seen.add(k)
            r=by[k];self.assertTrue(d['reviewed'])
            self.assertEqual(d['response_sha256'],hashlib.sha256(r['response'].encode()).hexdigest())
            self.assertEqual(d['source_sha256'],hashlib.sha256(r['source'].encode()).hexdigest())
            for issue in d['issues']:self.assertIn(issue['quote'],r['response'])
        self.assertFalse(read(ROOT/'evaluation/selection.json')['provided_fact_gate_passed'])

if __name__=='__main__':unittest.main(verbosity=2)
