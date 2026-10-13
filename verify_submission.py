"""Integrity and boundary checks; these do not grade semantic truth."""
import hashlib,json,unittest
from pathlib import Path
from memory_budget import budget
from pipeline import messages
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def rows(name):return [json.loads(s) for s in (ROOT/name).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(name):return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()

class SubmissionChecks(unittest.TestCase):
    def test_checkpoint_is_trained_and_finite(self):
        import torch
        from safetensors import safe_open
        artifact=read('training/artifact.json');config=read('adapter/adapter_config.json')
        self.assertEqual(sha(artifact['path']),artifact['sha256'])
        self.assertEqual((ROOT/artifact['path']).stat().st_size,artifact['size_bytes'])
        self.assertTrue(artifact['trained']);self.assertEqual(artifact['optimizer_steps'],10)
        self.assertGreater(artifact['lora_b_max_abs'],0)
        self.assertEqual(config['revision'],read('model_lock.json')['revision'])
        self.assertEqual(config['r'],8)
        with safe_open(ROOT/artifact['path'],framework='pt',device='cpu') as f:
            self.assertEqual(len(f.keys()),504)
            self.assertTrue(all(torch.isfinite(f.get_tensor(k)).all().item() for k in f.keys()))

    def test_data_separation_and_hashes(self):
        train=rows('data/train.jsonl');val=rows('data/validation.jsonl');demos=read('data/prompt_examples.json')
        originals=rows('data/original/train_type1.jsonl')+rows('data/original/train_type2.jsonl')
        self.assertEqual(len(originals),50);self.assertEqual((len(train),len(val),len(demos)),(38,10,2))
        self.assertFalse({r['group_id'] for r in train}&{r['group_id'] for r in val})
        ids={r['id'] for r in train+val};demo_ids={r['original_id'] for r in demos.values()}
        self.assertFalse(ids&demo_ids);self.assertEqual(len(ids|demo_ids),50)
        self.assertFalse(ids&{r['id'] for r in rows('data/final_holdout.jsonl')})
        for name,item in read('training_data_manifest.json')['files'].items():
            self.assertEqual(sha('data/'+name),item['sha256'])

    def test_frozen_generation_inputs_and_snapshots(self):
        m=read('evaluation/generation_manifest.json')
        for key,name in [('adapter_sha256','adapter/adapter_model.safetensors'),
                         ('prompt_sha256','prompt_template.json'),('test_inputs_sha256','data/test_inputs.jsonl'),
                         ('script_sha256','evaluation/source_snapshots/generate_submission.py'),
                         ('pipeline_sha256','evaluation/source_snapshots/pipeline.py')]:
            self.assertEqual(sha(name),m[key])
        self.assertTrue(m['holdout_checkpoint_and_prompt_frozen_before_read'])
        self.assertFalse(m['holdout_used_for_tuning'])
        self.assertEqual(sha('training/original_train.py'),read('training/config.json')['script_sha256'])
        self.assertEqual(sha('evaluation/final_outputs.jsonl'),read('evaluation/publication_review.json')['final_outputs_sha256'])

    def test_every_provided_source_has_two_traceable_outputs(self):
        sources={r['id']:r['text'] for r in rows('data/test_inputs.jsonl')}
        first=rows('evaluation/generation_first_attempts.jsonl');retry=rows('evaluation/editorial_retry.jsonl')
        final=rows('evaluation/final_outputs.jsonl');expected={(i,s) for i in sources for s in (1,2)}
        for stage in [first,retry,final]:
            self.assertEqual(len(stage),10);self.assertEqual({(r['id'],r['style']) for r in stage},expected)
            for r in stage:
                self.assertEqual(r['source'],sources[r['id']]);self.assertFalse(r['truncated'])
                self.assertTrue(r['response'].strip());self.assertIsNone(r['technical_error'])
        edits=read('evaluation/editorial_edits.json')['edits'];self.assertEqual(len(edits),23)
        retry_by={(r['id'],r['style']):r for r in retry}
        edited=0
        for r in final:
            text=retry_by[r['id'],r['style']]['response'];applied=[]
            for e in edits:
                if (e['id'],e['style'])==(r['id'],r['style']):
                    self.assertEqual(text.count(e['old']),1);text=text.replace(e['old'],e['new']);applied.append(e)
            self.assertEqual(text,r['response']);self.assertEqual(applied,r['editorial_edits'])
            edited+=bool(applied)
            if r['style']==1:self.assertTrue(text.endswith('@banki_oil'));self.assertEqual(text.count('@banki_oil'),1)
        self.assertEqual(edited,9)
        for s in (1,2):
            txt=(ROOT/f'outputs_type{s}.txt').read_text(encoding='utf-8')
            for r in final:
                if r['style']==s:self.assertIn(r['response'],txt)

    def test_negative_holdout_audit_matches_preserved_quotes(self):
        records=rows('evaluation/final_holdout.jsonl');self.assertEqual(len(records),32)
        by={(r['id'],r['style'],r['variant']):r for r in records}
        self.assertEqual(len(by),32)
        audit=read('evaluation/holdout_source_audit.json');counts={'base':set(),'solution':set()}
        for f in audit['findings']:
            r=by[f['id'],f['style'],f['variant']]
            self.assertIn(f['response_quote'],r['response']);self.assertIn(f['source_quote'],r['source'])
            if f['critical']:counts[f['variant']].add((f['id'],f['style']))
        self.assertEqual({k:len(v) for k,v in counts.items()},audit['documented_critical_posts_lower_bound'])
        self.assertFalse(audit['feedback_or_training_from_holdout'])
        checks=rows('evaluation/calibrated_checks.jsonl')
        held=[r for r in checks if r.get('phase')=='final_holdout' or r.get('dataset')=='final_holdout']
        # The summary independently records this judge's failure to reject either variant.
        summary=read('evaluation/calibrated_check_summary.json')
        for v in ('base','solution'):
            self.assertEqual(summary['final_holdout/'+v]['approved'],16)

    def test_source_not_conflated_with_style_demo(self):
        source='Уникальный новый источник: комиссия 2%, минимум 50 рублей.'
        for s in (1,2):
            full=messages(source,s);naive=messages(source,s,naive=True)
            self.assertIn(source,full[-1]['content']);self.assertIn('[S1]',full[-1]['content'])
            self.assertNotIn(source,full[1]['content']);self.assertIn('[EX1]',full[1]['content'])
            self.assertEqual(len(naive),2)
        with self.assertRaises(ValueError):messages('',1)
        with self.assertRaises(ValueError):messages(source,3)

    def test_gqa_memory_and_capacity_boundaries(self):
        b=budget();self.assertEqual(b['head_dim'],128)
        self.assertEqual(b['kv_bytes_per_token'],147456)
        self.assertEqual(b['kv_gib_per_request'],.5625);self.assertEqual(b['kv_gib_total'],8.4375)
        self.assertEqual(b['required_kv_capacity_tokens'],61440)
        self.assertTrue(b['estimated_fits']);self.assertTrue(b['required_runtime_profiling'])
        self.assertFalse(budget(total_gib=8,reserve_gib=1,util=.8)['estimated_fits'])
        self.assertTrue(budget(total_gib=8,reserve_gib=1,util=.8,seqs=1)['estimated_fits'])
        self.assertLessEqual(budget(other_gib=6)['model_executor_budget_gib'],16)
        self.assertEqual(budget(length=4608)['kv_gib_total'],9.4921875)
        with self.assertRaises(ValueError):budget(total_gib=8,other_gib=8)

if __name__=='__main__':unittest.main(verbosity=2)
