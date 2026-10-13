"""Thin local Transformers adapter around NVIDIA's installed self_check_facts action.

No word matching, custom factual classifier, or expected-answer text enters the check.
The factual verdict is produced by NeMo + the configured judge and evidence prompt.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import re
import time
from pathlib import Path

from nemoguardrails import RailsConfig
from nemoguardrails.library.self_check.facts.actions import self_check_facts
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.types import LLMResponse, UsageInfo

ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT.parent / 'telegram-style-benchmark'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def verdict_word(text):
    """Only parse an explicit verdict word; this does not assess any factual content."""
    match = re.match(r'^(yes|no|да|нет)(?=$|[\s,.;:!])', text.strip().lower())
    return {'yes': 'yes', 'да': 'yes', 'no': 'no', 'нет': 'no'}.get(match[1]) if match else None


def nemo_verdict_parser(text):
    # NeMo self_check_facts inverts the parser's first boolean: yes -> False.
    # Unknown is blocked upstream and remains an unknown result in our audit.
    return [verdict_word(text) != 'yes']


class LocalJudge:
    provider_name = 'local-transformers'
    provider_url = None

    def __init__(self, entry):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError('This recorded local configuration requires CUDA')
        self.model_name = entry['model']
        path, revision = entry['local_model_path'], entry['revision']
        config = AutoConfig.from_pretrained(path, revision=revision)
        kwargs = dict(config=config, dtype=torch.float16, device_map={'': 0}, attn_implementation='sdpa')
        if getattr(config, 'quantization_config', None):
            config.quantization_config['bnb_4bit_compute_dtype'] = 'float16'
        else:
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(path, revision=revision, **kwargs).eval()
        self.last_call = None

    async def generate_async(self, prompt, *, stop=None, **kwargs):
        import torch
        messages = [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else [m.to_dict() for m in prompt]
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors='pt').to('cuda')
        n = encoded['input_ids'].shape[-1]
        budget = int(kwargs.get('max_tokens', 32))
        if n + budget > 4096:
            raise ValueError('Evidence exceeds context; refuse silent source truncation')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            stop_kwargs = {'stop_strings': stop, 'tokenizer': self.tokenizer} if stop else {}
            result = self.model.generate(
                **encoded, max_new_tokens=budget, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id, **stop_kwargs)
        torch.cuda.synchronize()
        tokens = result[0, n:]
        raw = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        finish = 'length' if len(tokens) == budget else 'stop'
        self.last_call = dict(raw_verdict=raw, finish_reason=finish, messages=messages,
                              seconds=time.perf_counter()-started, prompt_tokens=n,
                              output_tokens=len(tokens), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
        return LLMResponse(content=raw, model=self.model_name, finish_reason=finish,
                           usage=UsageInfo(input_tokens=n, output_tokens=len(tokens), total_tokens=n+len(tokens)))

    async def stream_async(self, prompt, *, stop=None, **kwargs):
        # The fact-check action uses nonstreaming inference.
        raise NotImplementedError('Use generate_async for this local fact-check adapter')
        yield


class GroundingChecker:
    def __init__(self, judge, config_path=None):
        self.judge = judge
        self.config = RailsConfig.from_path(str(config_path or ROOT / 'config-v3'))
        self.manager = LLMTaskManager(self.config)
        self.manager.register_output_parser(nemo_verdict_parser, 'grounding_verdict')

    async def check(self, sources, response):
        # NeMo intentionally skips checking when evidence is absent. Our use case
        # requires evidence, so that bypass must not count as successful grounding.
        if not sources or any(not isinstance(s, str) or not s.strip() for s in sources):
            return {'supported': None, 'status': 'missing_sources'}
        if not isinstance(response, str) or not response.strip():
            return {'supported': None, 'status': 'missing_response'}
        self.judge.last_call = None
        try:
            outcome = await self_check_facts(
                llm_task_manager=self.manager, llm=self.judge, config=self.config,
                relevant_chunks=sources, bot_message=response)
            trace = self.judge.last_call or {}
            token = verdict_word(trace.get('raw_verdict', ''))
            if trace.get('finish_reason') == 'length' or token not in ('yes', 'no'):
                return {'supported': None, 'status': 'invalid_verdict', **trace}
            return {'supported': not outcome.is_blocked, 'status': 'checked',
                    'parsed_verdict': token, 'nemo_decision': outcome.decision.value, **trace}
        except Exception as error:
            return {'supported': None, 'status': 'technical_error',
                    'error': f'{type(error).__name__}: {error}', **(self.judge.last_call or {})}


async def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge', default='qwen3_4b', choices=['qwen25_1_5b', 'qwen25_3b', 'qwen3_4b'])
    parser.add_argument('--mode', choices=['checks', 'saved-posts', 'grounded-posts'], default='checks')
    parser.add_argument('--config', type=Path, default=ROOT / 'config-v3')
    parser.add_argument('--generation-names', nargs='+', help='Completed generation directories for grounded-posts')
    parser.add_argument('--name', required=True, help='New run directory; existing results never overwritten')
    args = parser.parse_args()
    directory = ROOT / 'runs' / args.name
    if directory.exists():
        raise RuntimeError(f'Run already exists: {directory}')
    models = json.loads((BENCHMARK / 'candidates.lock.json').read_text(encoding='utf-8'))['models']
    entry = next(m for m in models if m['alias'] == args.judge)
    inputs = []
    if args.mode == 'checks':
        inputs = read_rows(ROOT / 'checker_checks.jsonl')
    elif args.mode == 'saved-posts':
        sources = {r['id']: r['source'] for r in read_rows(BENCHMARK / 'data/dev.jsonl')}
        for model_entry in models:
            alias = model_entry['alias'] + '_v3'
            for record in read_rows(BENCHMARK / 'runs' / alias / 'dev.jsonl'):
                inputs.append({'id': record['id'], 'generator': alias, 'style': record['style'],
                               'sources': [sources[record['id']]], 'response': record['text'],
                               'generation_seconds': record.get('seconds'),
                               'generation_truncated': record.get('truncated'),
                               'generation_error': record.get('technical_error') or record.get('format_error')})
    else:
        if not args.generation_names:
            raise ValueError('grounded-posts requires --generation-names')
        for name in args.generation_names:
            source_directory = ROOT/'generations'/name
            source_manifest = json.loads((source_directory/'manifest.json').read_text(encoding='utf-8'))
            source_rows = read_rows(source_directory/'posts.jsonl')
            if len(source_rows) != source_manifest['cases']:
                raise ValueError('Do not evaluate an incomplete generation experiment')
            inputs.extend(source_rows)
    manifest = dict(version=3, mode=args.mode, cases=len(inputs), judge=entry,
                    judge_also_in_generator_pool=True, independent_human_validation=False,
                    tool='NeMo Guardrails self_check_facts', nemoguardrails=importlib.metadata.version('nemoguardrails'),
                    upstream_action_sha256=digest(inspect.getfile(self_check_facts)),
                    script_sha256=digest(__file__),
                    config_hashes={p.name: digest(p) for p in args.config.glob('*.yml')},
                    input_sha256=hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                    decoding={'do_sample': False, 'max_tokens': 32, 'context': 4096, 'stop_strings': ['\n']},
                    score_semantics='Binary source support verdict; not probability of real-world truth')
    directory.mkdir(parents=True)
    import shutil
    (directory / 'config-snapshot').mkdir()
    for path in args.config.glob('*.yml'):
        shutil.copyfile(path, directory / 'config-snapshot' / path.name)
    shutil.copyfile(__file__, directory / 'check_grounding.py')
    (directory / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    judge = LocalJudge(entry)
    checker = GroundingChecker(judge, args.config)
    warmup = await checker.check(['Библиотека открыта.'], 'Библиотека работает.')
    if warmup['status'] != 'checked':
        raise RuntimeError(f'Warmup failed: {warmup}')
    for row in inputs:
        verdict = await checker.check(row['sources'], row['response'])
        result = {**row, **verdict}
        with (directory / 'results.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(result, ensure_ascii=False)+'\n')
        print(json.dumps({key: result.get(key) for key in ('id', 'generator', 'style', 'supported', 'expected_supported', 'status', 'seconds')}, ensure_ascii=False), flush=True)
    print(json.dumps({'completed': len(inputs), 'directory': str(directory)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
