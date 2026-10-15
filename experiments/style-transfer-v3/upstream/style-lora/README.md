# style-lora

Train a small LoRA adapter that writes in a specific author's voice, using
**back-translation**: no paired data is required, only a corpus of text by the
author. A local LLM rewrites each paragraph into bland neutral prose, and the
adapter is trained on the reverse direction (neutral -> original), learning
"given this content, say it the way the author would."

The whole pipeline runs locally: data generation against any OpenAI-compatible
chat server (e.g. llama.cpp's `llama-server`), QLoRA training via
[Unsloth](https://github.com/unslothai/unsloth) (fits a 4B model comfortably
on a single 24 GB GPU), and evaluation with an embedding-based authorship
discriminator plus a blind human A/B page.

## Pipeline

```text
corpus.txt
  -> build_style_pairs.py   paragraphs -> neutral rewrites -> train/val/test JSONL
  -> train_lora.py          QLoRA adapter (Unsloth), metrics.json
  -> convert_lora_to_gguf   (llama.cpp) adapter -> GGUF
  -> llama-server           base GGUF + --lora adapter GGUF
  -> gen_samples.py         base-arm and adapter-arm generations on the test split
  -> eval_style.py          discriminator AUC, parroting rate, blind A/B html
```

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/), a CUDA GPU (~10 GB free for a
4B base model), a llama.cpp checkout for conversion/serving, and any
OpenAI-compatible chat endpoint for the neutralization pass.

```bash
uv sync

# 1. Get a corpus (any plain-text author corpus works; strip boilerplate).
#    Example: Mark Twain, Adventures of Huckleberry Finn from Project Gutenberg.
curl -sL https://www.gutenberg.org/cache/epub/76/pg76.txt -o artifacts/corpus/raw.txt
# ...strip the Gutenberg header/footer, save as artifacts/corpus/corpus.txt

# 2. Build neutral->original pairs (resumable; point --api-url at your server).
uv run scripts/build_style_pairs.py \
  --corpus artifacts/corpus/corpus.txt --out artifacts/pairs \
  --api-url http://127.0.0.1:8081/v1/chat/completions

# 3. Download a base model and train (smoke-test with --max-examples 10 first).
uv run scripts/train_lora.py \
  --data artifacts/pairs --base models/<hf-base-model> \
  --out artifacts/adapters/run1 --rank 16 --epochs 2

# 4. Convert base and adapter to GGUF (paths per your llama.cpp checkout).
python <llama.cpp>/convert_hf_to_gguf.py models/<hf-base-model> \
  --outfile models/base-f16.gguf --outtype f16
python <llama.cpp>/convert_lora_to_gguf.py artifacts/adapters/run1 \
  --base models/<hf-base-model> --outfile artifacts/adapters/run1/adapter.gguf

# 5. Serve both arms from one server (adapter toggled per pass).
<llama.cpp>/llama-server -m models/base-f16.gguf \
  --lora artifacts/adapters/run1/adapter.gguf --port 8083

# 6. Generate eval samples and score them.
uv run scripts/gen_samples.py --data artifacts/pairs --out artifacts/eval/samples.jsonl
uv run scripts/eval_style.py --samples artifacts/eval/samples.jsonl \
  --data artifacts/pairs --out artifacts/eval
```

Open `artifacts/eval/ab.html` for blind human judging (answer key in
`ab_key.json`) and `artifacts/eval/report.json` for metrics.

## Metrics

- **Discriminator AUC**: embeddings of real author text vs model output, fed
  to a logistic regression (5-fold CV). 1.0 = trivially distinguishable from
  the author; lower = better style match. Compare the adapter arm against the
  base arm.
- **Parroting rate**: fraction of generated 8-grams appearing verbatim in the
  training set — a memorization detector. If it spikes, reduce epochs or rank.
- **Blind A/B**: the honest metric. Sides are shuffled per item; judge first,
  then look at the key.

## Notes

- The system prompt used to build training pairs and the one used at
  inference must match (`scripts/prompts.py`).
- Splits are deterministic by content hash, so re-running data generation
  never contaminates the test set.
- Back-translation occasionally mis-paraphrases idioms; this is tolerable
  noise because the training target is always genuine author text.

## License

MIT. Author corpora fetched from Project Gutenberg are public domain in the
US; check the status of any other corpus you use.
