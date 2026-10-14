"""Build style-transfer SFT pairs by back-translation.

Segments an author corpus into paragraphs, asks a local llama-server to
rewrite each into neutral modern prose, and emits chat-format JSONL training
the reverse direction (neutral -> original author voice).

Resumable: already-neutralized paragraphs (by content hash) are skipped.

Usage:
    uv run scripts/build_style_pairs.py \
        --corpus artifacts/corpus/corpus.txt \
        --out artifacts/pairs \
        [--api-url http://127.0.0.1:8081/v1/chat/completions] \
        [--max-pairs 700]
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import httpx

from prompts import NEUTRALIZE_SYSTEM, STYLE_SYSTEM

CHANNEL_TAG = re.compile(r"<\|channel\>.*?<channel\|\>", re.S)
# Generic front-matter/heading patterns for Gutenberg-style texts.
HEADING = re.compile(
    r"^(CHAPTER|BOOK|PART|ACT|SCENE|CONTENTS|ILLUSTRATIONS?|PREFACE|"
    r"EXPLANATORY|NOTICE|APPENDIX|FOOTNOTES?)\b",
    re.I,
)


def paragraphs(corpus: Path, min_words: int, max_words: int) -> list[str]:
    text = corpus.read_text(encoding="utf-8")
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    keep = []
    for p in paras:
        n = len(p.split())
        if n < min_words or n > max_words:
            continue
        if HEADING.match(p) or p.isupper():
            continue
        keep.append(p)
    return keep


def neutralize(client: httpx.Client, api_url: str, text: str) -> str:
    body = {
        "messages": [
            {"role": "system", "content": NEUTRALIZE_SYSTEM},
            {"role": "user", "content": text},
        ],
        "reasoning_format": "none",
        "thinking_budget_tokens": 0,
        "temperature": 0.3,
        "max_tokens": 2 * len(text.split()) + 150,
    }
    r = client.post(api_url, json=body, timeout=180)
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    out = CHANNEL_TAG.sub("", out)
    out = re.sub(r"^(here is|here's)[^:]*:\s*", "", out.strip(), flags=re.I)
    return out.strip()


def split_of(text: str) -> str:
    h = int(hashlib.md5(text.encode()).hexdigest(), 16) % 100
    return "train" if h < 90 else ("val" if h < 95 else "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--api-url", default="http://127.0.0.1:8081/v1/chat/completions")
    ap.add_argument("--max-pairs", type=int, default=700)
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--max-words", type=int, default=300)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / "pairs.raw.jsonl"

    done: dict[str, dict] = {}
    if raw_path.exists():
        with raw_path.open() as f:
            for line in f:
                rec = json.loads(line)
                done[rec["hash"]] = rec

    paras = paragraphs(args.corpus, args.min_words, args.max_words)[: args.max_pairs]
    todo = [p for p in paras if hashlib.md5(p.encode()).hexdigest() not in done]
    print(f"paragraphs={len(paras)} done={len(paras) - len(todo)} todo={len(todo)}")

    client = httpx.Client()
    errors = 0
    t0 = time.time()
    with raw_path.open("a") as f:
        for i, p in enumerate(todo):
            try:
                neutral = neutralize(client, args.api_url, p)
            except Exception:
                time.sleep(2)
                try:
                    neutral = neutralize(client, args.api_url, p)
                except Exception as e:
                    errors += 1
                    print(
                        f"error on paragraph {i}: {type(e).__name__}", file=sys.stderr
                    )
                    continue
            if not neutral or len(neutral.split()) < args.min_words // 2:
                errors += 1
                continue
            rec = {
                "hash": hashlib.md5(p.encode()).hexdigest(),
                "original": p,
                "neutral": neutral,
                "split": split_of(p),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta_min = (len(todo) - i - 1) / rate / 60
                print(
                    f"progress {i + 1}/{len(todo)} eta={eta_min:.0f}m", file=sys.stderr
                )

    # Re-read everything and emit final per-split chat-format files.
    counts = {"train": 0, "val": 0, "test": 0}
    writers = {s: (args.out / f"{s}.jsonl").open("w") for s in counts}
    with raw_path.open() as f:
        for line in f:
            rec = json.loads(line)
            msg = {
                "messages": [
                    {"role": "system", "content": STYLE_SYSTEM},
                    {"role": "user", "content": rec["neutral"]},
                    {"role": "assistant", "content": rec["original"]},
                ]
            }
            writers[rec["split"]].write(json.dumps(msg) + "\n")
            counts[rec["split"]] += 1
    for w in writers.values():
        w.close()
    print(f"pairs: {counts} errors={errors}")


if __name__ == "__main__":
    main()
