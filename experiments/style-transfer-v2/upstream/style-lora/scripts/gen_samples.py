"""Generate base-vs-adapter restyle samples for the frozen test split.

Assumes a llama-server started with the base GGUF and --lora <adapter.gguf>.
Toggles the adapter scale through POST /lora-adapters between passes, so a
single server produces both arms.

Usage:
    uv run scripts/gen_samples.py \
        --data artifacts/pairs \
        --out artifacts/eval/samples.jsonl \
        [--server http://127.0.0.1:8083] [--limit 20]
"""

import argparse
import json
from pathlib import Path

import httpx

from prompts import STYLE_SYSTEM


def set_lora_scale(client: httpx.Client, server: str, scale: float) -> None:
    r = client.get(f"{server}/lora-adapters")
    r.raise_for_status()
    adapters = r.json()
    if not adapters:
        raise SystemExit("server has no --lora adapter loaded")
    for a in adapters:
        a["scale"] = scale
    client.post(f"{server}/lora-adapters", json=adapters).raise_for_status()


def restyle(client: httpx.Client, server: str, neutral: str) -> str:
    body = {
        "messages": [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": neutral},
        ],
        "temperature": 0.7,
        "max_tokens": 2 * len(neutral.split()) + 150,
    }
    r = client.post(f"{server}/v1/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8083")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (args.data / "pairs.raw.jsonl").open()]
    tests = [r for r in rows if r["split"] == "test"][: args.limit]
    print(f"test examples: {len(tests)}")

    client = httpx.Client()
    samples = []
    for arm, scale in [("base", 0.0), ("adapter", 1.0)]:
        set_lora_scale(client, args.server, scale)
        for i, rec in enumerate(tests):
            out = restyle(client, args.server, rec["neutral"])
            samples.append(
                {
                    "idx": i,
                    "arm": arm,
                    "neutral": rec["neutral"],
                    "original": rec["original"],
                    "output": out,
                }
            )
        print(f"{arm}: {len(tests)} generations done")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(samples)} samples")


if __name__ == "__main__":
    main()
