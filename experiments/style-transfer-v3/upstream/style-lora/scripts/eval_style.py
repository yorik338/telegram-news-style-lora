"""Evaluate style adapter: authorship discriminator, parroting check, A/B page.

Metrics:
- discriminator AUC: embeddings (real author text vs model output) fed to
  logistic regression, 5-fold CV. Lower AUC for an arm = harder to tell its
  output from real author text = better style match.
- parroting: fraction of generated 8-grams that appear verbatim in the
  training set (memorization detector).
- ab.html: shuffled blind A/B page for human judging.

Usage:
    uv run scripts/eval_style.py \
        --samples artifacts/eval/samples.jsonl \
        --data artifacts/pairs \
        --out artifacts/eval
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = text.lower().split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def discriminator_auc(model, real: list[str], fake: list[str]) -> float:
    emb = model.encode(real + fake, show_progress_bar=False)
    y = np.array([1] * len(real) + [0] * len(fake))
    scores = cross_val_score(
        LogisticRegression(max_iter=1000), emb, y, cv=5, scoring="roc_auc"
    )
    return float(scores.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    samples = [json.loads(l) for l in args.samples.open()]
    by_arm = {"base": [], "adapter": []}
    for s in samples:
        by_arm[s["arm"]].append(s)
    for arm in by_arm:
        by_arm[arm].sort(key=lambda s: s["idx"])

    rows = [json.loads(l) for l in (args.data / "pairs.raw.jsonl").open()]
    real_texts = [r["original"] for r in rows if r["split"] in ("val", "test")]
    train_grams = set()
    for r in rows:
        if r["split"] == "train":
            train_grams |= ngrams(r["original"], 8)

    st = SentenceTransformer("all-MiniLM-L6-v2")
    report = {}
    for arm, arm_samples in by_arm.items():
        outputs = [s["output"] for s in arm_samples]
        overlaps = []
        for o in outputs:
            grams = ngrams(o, 8)
            overlaps.append(len(grams & train_grams) / len(grams) if grams else 0.0)
        report[arm] = {
            "n": len(outputs),
            "discriminator_auc_vs_real": discriminator_auc(st, real_texts, outputs),
            "parroting_8gram_rate_mean": float(np.mean(overlaps)),
            "parroting_8gram_rate_max": float(np.max(overlaps)),
            "mean_len_ratio_vs_original": float(
                np.mean(
                    [
                        len(s["output"].split()) / max(len(s["original"].split()), 1)
                        for s in arm_samples
                    ]
                )
            ),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # Blind A/B page: adapter vs base, sides shuffled per item.
    rng = random.Random(42)
    items, key = [], []
    for b, a in zip(by_arm["base"], by_arm["adapter"]):
        left_is_adapter = rng.random() < 0.5
        left, right = (a, b) if left_is_adapter else (b, a)
        key.append("left" if left_is_adapter else "right")
        items.append((b["neutral"], left["output"], right["output"]))
    html = [
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:sans-serif;max-width:60em;margin:2em auto}"
        ".n{color:#666;font-style:italic}.p{display:flex;gap:1em}"
        ".c{flex:1;border:1px solid #ccc;padding:1em;border-radius:6px}"
        "</style></head><body><h1>Blind A/B: which sounds like the author?</h1>"
    ]
    for i, (neutral, left, right) in enumerate(items):
        html.append(
            f"<h3>#{i}</h3><p class='n'>{neutral}</p>"
            f"<div class='p'><div class='c'><b>A</b><p>{left}</p></div>"
            f"<div class='c'><b>B</b><p>{right}</p></div></div>"
        )
    html.append("</body></html>")
    (args.out / "ab.html").write_text("\n".join(html))
    (args.out / "ab_key.json").write_text(json.dumps({"adapter_side": key}))
    print(f"wrote {args.out / 'ab.html'} ({len(items)} items); key in ab_key.json")


if __name__ == "__main__":
    main()
