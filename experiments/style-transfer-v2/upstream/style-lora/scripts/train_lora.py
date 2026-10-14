"""QLoRA style-adapter training with Unsloth.

Usage:
    uv run scripts/train_lora.py \
        --data artifacts/pairs \
        --base models/qwen3-4b-instruct \
        --out artifacts/adapters/<name> \
        [--rank 16] [--epochs 2] [--max-examples 0] [--lr 2e-4]

Requires the GPU to be free (stop any local model servers first).
"""

import argparse
import json
from pathlib import Path

from unsloth import FastLanguageModel  # noqa: E402  (must import before trl)
from datasets import Dataset
from trl import SFTConfig, SFTTrainer


def load_split(data_dir: Path, split: str, limit: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in (data_dir / f"{split}.jsonl").open()]
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument(
        "--max-examples", type=int, default=0, help="0 = all (use 10 for smoke test)"
    )
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.base),
        max_seq_length=args.seq_len,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    def to_text(rows: list[dict]) -> Dataset:
        texts = [
            tokenizer.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=False
            )
            for r in rows
        ]
        return Dataset.from_dict({"text": texts})

    train_ds = to_text(load_split(args.data, "train", args.max_examples))
    val_ds = to_text(load_split(args.data, "val", args.max_examples))
    print(f"train={len(train_ds)} val={len(val_ds)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(args.out / "checkpoints"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=20,
            save_strategy="no",
            bf16=True,
            seed=42,
            max_length=args.seq_len,
            dataset_text_field="text",
            report_to="none",
        ),
    )
    result = trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    final_eval = trainer.evaluate()
    metrics = {
        "train_loss": result.training_loss,
        "eval_loss": final_eval.get("eval_loss"),
        "rank": args.rank,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_examples": len(train_ds),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
