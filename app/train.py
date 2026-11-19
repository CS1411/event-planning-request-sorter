"""Milestone 3: fine-tune GPT-2 on the example requests with the Hugging Face Trainer. `python -m app.train`.

Each request becomes one string, "<request text> <Request Type>", and the model learns to produce the final
token. The loss is masked to that single position (every other target is -100, which PyTorch ignores), so the
model is graded on one question only: given this request, which type is it?
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, default_data_collator

from app.data import load_train
from app.model import BASE_MODEL, build_model, build_tokenizer, label_ids

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"  # git-ignored: the fine-tuned weights are about 330 MB
DOCS_DIR = ROOT / "docs"


def encode_example(tok, example, max_len=64):
    """Token ids, attention mask and labels for one request. The label token is always kept, even for a long request."""
    label_id = label_ids(tok)[example["label"]]
    ids = tok(example["text"].strip())["input_ids"][:max_len - 1] + [label_id]
    pad = max_len - len(ids)
    labels = [-100] * (len(ids) - 1) + [label_id] + [-100] * pad  # only the label position counts
    return {"input_ids": ids + [tok.pad_token_id] * pad,
            "attention_mask": [1] * len(ids) + [0] * pad,
            "labels": labels}


class EncodedDataset(Dataset):
    def __init__(self, tok, rows, max_len=64):
        self.items = [{k: torch.tensor(v) for k, v in encode_example(tok, r, max_len).items()} for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def train(rows, tok=None, model=None, epochs=3, lr=5e-4, batch_size=8, seed=0, out_dir=None, log=print):
    """Fine-tune on rows. Returns (model, tokenizer, [(step, loss)]). If out_dir is given the model and tokenizer are saved."""
    torch.manual_seed(seed)
    tok = tok or build_tokenizer()
    model = model or build_model(tok)
    data = EncodedDataset(tok, rows)
    args = TrainingArguments(output_dir=str(out_dir or ROOT / "runs"), num_train_epochs=epochs,
                             per_device_train_batch_size=batch_size, learning_rate=lr, logging_steps=25,
                             report_to="none", save_strategy="no", disable_tqdm=True, seed=seed)
    trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=default_data_collator)
    log(f"{len(data)} training requests, {epochs} epochs, batch size {batch_size}, learning rate {lr}")
    start = time.time()
    trainer.train()
    history = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
    for step, loss in history:
        log(f"step {step:4d}: loss {loss:.4f}")
    log(f"trained in {time.time() - start:.0f} s")
    model.eval()
    if out_dir:
        trainer.save_model(str(out_dir))
        tok.save_pretrained(str(out_dir))
    return model, tok, history


def main():
    ap = argparse.ArgumentParser(description="Fine-tune GPT-2 to sort event requests")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=MODEL_DIR)
    args = ap.parse_args()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"base model: {BASE_MODEL}")
    train(load_train(), epochs=args.epochs, out_dir=args.out, log=log)
    log(f"saved the fine-tuned model and tokenizer to {args.out.name}/")
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "training_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
