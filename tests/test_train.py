"""Milestone 3: training with the Trainer. A tiny random GPT-2 on a handful of requests, so it runs in seconds."""
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from app.data import generate_requests
from app.model import build_tokenizer, extend_model, label_ids
from app.train import EncodedDataset, encode_example, train


@pytest.fixture(scope="module")
def tok():
    return build_tokenizer()


def tiny_model(tok):
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=50257, n_layer=2, n_head=2, n_embd=64, n_positions=64))
    return extend_model(model, tok)


def test_only_the_label_token_counts_in_the_loss(tok):
    ex = {"text": "Do you have a venue free on June 14?", "label": "venue"}
    enc = encode_example(tok, ex, max_len=32)
    ids, labels, mask = enc["input_ids"], enc["labels"], enc["attention_mask"]
    assert len(ids) == len(labels) == len(mask) == 32
    target = label_ids(tok)["venue"]
    positions = [i for i, lab in enumerate(labels) if lab != -100]
    assert positions == [sum(mask) - 1] and labels[positions[0]] == ids[positions[0]] == target
    assert all(m == 0 for m in mask[sum(mask):]) and ids[-1] == tok.pad_token_id  # padding is masked out


def test_a_very_long_request_is_cut_but_keeps_its_label(tok):
    ex = {"text": "venue " * 200, "label": "cancel"}
    enc = encode_example(tok, ex, max_len=24)
    assert enc["input_ids"][23] == label_ids(tok)["cancel"] and enc["labels"][23] == enc["input_ids"][23]
    assert sum(1 for x in enc["labels"] if x != -100) == 1


def test_dataset_items_are_tensors_ready_for_the_trainer(tok):
    rows, _ = generate_requests(per_template=1, seed=0)
    data = EncodedDataset(tok, rows[:6], max_len=40)
    assert len(data) == 6 and all(v.shape == (40,) for v in data[0].values())


def test_training_lowers_the_loss_and_teaches_the_label_tokens(tok, tmp_path):
    rows = [{"text": "Can we tour the venue and see the ballroom floor plan?", "label": "venue"},
            {"text": "Please cancel the booking and refund our deposit.", "label": "cancel"},
            {"text": "We need a vegetarian buffet menu for the guests.", "label": "catering"}] * 8
    model, tok, history = train(rows, tok=tok, model=tiny_model(tok), epochs=30, lr=3e-3, batch_size=8,
                                out_dir=tmp_path / "m", log=lambda *_: None)
    assert history[-1][1] < history[0][1] / 3
    ids = label_ids(tok)
    with torch.no_grad():
        for r in rows[:3]:
            logits = model(**tok(r["text"].strip(), return_tensors="pt")).logits[0, -1]  # same shape as training: text, then label
            assert max(ids, key=lambda k: logits[ids[k]].item()) == r["label"]
    assert (tmp_path / "m" / "config.json").exists() and (tmp_path / "m" / "tokenizer.json").exists()
