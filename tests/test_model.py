"""Milestone 2: the extended tokenizer and model. Uses the real GPT-2 tokenizer (a small download) and a tiny random model."""
import pytest
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from app.data import CATEGORIES, label_token
from app.model import build_tokenizer, extend_model, label_ids, label_tokens


@pytest.fixture(scope="module")
def tok():
    return build_tokenizer()


def tiny_gpt2():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=50257, n_layer=2, n_head=2, n_embd=32, n_positions=64))


def test_each_request_type_becomes_a_single_token(tok):
    plain = AutoTokenizer.from_pretrained("distilgpt2")
    for token in label_tokens():
        assert len(plain.tokenize(token)) > 1   # GPT-2 alone shatters it into pieces
        assert tok.tokenize(token) == [token]   # after add_tokens it is one atomic token
    ids = label_ids(tok)
    assert set(ids) == set(CATEGORIES) and len(set(ids.values())) == 8 and min(ids.values()) >= 50257


def test_text_with_a_label_ends_in_that_label_token(tok):
    enc = tok("Do you have a venue free on June 14? " + label_token("venue"))
    assert enc["input_ids"][-1] == label_ids(tok)["venue"]


def test_gpt2_gets_a_pad_token_and_padding_works(tok):
    assert tok.pad_token == tok.eos_token and tok.pad_token_id == 50256
    enc = tok(["a short one", "a much longer request about catering for two hundred guests"], padding=True,
              return_tensors="pt")
    assert enc["attention_mask"][0].tolist().count(0) > 0 and enc["attention_mask"][1].tolist().count(0) == 0


def test_resizing_adds_one_embedding_row_per_token_and_keeps_old_rows(tok):
    model = tiny_gpt2()
    old = model.get_input_embeddings().weight.detach().clone()
    extend_model(model, tok)
    new = model.get_input_embeddings().weight
    assert new.shape[0] == len(tok) == 50257 + len(CATEGORIES)
    assert torch.equal(new[:50257], old)                      # pretrained rows untouched
    assert model.lm_head.weight.shape[0] == len(tok)          # the output head grew too, so it can emit the labels


def test_new_label_rows_are_distinct_not_copies_of_one_mean_vector(tok):
    model = extend_model(tiny_gpt2(), tok)
    rows = model.get_input_embeddings().weight[50257:].detach()
    cos = torch.nn.functional.cosine_similarity(rows[:, None, :], rows[None, :, :], dim=-1)
    off_diagonal = cos[~torch.eye(len(rows), dtype=torch.bool)]
    assert off_diagonal.abs().max() < 0.8  # random rows, nearly orthogonal; the transformers-5 default would give 1.0
