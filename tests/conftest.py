"""Shared fixtures: a tiny random GPT-2 saved like a real fine-tuned model, so no big download is needed."""
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from app.model import build_tokenizer, extend_model


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-model")
    torch.manual_seed(0)
    tok = build_tokenizer()
    model = extend_model(GPT2LMHeadModel(GPT2Config(vocab_size=50257, n_layer=2, n_head=2, n_embd=32,
                                                     n_positions=64)), tok)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path
