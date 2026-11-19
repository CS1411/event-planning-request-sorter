"""Milestone 2: configure GPT-2 with an extended tokenizer and model for the request-type tokens.

Three steps: add one atomic token per request type (so each type is a single prediction, not a spelling
exercise), resize the model's embedding tables so each new token has a row, and give GPT-2 a pad token.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.data import CATEGORIES, label_token

BASE_MODEL = "distilgpt2"


def label_tokens():
    return [label_token(key) for key in CATEGORIES]


def build_tokenizer(base=BASE_MODEL):
    tok = AutoTokenizer.from_pretrained(base)
    tok.add_tokens(label_tokens())  # 1. each <Request Type> becomes one token
    tok.pad_token = tok.eos_token   # 3. GPT-2 ships without a pad token; reuse end-of-text
    return tok


def extend_model(model, tok):
    """2. Grow the input embeddings and the output head to the new vocabulary size.

    mean_resizing=False keeps the new rows as distinct small random vectors. transformers 5 would otherwise
    start every new row at the same mean vector, which makes all the request types look identical to the model.
    """
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    return model


def build_model(tok, base=BASE_MODEL):
    return extend_model(AutoModelForCausalLM.from_pretrained(base), tok)


def label_ids(tok):
    "{category key: token id of its label token}."
    return {key: tok.convert_tokens_to_ids(label_token(key)) for key in CATEGORIES}
