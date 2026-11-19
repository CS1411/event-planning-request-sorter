"""Milestone 4: load the fine-tuned model and tokenizer and classify new requests.

A request is fed to the model exactly as in training (the text, with nothing after it) and the model's very next
token is read. Only the eight request-type tokens are compared, so the answer is always a valid type, and a softmax
over those eight scores gives a confidence.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.data import CATEGORIES
from app.model import label_ids

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MAX_TOKENS = 63  # training used 64 positions: up to 63 request tokens, then the label


@dataclass
class Classification:
    key: str            # category key, for example "venue"
    name: str           # display name, for example "Venue Inquiry"
    team: str           # who should handle it
    priority: str       # urgent, high, normal or low
    next_steps: str     # a suggested first action
    confidence: float   # softmax probability of the chosen type among the eight
    probabilities: dict  # {category key: probability}


class RequestSorter:
    def __init__(self, path=MODEL_DIR, tok=None, model=None):
        if tok is None or model is None:
            if not (Path(path) / "config.json").exists():
                raise FileNotFoundError(f"No fine-tuned model in {path}. Train one first: python -m app.train")
            tok, model = AutoTokenizer.from_pretrained(path), AutoModelForCausalLM.from_pretrained(path)
        self.tok, self.model = tok, model.eval()  # eval() switches dropout off, so results are repeatable
        self.ids = label_ids(tok)

    @torch.no_grad()
    def scores(self, request):
        "The model's raw score for each request type as the next token after the request."
        enc = self.tok(request.strip(), return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        logits = self.model(**enc).logits[0, -1]
        return {key: logits[token_id].item() for key, token_id in self.ids.items()}

    def classify(self, request):
        if not request or not request.strip():
            raise ValueError("The request is empty.")
        scores = self.scores(request)
        keys = list(scores)
        probs = F.softmax(torch.tensor([scores[k] for k in keys]), dim=0).tolist()
        probabilities = dict(zip(keys, probs))
        best = max(probabilities, key=probabilities.get)
        info = CATEGORIES[best]
        return Classification(best, info["name"], info["team"], info["priority"], info["next_steps"],
                              probabilities[best], probabilities)
