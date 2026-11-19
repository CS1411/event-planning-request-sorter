"""Evaluate the fine-tuned sorter honestly: `python -m app.evaluate`. Writes docs/results.md.

Two held-out sets are used. The test set has sentence patterns never seen in training (but written by the same
author as the training patterns). The challenge set is 48 hand-written requests in a looser, messier style.
Two baselines are shown for context: always guessing one type, and a small word-count (naive Bayes) classifier.
"""
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from app.data import CATEGORIES, load_challenge, load_test, load_train

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def words(text):
    return re.findall(r"[a-z']+", text.lower())


class NaiveBayes:
    "A bag-of-words baseline: which type's training words does this request look most like?"

    def fit(self, rows):
        self.word_counts = defaultdict(Counter)
        self.type_counts = Counter(r["label"] for r in rows)
        for r in rows:
            self.word_counts[r["label"]].update(words(r["text"]))
        self.vocab = {w for c in self.word_counts.values() for w in c}
        return self

    def predict(self, text):
        total = sum(self.type_counts.values())
        best, best_score = None, -math.inf
        for key in CATEGORIES:
            n = sum(self.word_counts[key].values())
            score = math.log((self.type_counts[key] + 1) / (total + len(CATEGORIES)))  # add-one, so an unseen type is not impossible
            for w in words(text):
                score += math.log((self.word_counts[key][w] + 1) / (n + len(self.vocab) + 1))
            if score > best_score:
                best, best_score = key, score
        return best


def accuracy(predictions, rows):
    return sum(p == r["label"] for p, r in zip(predictions, rows)) / len(rows)


def per_type(predictions, rows):
    return {k: (sum(1 for p, r in zip(predictions, rows) if r["label"] == k and p == k),
                sum(1 for r in rows if r["label"] == k)) for k in CATEGORIES}


def confusions(predictions, rows):
    return Counter((r["label"], p) for p, r in zip(predictions, rows) if p != r["label"])


def build_report(results):
    "results: {set name: {'rows': [...], 'gpt2': [...], 'bayes': [...], 'confidence': [...]}} -> markdown."
    majority = 1 / len(CATEGORIES)
    lines = ["# Evaluation", "",
             "Accuracy of the fine-tuned model on requests it was not trained on, next to two simple baselines.", "",
             "| set | requests | fine-tuned GPT-2 | word-count baseline | always one type |", "|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {len(r['rows'])} | **{accuracy(r['gpt2'], r['rows']):.1%}** | "
                     f"{accuracy(r['bayes'], r['rows']):.1%} | {majority:.1%} |")
    lines += ["", "- **test**: sentence patterns that never appear in training (same author, same style as the training data).",
              "- **challenge**: 48 hand-written requests in a looser style, 6 per type. Small, so one request is about 2 points."]
    for name, r in results.items():
        correct = [c for c, p, x in zip(r["confidence"], r["gpt2"], r["rows"]) if p == x["label"]]
        wrong = [c for c, p, x in zip(r["confidence"], r["gpt2"], r["rows"]) if p != x["label"]]
        pt = per_type(r["gpt2"], r["rows"])
        lines += ["", f"## {name}", "", "| request type | correct |", "|---|---|"]
        lines += [f"| {CATEGORIES[k]['name']} | {a} / {n} |" for k, (a, n) in pt.items()]
        mean = lambda xs: f"{sum(xs) / len(xs):.0%}" if xs else "n/a"
        lines += ["", f"Mean confidence: {mean(correct)} when right, {mean(wrong)} when wrong."]
        conf = confusions(r["gpt2"], r["rows"]).most_common(5)
        if conf:
            lines += ["", "Most common mix-ups (true type -> predicted):", ""]
            lines += [f"- {CATEGORIES[t]['name']} -> {CATEGORIES[p]['name']}: {n}" for (t, p), n in conf]
        misses = [(x, p) for p, x in zip(r["gpt2"], r["rows"]) if p != x["label"]][:12]
        if misses:
            lines += ["", "Misses:", ""]
            lines += [f"- \"{x['text']}\"  \n  true: {CATEGORIES[x['label']]['name']}, predicted: {CATEGORIES[p]['name']}"
                      for x, p in misses]
    return "\n".join(lines) + "\n"


def main():
    from app.sorter import RequestSorter
    sorter, bayes = RequestSorter(), NaiveBayes().fit(load_train())
    results = {}
    for name, rows in (("test", load_test()), ("challenge", load_challenge())):
        cls = [sorter.classify(r["text"]) for r in rows]
        results[name] = {"rows": rows, "gpt2": [c.key for c in cls], "confidence": [c.confidence for c in cls],
                         "bayes": [bayes.predict(r["text"]) for r in rows]}
        print(f"{name}: fine-tuned GPT-2 {accuracy(results[name]['gpt2'], rows):.1%}, "
              f"word-count baseline {accuracy(results[name]['bayes'], rows):.1%} ({len(rows)} requests)")
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "results.md").write_text(build_report(results), encoding="utf-8")
    print(f"wrote {DOCS_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
