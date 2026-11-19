"""Milestone 4: classifying new requests, and evaluating the classifier."""
import pytest

from app.data import CATEGORIES
from app.evaluate import NaiveBayes, accuracy, build_report, confusions, per_type
from app.sorter import Classification, RequestSorter


def test_a_request_gets_a_valid_type_team_priority_and_confidence(tiny_model_dir):
    sorter = RequestSorter(tiny_model_dir)
    c = sorter.classify("Do you have a venue free on June 14 for 80 guests?")
    assert isinstance(c, Classification) and c.key in CATEGORIES
    assert (c.name, c.team, c.priority) == (CATEGORIES[c.key]["name"], CATEGORIES[c.key]["team"],
                                            CATEGORIES[c.key]["priority"])
    assert set(c.probabilities) == set(CATEGORIES) and abs(sum(c.probabilities.values()) - 1) < 1e-5
    assert c.confidence == max(c.probabilities.values()) and 1 / 8 <= c.confidence <= 1


def test_the_answer_is_repeatable_and_only_chooses_among_the_eight_types(tiny_model_dir):
    sorter = RequestSorter(tiny_model_dir)
    text = "Please cancel our booking and refund the deposit."
    assert sorter.scores(text) == sorter.scores(text) and len(sorter.scores(text)) == 8
    assert sorter.classify(text).key == sorter.classify(text).key


def test_empty_and_very_long_requests(tiny_model_dir):
    sorter = RequestSorter(tiny_model_dir)
    for bad in ("", "   ", None):
        with pytest.raises(ValueError):
            sorter.classify(bad)
    assert sorter.classify("catering " * 500).key in CATEGORIES  # cut to the context window, not an error


def test_a_missing_model_gives_a_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m app.train"):
        RequestSorter(tmp_path / "nowhere")


TRAIN = [{"text": "send a venue quote", "label": "venue"}, {"text": "venue tour please", "label": "venue"},
         {"text": "refund my deposit", "label": "cancel"}, {"text": "cancel and refund", "label": "cancel"}]


def test_the_word_count_baseline_learns_the_training_words():
    nb = NaiveBayes().fit(TRAIN)
    assert nb.predict("can I get a refund") == "cancel" and nb.predict("venue tour") == "venue"


def test_accuracy_per_type_and_confusions():
    rows = [{"label": "venue"}, {"label": "venue"}, {"label": "cancel"}, {"label": "general"}]
    preds = ["venue", "cancel", "cancel", "general"]
    assert accuracy(preds, rows) == 0.75
    pt = per_type(preds, rows)
    assert pt["venue"] == (1, 2) and pt["cancel"] == (1, 1) and pt["catering"] == (0, 0)
    assert confusions(preds, rows) == {("venue", "cancel"): 1}


def test_report_is_markdown_with_both_sets_and_the_misses():
    rows = [{"text": "a venue question", "label": "venue"}, {"text": "cancel it", "label": "cancel"}]
    results = {"test": {"rows": rows, "gpt2": ["venue", "venue"], "bayes": ["venue", "cancel"],
                        "confidence": [0.9, 0.6]}}
    report = build_report(results)
    assert "| test | 2 | **50.0%** | 100.0% | 12.5% |" in report
    assert "cancel it" in report and "Cancellation or Refund -> Venue Inquiry: 1" in report
