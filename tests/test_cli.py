"""Milestone 4: the command-line interface for classifying requests."""
import pytest

from app.cli import describe, main
from app.sorter import RequestSorter


def test_hello_needs_no_model(capsys):
    main(["hello"])
    assert "is ready" in capsys.readouterr().out


def test_classify_prints_type_team_priority_and_next_step(tiny_model_dir, capsys):
    main(["classify", "Do you have a venue free on June 14?", "--model-dir", str(tiny_model_dir)])
    out = capsys.readouterr().out
    assert out.startswith("Request  : Do you have a venue free on June 14?")
    for label in ("Type     :", "Hand to  :", "Next step:"):
        assert label in out


def test_top_shows_the_runner_ups(tiny_model_dir, capsys):
    main(["classify", "hello there", "--top", "3", "--model-dir", str(tiny_model_dir)])
    assert capsys.readouterr().out.count("%") >= 3  # the confidence and two more types


def test_interactive_mode_answers_each_request_until_a_blank_line(tiny_model_dir, monkeypatch, capsys):
    answers = iter(["Please cancel the booking", "What are your opening hours?", ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    main(["classify", "--model-dir", str(tiny_model_dir)])
    out = capsys.readouterr().out
    assert "Please cancel the booking" in out and "What are your opening hours?" in out


def test_describe_formats_one_classified_request(tiny_model_dir):
    c = RequestSorter(tiny_model_dir).classify("Can you cater lunch for 60?")
    text = describe("Can you cater lunch for 60?", c)
    assert text.splitlines()[0] == "Request  : Can you cater lunch for 60?" and c.name in text and c.team in text


def test_a_missing_model_stops_with_the_training_command(tmp_path):
    with pytest.raises(SystemExit) as err:
        main(["classify", "hi", "--model-dir", str(tmp_path)])
    assert "python -m app.train" in str(err.value)
