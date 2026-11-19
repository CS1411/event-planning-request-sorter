"""Milestone 5: saving classified requests to a file and reviewing them."""
from datetime import datetime

from app.cli import main
from app.log import RequestLog, for_review, summary
from app.sorter import RequestSorter

T1, T2, T3 = datetime(2026, 11, 19, 9, 0), datetime(2026, 11, 19, 10, 30), datetime(2026, 11, 20, 8, 15)


def classified(tiny_model_dir, text):
    return RequestSorter(tiny_model_dir).classify(text)


def test_requests_are_appended_one_per_line_and_read_back(tiny_model_dir, tmp_path):
    log = RequestLog(tmp_path / "deep" / "folder" / "requests.txt")  # missing folders are created
    assert log.entries() == []
    c = classified(tiny_model_dir, "Please cancel our booking")
    line = log.add("Please cancel our booking", c, when=T1)
    log.add("Second request", c, when=T2)
    assert line.startswith("2026-11-19 09:00 | ") and line.endswith("| Please cancel our booking")
    got = log.entries()
    assert [e.request for e in got] == ["Please cancel our booking", "Second request"] and got[0].when == T1
    assert got[0].type_name == c.name and got[0].team == c.team and got[0].priority == c.priority


def test_awkward_text_cannot_break_the_file_format(tiny_model_dir, tmp_path):
    log = RequestLog(tmp_path / "r.txt")
    c = classified(tiny_model_dir, "x")
    log.add("Line one\nLine two | with a separator\t and tabs", c, when=T1)
    (tmp_path / "r.txt").write_text((tmp_path / "r.txt").read_text() + "garbage line\n", encoding="utf-8")
    entries = log.entries()
    assert len(entries) == 1 and entries[0].request == "Line one Line two / with a separator and tabs"


def test_review_lists_the_most_urgent_first_and_can_filter(tiny_model_dir, tmp_path):
    log = RequestLog(tmp_path / "r.txt")
    c = classified(tiny_model_dir, "x")
    for when, priority, name in ((T1, "low", "General Question"), (T2, "urgent", "Cancellation or Refund"),
                                 (T3, "urgent", "Cancellation or Refund"), (T1, "normal", "Venue Inquiry")):
        c.priority, c.name = priority, name
        log.add(f"{name} {when:%H%M}", c, when=when)
    entries = log.entries()
    ordered = for_review(entries)
    assert [e.priority for e in ordered] == ["urgent", "urgent", "normal", "low"]
    assert ordered[0].when < ordered[1].when                              # oldest first within a priority
    assert len(for_review(entries, priority="urgent")) == 2
    assert [e.type_name for e in for_review(entries, category="venue inquiry")] == ["Venue Inquiry"]
    text = summary(entries)
    assert "4 saved request(s)" in text and "urgent 2" in text and "Cancellation or Refund: 2" in text
    assert summary([]) == "No saved requests."


def test_cli_save_then_review(tiny_model_dir, tmp_path, capsys):
    out_file = tmp_path / "requests.txt"
    for text in ("Please cancel the booking", "What are your opening hours?"):
        main(["classify", text, "--save", str(out_file), "--model-dir", str(tiny_model_dir)])
    assert "Saved to" in capsys.readouterr().out
    assert len(out_file.read_text().strip().splitlines()) == 2
    main(["review", str(out_file)])
    shown = capsys.readouterr().out
    assert "Please cancel the booking" in shown and "What are your opening hours?" in shown
    assert "2 saved request(s)" in shown
    main(["review", str(tmp_path / "nothing.txt")])
    assert "No saved requests" in capsys.readouterr().out
