"""Milestone 5: save classified requests to a text file and read them back for review.

One request per line, easy to read and to grep:

    2026-11-19 18:40 | Cancellation or Refund | Finance | urgent | 97% | Please cancel our booking...
"""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.data import CATEGORIES, PRIORITY_ORDER

SEPARATOR = " | "
TIME_FORMAT = "%Y-%m-%d %H:%M"


@dataclass
class Entry:
    when: datetime
    type_name: str
    team: str
    priority: str
    confidence: str
    request: str


def clean(text):
    "One line only, and the separator cannot appear inside the request."
    return " ".join(text.split()).replace(SEPARATOR, " / ")


class RequestLog:
    def __init__(self, path):
        self.path = Path(path)

    def add(self, request, classification, when=None):
        "Append one classified request. Creates the file and its folder if needed."
        when = when or datetime.now()
        line = SEPARATOR.join([when.strftime(TIME_FORMAT), classification.name, classification.team,
                               classification.priority, f"{classification.confidence:.0%}", clean(request)])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return line

    def entries(self):
        "Every saved request, oldest first. Lines that are not in the expected format are skipped."
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            parts = line.split(SEPARATOR, 5)
            if len(parts) != 6:
                continue
            try:
                out.append(Entry(datetime.strptime(parts[0], TIME_FORMAT), *parts[1:]))
            except ValueError:
                continue
        return out


def for_review(entries, priority=None, category=None):
    "Most urgent first, then oldest first. Optionally keep only one priority or one request type name."
    kept = [e for e in entries
            if (priority is None or e.priority == priority)
            and (category is None or e.type_name.lower() == category.lower())]
    return sorted(kept, key=lambda e: (PRIORITY_ORDER.get(e.priority, 99), e.when))


def summary(entries):
    "Counts by priority and by request type, as text."
    if not entries:
        return "No saved requests."
    by_priority = Counter(e.priority for e in entries)
    by_type = Counter(e.type_name for e in entries)
    lines = [f"{len(entries)} saved request(s)", "By priority: " + ", ".join(
        f"{p} {by_priority[p]}" for p in PRIORITY_ORDER if p in by_priority), "By type:"]
    lines += [f"  {name}: {n}" for name, n in by_type.most_common()]
    return "\n".join(lines)


def known_type_names():
    return [info["name"] for info in CATEGORIES.values()]
