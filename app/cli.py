"""Command-line entry point: `python -m app.cli --help`.

  python -m app.cli classify "Do you have a venue free on June 14 for 80 guests?"
  python -m app.cli classify --save requests.txt "Please cancel our booking and refund the deposit"
  python -m app.cli classify                      # interactive: type requests one by one
  python -m app.cli review requests.txt           # saved requests, most urgent first
"""
import argparse
from pathlib import Path

from app.data import CATEGORIES, PRIORITY_ORDER
from app.log import RequestLog, for_review, known_type_names, summary

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def describe(request, c, top=1):
    "The lines shown for one classified request."
    lines = [f"Request  : {request.strip()}", f"Type     : {c.name} ({c.confidence:.0%} sure)",
             f"Hand to  : {c.team} (priority: {c.priority})", f"Next step: {c.next_steps}"]
    if top > 1:
        ranked = sorted(c.probabilities.items(), key=lambda kv: -kv[1])[1:top]
        lines.append("Also     : " + ", ".join(f"{CATEGORIES[k]['name']} {p:.0%}" for k, p in ranked))
    return "\n".join(lines)


def classify_and_show(sorter, request, args):
    result = sorter.classify(request)
    print(describe(request, result, args.top))
    if args.save:
        RequestLog(args.save).add(request, result)
        print(f"Saved to {args.save}")


def review(args):
    entries = RequestLog(args.file).entries()
    shown = for_review(entries, args.priority, args.type)
    if not entries:
        print(f"No saved requests in {args.file}.")
        return
    for e in shown:
        print(f"[{e.priority.upper():6s}] {e.when:%Y-%m-%d %H:%M}  {e.type_name} -> {e.team} ({e.confidence})\n"
              f"         {e.request}")
    if not shown:
        print("Nothing matches that filter.")
    print("\n" + summary(entries))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="event-planning-request-sorter", description="Event planners use this tool to automatically sort incoming client requests (e.g., \"venue inquiry\", \"catering request\") for efficient task delegation and response prioritization.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    c = sub.add_parser("classify", help="sort a request into a type")
    c.add_argument("request", nargs="*", help="the client's message; leave empty for interactive mode")
    c.add_argument("--top", type=int, default=1, help="also show the next most likely types")
    c.add_argument("--save", metavar="FILE", help="append the classified request to a text file for later review")
    c.add_argument("--model-dir", default=MODEL_DIR, help="folder with the fine-tuned model")
    r = sub.add_parser("review", help="list saved requests, most urgent first")
    r.add_argument("file", help="the file written by --save")
    r.add_argument("--priority", choices=list(PRIORITY_ORDER), help="only this priority")
    r.add_argument("--type", choices=known_type_names(), help="only this request type")
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Event Planning Request Sorter is ready. Open README.md for your milestones.")
        return
    if args.command == "review":
        review(args)
        return
    from app.sorter import RequestSorter
    try:
        sorter = RequestSorter(args.model_dir)
    except FileNotFoundError as err:
        raise SystemExit(str(err))
    if args.request:
        classify_and_show(sorter, " ".join(args.request), args)
    else:
        print("Type a client request and press Enter (empty line to quit).")
        while (line := input("request> ").strip()):
            classify_and_show(sorter, line, args)
            print()


if __name__ == "__main__":
    main()
