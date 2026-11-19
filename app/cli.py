"""Command-line entry point: `python -m app.cli --help`.

  python -m app.cli classify "Do you have a venue free on June 14 for 80 guests?"
  python -m app.cli classify                      # interactive: type requests one by one
"""
import argparse
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def describe(request, c, top=1):
    "The lines shown for one classified request."
    lines = [f"Request  : {request.strip()}", f"Type     : {c.name} ({c.confidence:.0%} sure)",
             f"Hand to  : {c.team} (priority: {c.priority})", f"Next step: {c.next_steps}"]
    if top > 1:
        from app.data import CATEGORIES
        ranked = sorted(c.probabilities.items(), key=lambda kv: -kv[1])[1:top]
        lines.append("Also     : " + ", ".join(f"{CATEGORIES[k]['name']} {p:.0%}" for k, p in ranked))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="event-planning-request-sorter", description="Event planners use this tool to automatically sort incoming client requests (e.g., \"venue inquiry\", \"catering request\") for efficient task delegation and response prioritization.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    c = sub.add_parser("classify", help="sort a request into a type")
    c.add_argument("request", nargs="*", help="the client's message; leave empty for interactive mode")
    c.add_argument("--top", type=int, default=1, help="also show the next most likely types")
    c.add_argument("--model-dir", default=MODEL_DIR, help="folder with the fine-tuned model")
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Event Planning Request Sorter is ready. Open README.md for your milestones.")
        return
    from app.sorter import RequestSorter
    try:
        sorter = RequestSorter(args.model_dir)
    except FileNotFoundError as err:
        raise SystemExit(str(err))
    if args.request:
        print(describe(" ".join(args.request), sorter.classify(" ".join(args.request)), args.top))
    else:
        print("Type a client request and press Enter (empty line to quit).")
        while (line := input("request> ").strip()):
            print(describe(line, sorter.classify(line), args.top) + "\n")


if __name__ == "__main__":
    main()
