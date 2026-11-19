# Event Planning Request Sorter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/y846vfkg73/badge.svg)](https://ask.thili.ai/b/y846vfkg73)
[![tests](https://github.com/CS1411/event-planning-request-sorter/actions/workflows/ci.yml/badge.svg)](https://github.com/CS1411/event-planning-request-sorter/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

A command-line tool for event planners that reads an incoming client message, sorts it into one of eight request types with a **fine-tuned GPT-2**, says which team should handle it and how urgent it usually is, and saves it to a file for later review.

```console
$ python -m app.cli classify "Please cancel our booking and refund the deposit, the event is off." --save requests.txt --top 3
Request  : Please cancel our booking and refund the deposit, the event is off.
Type     : Cancellation or Refund (100% sure)
Hand to  : Finance (priority: urgent)
Next step: Review the contract's cancellation terms and reply the same day.
Also     : Venue Inquiry 0%, Date or Schedule Change 0%
Saved to requests.txt
```

It uses the technique from the *Fine-Tune GPT-2 to Tag Legal Contract Clauses* drill: **classification as generation**. Each request is followed by a special label token (for example `<Venue Inquiry>`), GPT-2's tokenizer and embeddings are extended to match, and the model is fine-tuned to produce the label token.

## Contents

- [Quick start](#quick-start)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Milestones](#milestones)
- [Results](#results)
- [Data](#data)
- [Reproduce it](#reproduce-it)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Limitations](#limitations)
- [Honest development notes](#honest-development-notes)
- [License](#license)

## Quick start

Requires Python 3.11 or newer and git. Training downloads distilgpt2 (about 350 MB) from the Hugging Face Hub on first use; no keys are needed. The fine-tuned weights are about 330 MB, so they are **not** in the repository: train once (about 5 minutes on a laptop CPU).

```bash
git clone https://github.com/CS1411/event-planning-request-sorter.git
cd event-planning-request-sorter
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.train            # fine-tune and save to model/
python -m app.cli classify "Do you have a venue free on June 14 for about 80 guests?"
```

## Usage

```bash
python -m app.cli hello                                    # check that everything is wired up
python -m app.cli classify "Can the caterer do vegan options for 60 guests?"
python -m app.cli classify                                 # interactive: type requests one by one
python -m app.cli classify "..." --save requests.txt       # also append it to a file for review
python -m app.cli classify "..." --top 3                   # also show the runner-up types
python -m app.cli review requests.txt                      # saved requests, most urgent first
python -m app.cli review requests.txt --priority urgent    # filter by priority or by --type
```

`review` shows the saved requests most urgent first (oldest first within a priority) followed by counts:

```console
$ python -m app.cli review requests.txt
[URGENT] 2026-11-19 18:44  Cancellation or Refund -> Finance (100%)
         Please cancel our booking and refund the deposit, the event is off.
[HIGH  ] 2026-11-19 18:44  Date or Schedule Change -> Operations (100%)
         Could we move the conference to the following Friday?
[NORMAL] 2026-11-19 18:43  Venue Inquiry -> Venue team (100%)
         Do you have a venue free on June 14 for about 80 guests?
[LOW   ] 2026-11-19 18:44  General Question -> Front desk (100%)
         What are your opening hours?

4 saved request(s)
By priority: urgent 1, high 1, normal 1, low 1
```

The saved file is plain text, one request per line: `time | type | team | priority | confidence | request`.

| Request type | Hand to | Priority |
|---|---|---|
| Venue Inquiry | Venue team | normal |
| Catering Request | Catering team | normal |
| Quote and Budget | Sales team | high |
| Guest Management | Guest services | normal |
| Decor and Entertainment | Creative team | normal |
| Date or Schedule Change | Operations | high |
| Cancellation or Refund | Finance | urgent |
| General Question | Front desk | low |

The teams, priorities and suggested next steps are a table in `app/data.py`; edit it to match how your own business works.

## How it works

1. **Training strings.** Every example becomes `request text` + `<Request Type>`. At test time the model is given only the request and its very next token is read.
2. **Extended tokenizer and model.** `<Venue Inquiry>` would be spelled out by GPT-2 as several pieces, so the eight types are added as single tokens, the model's input embeddings and output head grow from 50,257 to 50,265 rows, and GPT-2 gets a pad token (it ships without one).
3. **Loss on the label only.** Every training target is set to -100 (ignored by PyTorch's loss) except the single label-token position, so the model is graded on one question: which type is this request? 3 epochs, batch size 8, learning rate 5e-4, the Hugging Face `Trainer`.
4. **Constrained prediction.** The sorter compares only the eight label-token scores, so the answer is always a valid type, and a softmax over those eight gives the confidence shown.

## Milestones

Each box is ticked because the milestone was run and checked.

- [x] **1. Define the request types and create a labelled dataset.** `app/data.py`; 8 types and 704 training, 192 test and 48 hand-written challenge requests in `dataset/`.
- [x] **2. Configure GPT-2 with an extended tokenizer and model.** `app/model.py`; tests check each type is a single token, the embeddings grow, and the new rows are distinct.
- [x] **3. Train on the example requests with the Hugging Face Trainer.** `app/train.py`; the training log is in `docs/training_log.txt`.
- [x] **4. A command-line interface that classifies new requests.** `app/sorter.py` and `app/cli.py`; evaluated in `docs/results.md`.
- [x] **5. Save classified requests to a file for review.** `app/log.py`; `classify --save` and `review`.

Stretch ideas covered: a suggested next step for each type, and routing to a team with a priority. Not built: a web interface.

## Results

Measured by `python -m app.evaluate` on requests the model was not trained on (full detail, per-type scores and every miss in [`docs/results.md`](docs/results.md)):

| set | requests | fine-tuned GPT-2 | word-count baseline | always one type |
|---|---|---|---|---|
| test (sentence patterns never seen in training) | 192 | **78.6%** | 37.0% | 12.5% |
| challenge (48 hand-written messages in a looser style) | 48 | **70.8%** | 52.1% | 12.5% |

How to read this honestly:

- The model clearly beats both baselines, so it learned something real about what the types mean, not just keywords.
- It is far from perfect on new phrasings, and it **memorised its training patterns**: training loss reached about 0 after roughly 200 steps. The test set holds out 3 whole sentence patterns per type, so a pattern is either understood or missed, which is why per-type scores jump in steps of 8 (for example Venue Inquiry 16/24).
- It is **overconfident**: its average confidence is 97% to 99% when right and about 90% when wrong, so the percentage is not a trustworthy probability. Treat low confidence as a warning, and high confidence as nothing more than "the model's favourite".
- The challenge set is only 48 messages written by the same author as the training data, so it is small and not independent. One request is worth about 2 points.

## Data

The data is **synthetic**; no real client messages were used. Each type has 14 hand-written sentence patterns, filled in with random events, cities, dates, guest counts and amounts and wrapped in greetings and sign-offs, with some lower-casing and dropped punctuation. Patterns 3, 7 and 11 of every type are used only for testing. `python -m app.data` regenerates the files. The 48 challenge requests in `dataset/challenge.jsonl` are written by hand.

## Reproduce it

```bash
python -m app.data       # rebuild dataset/train.jsonl and dataset/test.jsonl (optional, they are committed)
python -m app.train      # fine-tune, writes model/ and docs/training_log.txt (about 5 minutes on CPU)
python -m app.evaluate   # writes docs/results.md
```

Training is seeded, and library versions are pinned in `requirements.txt`.

## Project structure

```text
app/
  data.py       request types, teams and priorities, sentence patterns, dataset
  model.py      extended tokenizer, resized model
  train.py      loss-on-the-label-only encoding and the Trainer
  sorter.py     RequestSorter: load the model and classify
  evaluate.py   accuracy on the test and challenge sets, with a word-count baseline
  log.py        save classified requests to a text file and review them
  cli.py        command-line interface
dataset/        training, test and challenge requests
docs/           training log and evaluation results
tests/          one test file per milestone
```

## Testing

```bash
pytest
```

The tests need no fine-tuned model: they use a tiny random GPT-2 built on the spot, plus the real (small) GPT-2 tokenizer, which is downloaded on first use. They cover the dataset (balanced, reproducible, no duplicates, test patterns disjoint from training), the tokenizer and embedding extension (including a check that the new label rows are distinct, which catches a bug in newer library versions), the label-only loss mask, a short training run that really learns, classification, the file log and the command line. They run on GitHub for every push.

## Limitations

- **Synthetic, templated data.** The model learned a few dozen sentence shapes per type. Real client messages are messier, longer and often cover more than one thing at once; expect it to miss more than the numbers above suggest.
- **One label per request.** "Can we move the date and get a quote for more guests?" is two things; the tool picks one.
- **Eight types only**, and no "unknown" option: a request unrelated to events still gets one of the eight, often with high confidence.
- **Overconfident** (see Results). Do not use the confidence to decide automatically.
- **English only**, and distilgpt2 is a small model; a larger GPT-2 and far more varied data would help.
- **The routing table is an example.** Teams, priorities and next steps are placeholders to edit.
- **Weights are not in the repository** (about 330 MB), so a fresh clone must run `python -m app.train` first.

## Honest development notes

This project was built with AI assistance (Claude) in a working session with the repository owner, who chose the topic and directed the work. It is not hand-written code. The commits were made after the code was finished, so their dates show when the files were committed, not how long each milestone took. The numbers in this README were produced by running this code, not copied from the course page.

## License

MIT. See [LICENSE](LICENSE).
