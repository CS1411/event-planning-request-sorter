# Event Planning Request Sorter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/y846vfkg73/badge.svg)](https://ask.thili.ai/b/y846vfkg73)

This project creates an "Event Planning Request Sorter" by fine-tuning a GPT-2 model. It helps event planners by automatically categorizing client requests like "venue inquiry" or "catering request". The tool uses the fine-tuning logic from the "Fine-Tune GPT-2 to Tag Legal Contract Clauses" drill to streamline task delegation and response prioritization.

## What you are building

**Goal.** You will build an automated tool that classifies incoming client requests for event planning, such as "venue inquiry" or "catering request", to help event planners efficiently delegate tasks and prioritize responses.

**Done when.** The project is done when you have a command-line interface (CLI) that can take a new event request as input, classify it into predefined types using a fine-tuned GPT-2 model, and display the classification. Additionally, the tool should be able to save these classified requests to a file for later review.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Define common event request types and create a small dataset of example requests.**  
      Hint: Data preparation: Define categories and create a small, labeled dataset.
- [ ] **2. Configure GPT-2 with an extended tokenizer and model for custom tokens.**  
      Hint: Tokenizer extension: Add special tokens for your request categories to the GPT-2 tokenizer.
- [ ] **3. Train the model on the example requests using the Hugging Face Trainer.**  
      Hint: Model fine-tuning: Use the Hugging Face Trainer to train GPT-2 on your dataset.
- [ ] **4. Build an interface (CLI) to input new requests and show their classification.**  
      Hint: Inference pipeline: Load your fine-tuned model and tokenizer to classify new requests via a CLI.
- [ ] **5. Allow saving classified requests to a file for review.**  
      Hint: File I/O: Implement saving the classified request and its label to a text file.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Expand the dataset with more diverse examples and request types to improve classification accuracy and coverage.
- Implement a feature to suggest common next steps or template responses based on the classified request type.
- Develop a simple web-based interface instead of a CLI for easier access and use by event planners.
