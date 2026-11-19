"""Milestone 1: the request types and the labelled dataset."""
from collections import Counter

from app.data import (CATEGORIES, PRIORITY_ORDER, TEMPLATES, TEST_TEMPLATE_IDS, generate_requests, label_token,
                      load_challenge, load_test, load_train)


def test_every_category_has_a_team_priority_and_label_token():
    assert len(CATEGORIES) == 8 and len(TEMPLATES) == 8
    for key, info in CATEGORIES.items():
        assert info["team"] and info["next_steps"] and info["priority"] in PRIORITY_ORDER
        assert label_token(key) == f"<{info['name']}>"
    assert len({label_token(k) for k in CATEGORIES}) == 8  # all different, so each can be one token


def test_generation_is_balanced_reproducible_and_distinct():
    train, test = generate_requests(per_template=3, seed=5)
    assert (train, test) == generate_requests(per_template=3, seed=5)
    assert generate_requests(per_template=3, seed=6) != (train, test)
    n_train = 14 - len(TEST_TEMPLATE_IDS)
    assert Counter(e["label"] for e in train) == {k: n_train * 3 for k in CATEGORIES}
    assert Counter(e["label"] for e in test) == {k: len(TEST_TEMPLATE_IDS) * 3 for k in CATEGORIES}
    texts = [e["text"] for e in train + test]
    assert len(texts) == len(set(texts))  # no duplicates, so nothing can leak between train and test


def test_test_set_uses_sentence_patterns_never_seen_in_training():
    train, test = generate_requests(per_template=4, seed=1)
    assert {e["template"] for e in train}.isdisjoint({e["template"] for e in test})
    assert {e["template"] for e in test} == set(TEST_TEMPLATE_IDS)


def test_requests_look_like_real_messages():
    train, _ = generate_requests(per_template=5, seed=2)
    assert all("{" not in e["text"] and len(e["text"]) > 15 for e in train)  # every placeholder was filled
    assert len({e["text"].split()[0].lower() for e in train}) > 5            # varied greetings and openings


def test_committed_dataset_files():
    train, test, challenge = load_train(), load_test(), load_challenge()
    assert len(train) >= 600 and len(test) >= 150
    assert {e["label"] for e in train} == set(CATEGORIES) == {e["label"] for e in challenge}
    assert Counter(e["label"] for e in challenge) == {k: 6 for k in CATEGORIES}   # 48 hand-written requests
    assert not {e["text"] for e in train} & {e["text"] for e in challenge}
