"""The few-shot examples in data/examples.toml must be valid model answers."""

import tomllib
from pathlib import Path

import pytest

from mandarin_core.checker import check
from mandarin_core.evaluation import load_cases
from mandarin_core.generate import PassageRequest, judge, load_examples
from mandarin_core.vocab import VocabProfile

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "data" / "examples.toml"
LEVELS = {"hsk1": 1, "hsk1-2": 2, "hsk1-3": 3}
RAW = tomllib.loads(EXAMPLES.read_text(encoding="utf-8"))["examples"]


@pytest.mark.parametrize("example", RAW, ids=lambda e: f"{e['profile']}:{e['topic'][:6]}")
def test_example_passes_with_no_unknown_words(example):
    profile = VocabProfile.from_hsk(LEVELS[example["profile"]])
    request = PassageRequest(
        profile, example["topic"], tuple(example["targets"]), example["min_coverage"]
    )
    report = check(example["text"], profile, request.targets, request.names)
    assert judge(report, request).passed
    assert report.unknown_words == []


def test_examples_do_not_overlap_eval_topics_or_targets():
    cases = load_cases(REPO / "eval" / "cases.toml", root=REPO)
    eval_topics = {c.request.topic for c in cases}
    eval_targets = {t for c in cases for t in c.request.targets}
    for example in RAW:
        assert example["topic"] not in eval_topics
        assert not set(example["targets"]) & eval_targets


def test_cases_get_examples_for_their_profile():
    cases = load_cases(REPO / "eval" / "cases.toml", root=REPO, examples=EXAMPLES)
    for case in cases:
        assert case.request.examples == load_examples(EXAMPLES, case.profile)
        assert len(case.request.examples) == 2
