"""Segmentation policy edge cases. Each test maps to a rule in docs/segmentation-policy.md."""

import pytest

from mandarin_core.checker import Status, check, composition_rule
from mandarin_core.segment import fine_segment
from mandarin_core.vocab import VocabProfile

HSK1 = VocabProfile.from_hsk(1)


def statuses(text: str, profile: VocabProfile = HSK1, **kwargs) -> dict[str, Status]:
    report = check(text, profile, **kwargs)
    return {s.text: s.status for s in report.segments if s.status is not None}


def test_lexicon_words_are_atomic_even_when_characters_are_known():
    # Rule 2: knowing 马 and 上 doesn't mean knowing 马上 ("immediately").
    profile = VocabProfile("chars", frozenset({"我", "马", "上", "就", "来"}))
    report = check("我马上就来。", profile)
    assert [s.text for s in report.segments if s.status] == ["我", "马上", "就", "来"]
    assert report.unknown_words == ["马上"]


def test_known_lexicon_word_is_known():
    assert statuses("我马上就来。")["马上"] == Status.KNOWN


def test_higher_level_word_is_unknown_not_decomposed():
    # 不过 is HSK 2; 不 and 过 are HSK 1, but 不过 means "however".
    assert statuses("不过我觉得很好。")["不过"] == Status.UNKNOWN


@pytest.mark.parametrize(
    ("text", "phrase", "rule"),
    [
        ("我们去看看。", "看看", "reduplication"),
        ("他有两本书。", "两本书", "numeral"),
        ("这个很好。", "这个", "determiner"),
        ("我不是学生。", "不是", "negation"),
        ("他有很多朋友。", "很多", "degree"),
        ("我们去看电影吧。", "电影吧", "particle"),
        ("他说得对。", "说得对", "particle"),
        ("我们在马路上。", "马路上", "localizer"),
    ],
)
def test_phrases_are_split_and_labelled(text, phrase, rule):
    # Rule 3.
    report = check(text, HSK1)
    assert report.composed.get(phrase) == rule
    assert phrase not in [s.text for s in report.segments]


def test_unverified_phrase_lowers_strict_coverage_only():
    # Rule 4: 喝茶 matches no pattern. Its parts are known, so coverage is 100%, but strict isn't.
    report = check("我喝茶。", HSK1)
    assert report.unverified == ["喝茶"]
    assert report.coverage == 1.0
    assert report.strict_coverage == pytest.approx(1 / 3)


def test_fine_split_prefers_longest_left_piece():
    assert fine_segment("马路上", {"马", "马路", "路上", "上"}) == ["马路", "上"]


def test_fine_split_keeps_numeral_runs_together():
    assert fine_segment("二十岁", {"二", "十", "岁"}) == ["二十", "岁"]


def test_numeral_run_known_when_all_characters_known():
    # Rule 5.
    assert statuses("我二十岁。")["二十"] == Status.KNOWN


def test_punctuation_digits_and_latin_are_not_counted():
    # Rule 6.
    report = check("2026年，我学习Chinese！", HSK1)
    assert [s.text for s in report.segments if s.status] == ["年", "我", "学习"]
    assert report.word_count == 3
    assert report.han_chars == 4


def test_free_words_count_as_known():
    # Rule 7.
    report = check("小明是学生。", HSK1, free_words=["小明"])
    assert report.unknown_words == []


def test_names_are_not_glued_to_neighbouring_words():
    # jieba alone gives 小 + 明说 ("to say outright"), breaking the allowed name.
    report = check("小明说：“好。”", HSK1, free_words=["小明"])
    assert [s.text for s in report.segments if s.status] == ["小明", "说", "好"]
    assert report.unknown_words == []


def test_unlisted_name_is_unknown():
    report = check("小丽是学生。", HSK1)
    assert report.unknown_words != []


def test_targets_are_counted_separately_from_unknowns():
    report = check("我喜欢苹果。", HSK1, targets=["苹果", "以后"])
    assert statuses("我喜欢苹果。", targets=["苹果"])["苹果"] == Status.TARGET
    assert report.targets_hit == ["苹果"]
    assert report.targets_missing == ["以后"]
    assert report.unknown_words == []
    assert report.coverage == pytest.approx(2 / 3)


def test_learner_words_missing_from_jieba_are_recognized():
    # Custom Anki entries are lexicon words even if jieba's dictionary splits them.
    profile = VocabProfile("custom", HSK1.known | {"咖啡店员"})
    report = check("她是咖啡店员。", profile)
    assert "咖啡店员" in [s.text for s in report.segments]
    assert report.unknown_words == []


@pytest.mark.parametrize(
    ("pieces", "rule"),
    [
        (["天", "天"], "reduplication"),
        (["高", "高", "兴", "兴"], "reduplication"),
        (["北京", "大学"], None),
        (["喝", "茶"], None),
    ],
)
def test_composition_rule(pieces, rule):
    assert composition_rule(pieces) == rule
