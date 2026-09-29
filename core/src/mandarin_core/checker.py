"""Measure how well a passage fits a learner's vocabulary (docs/segmentation-policy.md)."""

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from mandarin_core.segment import (
    Kind,
    fine_segment,
    han_char_count,
    is_numeral_run,
    kind_of,
    segment,
)
from mandarin_core.vocab import VocabProfile, load_hsk


class Status(StrEnum):
    KNOWN = "known"
    TARGET = "target"
    UNKNOWN = "unknown"


# Productive patterns whose meaning follows from their parts (policy rule 3).
PARTICLES = frozenset("了的得地着过吧吗呢啊呀啦")
DETERMINERS = frozenset("这那哪每几第")
NEGATIONS = frozenset("不没")
DEGREE_ADVERBS = frozenset("很太真最更都也")
LOCALIZERS = frozenset("上下里外中前后边")


def composition_rule(pieces: Sequence[str]) -> str | None:
    """Name the grammatical pattern a decomposed phrase follows, or None if unverified."""
    if len(set(pieces)) == 1 or (
        len(pieces) == 4 and pieces[0] == pieces[1] and pieces[2] == pieces[3]
    ):
        return "reduplication"
    if is_numeral_run(pieces[0]):
        return "numeral"
    if pieces[0] in DETERMINERS:
        return "determiner"
    if pieces[0] in NEGATIONS:
        return "negation"
    if pieces[0] in DEGREE_ADVERBS:
        return "degree"
    if any(p in PARTICLES for p in pieces):
        return "particle"
    if pieces[-1] in LOCALIZERS:
        return "localizer"
    return None


@dataclass(frozen=True)
class Segment:
    text: str
    kind: Kind
    status: Status | None = None
    # The coarse token this piece was split from, when the fine pass decomposed it.
    composed_from: str | None = None


@dataclass
class Report:
    segments: list[Segment]
    han_chars: int
    word_count: int
    known_count: int
    target_count: int
    unknown_count: int
    # Share of words the learner knows.
    coverage: float
    # Coverage if known pieces of unverified phrases (see `unverified`) counted as unknown.
    # The true value lies between the two; a large gap means those phrases need review.
    strict_coverage: float
    unknown_words: list[str] = field(default_factory=list)
    targets_hit: list[str] = field(default_factory=list)
    targets_missing: list[str] = field(default_factory=list)
    # Decomposed phrases that matched a grammatical pattern, mapped to the pattern name.
    composed: dict[str, str] = field(default_factory=dict)
    # Decomposed phrases that matched no pattern (e.g. 北京大学, 喝茶). Candidates for review.
    unverified: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["segments"] = [
            {k: v for k, v in asdict(s).items() if v is not None} for s in self.segments
        ]
        return data


def _status(word: str, known: frozenset[str], targets: frozenset[str]) -> Status:
    if word in known:
        return Status.KNOWN
    if is_numeral_run(word) and all(ch in known for ch in word):
        return Status.KNOWN
    if word in targets:
        return Status.TARGET
    return Status.UNKNOWN


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def check(
    text: str,
    profile: VocabProfile,
    targets: Iterable[str] = (),
    free_words: Iterable[str] = (),
) -> Report:
    """Segment `text` and classify each word as known, target or unknown for `profile`.

    `targets` are the new words the passage is meant to introduce. `free_words` (e.g. character
    names allowed by the prompt) count as known.
    """
    target_list = _unique(targets)
    target_set = frozenset(target_list)
    known = profile.known | frozenset(free_words)
    lexicon = known | target_set | profile.learning | load_hsk().keys()

    segments: list[Segment] = []
    composed: dict[str, str] = {}
    unverified: list[str] = []
    unverified_known = 0
    for token in segment(text, lexicon):
        kind = kind_of(token)
        if kind != "han":
            segments.append(Segment(token, kind))
            continue
        if token in lexicon or len(token) == 1 or is_numeral_run(token):
            segments.append(Segment(token, kind, _status(token, known, target_set)))
            continue
        pieces = fine_segment(token, lexicon)
        judged = [Segment(p, "han", _status(p, known, target_set), token) for p in pieces]
        segments.extend(judged)
        rule = composition_rule(pieces)
        if rule:
            composed[token] = rule
        else:
            unverified.append(token)
            unverified_known += sum(s.status == Status.KNOWN for s in judged)

    words = [s for s in segments if s.status is not None]
    counts = {status: sum(s.status == status for s in words) for status in Status}
    word_count = len(words)
    present = {s.text for s in words}

    def share(n: int) -> float:
        return n / word_count if word_count else 0.0

    return Report(
        segments=segments,
        han_chars=han_char_count(text),
        word_count=word_count,
        known_count=counts[Status.KNOWN],
        target_count=counts[Status.TARGET],
        unknown_count=counts[Status.UNKNOWN],
        coverage=share(counts[Status.KNOWN]),
        strict_coverage=share(counts[Status.KNOWN] - unverified_known),
        unknown_words=_unique(s.text for s in words if s.status == Status.UNKNOWN),
        targets_hit=[t for t in target_list if t in present],
        targets_missing=[t for t in target_list if t not in present],
        composed=composed,
        unverified=_unique(unverified),
    )
