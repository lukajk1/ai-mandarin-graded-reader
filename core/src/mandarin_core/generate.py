"""Prompted passage generation with checker feedback: generate, check, optionally revise."""

from dataclasses import dataclass, field
from typing import Literal

from mandarin_core.checker import Report, check
from mandarin_core.llm import ChatClient, Completion
from mandarin_core.vocab import VocabProfile

DEFAULT_NAMES = ("小明", "小红")
LENGTH_TOLERANCE = 0.2

SYSTEM_PROMPT = (
    "You write short reading passages in simplified Chinese for language learners. "
    "Output only the passage in Chinese characters: no title, pinyin, translation or notes."
)


@dataclass(frozen=True)
class PassageRequest:
    profile: VocabProfile
    topic: str
    targets: tuple[str, ...]
    min_coverage: float
    length: tuple[int, int] = (100, 150)
    names: tuple[str, ...] = DEFAULT_NAMES


@dataclass(frozen=True)
class Verdict:
    passed: bool
    problems: list[str]


def judge(report: Report, request: PassageRequest) -> Verdict:
    lo, hi = request.length
    problems = []
    if report.coverage < request.min_coverage:
        problems.append(
            f"coverage {report.coverage:.1%} below {request.min_coverage:.0%}; "
            f"words outside the list: {'、'.join(report.unknown_words)}"
        )
    if report.targets_missing:
        problems.append(f"missing new words: {'、'.join(report.targets_missing)}")
    if not lo * (1 - LENGTH_TOLERANCE) <= report.han_chars <= hi * (1 + LENGTH_TOLERANCE):
        problems.append(f"length {report.han_chars} characters, wanted {lo}-{hi}")
    return Verdict(passed=not problems, problems=problems)


@dataclass
class Attempt:
    kind: Literal["initial", "revision"]
    completion: Completion
    report: Report
    verdict: Verdict


@dataclass
class GenerationResult:
    request: PassageRequest
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def final(self) -> Attempt:
        return self.attempts[-1]


def build_messages(request: PassageRequest) -> list[dict[str, str]]:
    lo, hi = request.length
    known = "、".join(sorted(request.profile.known))
    user = (
        f"Write a passage about: {request.topic}\n"
        f"Length: {lo}-{hi} Chinese characters.\n"
        f"Use only words from the known-word list below, plus these new words, "
        f"each at least once: {'、'.join(request.targets)}.\n"
        f"If you need names, use only: {'、'.join(request.names)}.\n"
        f"At least {request.min_coverage:.0%} of the words must come from the known-word list.\n"
        f"\nKnown words:\n{known}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_revision_messages(request: PassageRequest, previous: Attempt) -> list[dict[str, str]]:
    feedback = "Your passage has these problems:\n" + "\n".join(
        f"- {p}" for p in previous.verdict.problems
    )
    feedback += "\nRewrite the passage to fix them. Keep the same topic. Output only the passage."
    return [
        *build_messages(request),
        {"role": "assistant", "content": previous.completion.text},
        {"role": "user", "content": feedback},
    ]


def _attempt(
    client: ChatClient,
    request: PassageRequest,
    messages: list[dict[str, str]],
    kind: Literal["initial", "revision"],
) -> Attempt:
    completion = client.complete(messages)
    report = check(completion.text, request.profile, request.targets, request.names)
    return Attempt(kind, completion, report, judge(report, request))


def generate(
    client: ChatClient, request: PassageRequest, max_revisions: int = 1
) -> GenerationResult:
    result = GenerationResult(request)
    result.attempts.append(_attempt(client, request, build_messages(request), "initial"))
    while not result.final.verdict.passed and len(result.attempts) <= max_revisions:
        messages = build_revision_messages(request, result.final)
        result.attempts.append(_attempt(client, request, messages, "revision"))
    return result
