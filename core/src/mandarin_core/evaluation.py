"""Run the evaluation grid through a model and summarize the results."""

import json
import random
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from mandarin_core.generate import (
    Attempt,
    GenerationResult,
    PassageRequest,
    generate,
    load_examples,
    score,
)
from mandarin_core.llm import ChatClient, Completion
from mandarin_core.vocab import VocabProfile, load_hsk


@dataclass(frozen=True)
class EvalCase:
    id: str
    profile: str
    topic: str
    difficulty: str
    request: PassageRequest


def pick_targets(
    case_id: str, level: int, known: frozenset[str], k: int, pool: int
) -> tuple[str, ...]:
    """Sample `k` multi-character words from the `pool` most frequent words of `level`.

    Single-character words (道, 发) are skipped: they have many senses and also appear inside
    other words, which makes "was the target used?" ambiguous.
    """
    candidates = sorted(
        (
            e
            for e in load_hsk().values()
            if e.level == level and len(e.word) > 1 and e.word not in known and e.rank
        ),
        key=lambda e: (e.rank, e.word),
    )[:pool]
    return tuple(e.word for e in random.Random(case_id).sample(candidates, k))


def load_cases(
    path: Path, root: Path | None = None, examples: Path | None = None
) -> list[EvalCase]:
    """Expand the grid in `path`. Relative wordlist paths resolve against `root` (default: cwd).

    With `examples`, each request carries the few-shot examples written for its profile.
    """
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    defaults = config["defaults"]
    root = root or Path.cwd()
    cases = []
    for p in config["profiles"]:
        if "hsk_level" in p:
            profile = VocabProfile.from_hsk(p["hsk_level"])
        else:
            profile = VocabProfile.from_wordlist(root / p["wordlist"], name=p["name"])
        for topic in config["topics"]:
            for d in config["difficulties"]:
                case_id = f"{p['name']}/{topic['id']}/{d['name']}"
                targets = pick_targets(
                    case_id, p["target_level"], profile.known, d["targets"], defaults["target_pool"]
                )
                request = PassageRequest(
                    profile=profile,
                    topic=topic["text"],
                    targets=targets,
                    min_coverage=d["min_coverage"],
                    length=tuple(defaults["length"]),
                    names=tuple(defaults["names"]),
                    examples=load_examples(examples, p["name"]) if examples else (),
                )
                cases.append(EvalCase(case_id, p["name"], topic["id"], d["name"], request))
    return cases


def result_row(case: EvalCase, result: GenerationResult) -> dict:
    return {
        "case": case.id,
        "profile": case.profile,
        "topic": case.topic,
        "difficulty": case.difficulty,
        "targets": list(case.request.targets),
        "min_coverage": case.request.min_coverage,
        "attempts": [
            {
                "kind": a.kind,
                "text": a.completion.text,
                "model": a.completion.model,
                "latency_s": round(a.completion.latency_s, 3),
                "prompt_tokens": a.completion.prompt_tokens,
                "completion_tokens": a.completion.completion_tokens,
                "passed": a.verdict.passed,
                "problems": a.verdict.problems,
                "report": {k: v for k, v in a.report.to_dict().items() if k != "segments"},
            }
            for a in result.attempts
        ],
    }


def run_eval(
    cases: Iterable[EvalCase], client: ChatClient, out_path: Path, max_revisions: int = 1
) -> list[dict]:
    """Generate every case, appending one JSON line per case to `out_path` as it finishes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out_path.open("w", encoding="utf-8") as out:
        for case in cases:
            try:
                row = result_row(case, generate(client, case.request, max_revisions))
            except Exception as exc:  # record and keep going; one bad case shouldn't end the run
                row = {"case": case.id, "difficulty": case.difficulty, "error": repr(exc)}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            rows.append(row)
    return rows


def manual_path(directory: Path, case_id: str, kind: str) -> Path:
    return directory / f"{case_id.replace('/', '__')}.{kind}.txt"


def import_manual(
    cases: Iterable[EvalCase], directory: Path, model: str, out_path: Path
) -> list[dict]:
    """Score hand-written passages (`<case>.initial.txt`, optional `<case>.revision.txt`).

    Used for baselines that aren't reachable through an API, e.g. passages written by a
    frontier model in a chat session under the same one-revision protocol.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out_path.open("w", encoding="utf-8") as out:
        for case in cases:
            attempts: list[Attempt] = []
            for kind in ("initial", "revision"):
                path = manual_path(directory, case.id, kind)
                if not path.exists() or (attempts and attempts[-1].verdict.passed):
                    break
                text = path.read_text(encoding="utf-8").strip()
                attempts.append(score(case.request, Completion(text, 0.0, model), kind))
            if attempts:
                row = result_row(case, GenerationResult(case.request, attempts))
            else:
                row = {"case": case.id, "difficulty": case.difficulty, "error": "no passage"}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    return rows


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarize(rows: list[dict]) -> str:
    """Markdown table: initial attempt vs. after revision, per difficulty and overall."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["difficulty"], []).append(row)
    groups["all"] = rows

    header = (
        "| group | n | errors | pass (initial) | pass (final) | coverage | strict coverage "
        "| unknown words | targets hit | latency s |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for name, group in groups.items():
        ok = [r for r in group if "attempts" in r]
        initial = [r["attempts"][0] for r in ok]
        final = [r["attempts"][-1] for r in ok]
        targets = sum(len(r["targets"]) for r in ok)
        hit = sum(len(a["report"]["targets_hit"]) for a in final)
        lines.append(
            f"| {name} | {len(group)} | {len(group) - len(ok)} "
            f"| {_mean([a['passed'] for a in initial]):.0%} "
            f"| {_mean([a['passed'] for a in final]):.0%} "
            f"| {_mean([a['report']['coverage'] for a in final]):.1%} "
            f"| {_mean([a['report']['strict_coverage'] for a in final]):.1%} "
            f"| {_mean([len(a['report']['unknown_words']) for a in final]):.1f} "
            f"| {hit}/{targets} "
            f"| {_mean([sum(a['latency_s'] for a in r['attempts']) for r in ok]):.1f} |"
        )
    lines.append(
        "\nCoverage, unknown words and targets are for the final attempt. "
        "Latency is the total across attempts."
    )
    return "\n".join(lines)
