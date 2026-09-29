"""Command-line interface: `mandarin check | generate | eval`."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from mandarin_core.checker import Report, Status, check
from mandarin_core.evaluation import load_cases, read_rows, run_eval, summarize
from mandarin_core.generate import DEFAULT_NAMES, GenerationResult, PassageRequest, generate
from mandarin_core.llm import ChatClient
from mandarin_core.vocab import VocabProfile, load_hsk

RED, GREEN, YELLOW, DIM, RESET = "\033[1;31m", "\033[1;32m", "\033[33m", "\033[2m", "\033[0m"


def _profile(args: argparse.Namespace) -> VocabProfile:
    if args.wordlist:
        return VocabProfile.from_wordlist(Path(args.wordlist), column=args.column)
    return VocabProfile.from_hsk(args.hsk)


def _client(args: argparse.Namespace) -> ChatClient:
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_thinking else {}
    return ChatClient(
        base_url=args.base_url, model=args.model, api_key=args.api_key, extra_body=extra
    )


def render(report: Report, plain: bool = False) -> str:
    unverified = set(report.unverified)
    out = []
    for s in report.segments:
        if s.status == Status.UNKNOWN:
            out.append(f"【{s.text}】" if plain else f"{RED}{s.text}{RESET}")
        elif s.status == Status.TARGET:
            out.append(f"〔{s.text}〕" if plain else f"{GREEN}{s.text}{RESET}")
        elif s.composed_from in unverified and not plain:
            out.append(f"{YELLOW}{s.text}{RESET}")
        else:
            out.append(s.text)
    return "".join(out)


def describe(report: Report, plain: bool = False) -> str:
    hsk = load_hsk()
    lines = [
        render(report, plain),
        "",
        f"characters {report.han_chars}  words {report.word_count}  "
        f"coverage {report.coverage:.1%}  strict {report.strict_coverage:.1%}",
    ]
    if report.targets_hit or report.targets_missing:
        lines.append(
            f"targets hit: {'、'.join(report.targets_hit) or '-'}  "
            f"missing: {'、'.join(report.targets_missing) or '-'}"
        )
    dim, reset = ("", "") if plain else (DIM, RESET)
    for word in report.unknown_words:
        entry = hsk.get(word)
        info = f"HSK {entry.level}  {entry.pinyin}  {entry.gloss}" if entry else "not in HSK"
        lines.append(f"  unknown {word}  {dim}{info}{reset}")
    if report.unverified:
        lines.append(f"unverified phrases (review): {'、'.join(report.unverified)}")
    return "\n".join(lines)


def cmd_check(args: argparse.Namespace) -> None:
    text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
    report = check(text, _profile(args), args.target, args.name or DEFAULT_NAMES)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(describe(report, args.plain))


def cmd_generate(args: argparse.Namespace) -> None:
    request = PassageRequest(
        profile=_profile(args),
        topic=args.topic,
        targets=tuple(args.target),
        min_coverage=args.min_coverage,
    )
    result: GenerationResult = generate(_client(args), request, args.revisions)
    for attempt in result.attempts:
        status = "PASS" if attempt.verdict.passed else "FAIL"
        print(f"--- {attempt.kind} ({attempt.completion.latency_s:.1f}s) {status}")
        print(describe(attempt.report, args.plain))
        for problem in attempt.verdict.problems:
            print(f"  problem: {problem}")
        print()


def cmd_eval_cases(args: argparse.Namespace) -> None:
    for case in load_cases(Path(args.cases)):
        r = case.request
        print(f"{case.id:32} targets {'、'.join(r.targets):20} min coverage {r.min_coverage:.0%}")


def cmd_eval_run(args: argparse.Namespace) -> None:
    cases = load_cases(Path(args.cases))[: args.limit]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or f"eval/results/{stamp}-{args.model.replace('/', '_')}.jsonl")
    print(f"running {len(cases)} cases against {args.model} -> {out}", file=sys.stderr)
    rows = run_eval(cases, _client(args), out, args.revisions)
    summary = summarize(rows)
    out.with_suffix(".md").write_text(summary + "\n", encoding="utf-8")
    print(summary)


def cmd_eval_summarize(args: argparse.Namespace) -> None:
    print(summarize(read_rows(Path(args.results))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mandarin", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    vocab = argparse.ArgumentParser(add_help=False)
    group = vocab.add_mutually_exclusive_group()
    group.add_argument("--hsk", type=int, default=1, help="known words: HSK 3.0 levels 1..N")
    group.add_argument("--wordlist", help="known words: word list or Anki plain-text export")
    vocab.add_argument(
        "--column", type=int, help="0-based word column in --wordlist (default: auto-detect)"
    )
    vocab.add_argument("--target", action="append", default=[], help="new word to introduce")
    vocab.add_argument("--plain", action="store_true", help="brackets instead of colors")

    model = argparse.ArgumentParser(add_help=False)
    model.add_argument(
        "--base-url", default=os.environ.get("MANDARIN_BASE_URL", "http://127.0.0.1:8080/v1")
    )
    model.add_argument("--model", default=os.environ.get("MANDARIN_MODEL", "local"))
    model.add_argument("--api-key", default=os.environ.get("MANDARIN_API_KEY"))
    model.add_argument(
        "--no-thinking", action="store_true", help="disable reasoning mode (Qwen via llama.cpp)"
    )
    model.add_argument("--revisions", type=int, default=1, help="max revision rounds")

    p = sub.add_parser("check", parents=[vocab], help="check a passage against a vocabulary")
    p.add_argument("file", nargs="?", default="-", help="passage file, or - for stdin")
    p.add_argument("--name", action="append", default=[], help="allowed name (default: 小明, 小红)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("generate", parents=[vocab, model], help="generate and check a passage")
    p.add_argument("--topic", required=True)
    p.add_argument("--min-coverage", type=float, default=0.95)
    p.set_defaults(func=cmd_generate)

    eval_parser = sub.add_parser("eval", help="evaluation grid")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    cases = argparse.ArgumentParser(add_help=False)
    cases.add_argument("--cases", default="eval/cases.toml")

    p = eval_sub.add_parser("cases", parents=[cases], help="list the resolved cases")
    p.set_defaults(func=cmd_eval_cases)
    p = eval_sub.add_parser("run", parents=[cases, model], help="run all cases through a model")
    p.add_argument("--limit", type=int, help="only the first N cases")
    p.add_argument("--out", help="results path (default eval/results/<time>-<model>.jsonl)")
    p.set_defaults(func=cmd_eval_run)
    p = eval_sub.add_parser("summarize", help="summarize a results file")
    p.add_argument("results")
    p.set_defaults(func=cmd_eval_summarize)
    return parser


def main(argv: list[str] | None = None) -> None:
    # Windows consoles default to a legacy code page.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
