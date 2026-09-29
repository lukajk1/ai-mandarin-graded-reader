from pathlib import Path

from mandarin_core.evaluation import load_cases, read_rows, run_eval, summarize
from mandarin_core.vocab import hsk_words

REPO = Path(__file__).resolve().parents[2]
CASES = REPO / "eval" / "cases.toml"


def test_grid_has_30_cases_with_unique_ids():
    cases = load_cases(CASES, root=REPO)
    assert len(cases) == 30
    assert len({c.id for c in cases}) == 30


def test_targets_are_deterministic_and_unknown_to_the_profile():
    first = load_cases(CASES, root=REPO)
    second = load_cases(CASES, root=REPO)
    assert [c.request.targets for c in first] == [c.request.targets for c in second]
    for case in first:
        assert not set(case.request.targets) & case.request.profile.known


def test_targets_come_from_the_next_level():
    case = next(c for c in load_cases(CASES, root=REPO) if c.profile == "hsk1")
    assert set(case.request.targets) <= hsk_words(2) - hsk_words(1)


def test_run_eval_writes_rows_and_summary(tmp_path, make_llm):
    cases = load_cases(CASES, root=REPO)[:2]
    llm = make_llm("我是学生。")
    out = tmp_path / "run.jsonl"
    rows = run_eval(cases, llm.client(), out, max_revisions=1)

    assert read_rows(out) == rows
    assert len(rows) == 2
    assert [a["kind"] for a in rows[0]["attempts"]] == ["initial", "revision"]
    assert rows[0]["attempts"][0]["report"]["targets_missing"] == rows[0]["targets"]

    table = summarize(rows)
    assert "| all | 2 | 0 | 0% | 0% |" in table


def test_run_eval_records_errors_and_continues(tmp_path):
    from mandarin_core.llm import ChatClient

    cases = load_cases(CASES, root=REPO)[:2]
    client = ChatClient(base_url="http://127.0.0.1:9/v1", model="nothing", timeout_s=1)
    rows = run_eval(cases, client, tmp_path / "run.jsonl")
    assert all("error" in r for r in rows)
    assert "| all | 2 | 2 |" in summarize(rows)
