import json

from mandarin_core.cli import main


def test_check_json(tmp_path, capsys):
    passage = tmp_path / "p.txt"
    passage.write_text("小明喜欢吃苹果。", encoding="utf-8")
    main(["check", str(passage), "--hsk", "1", "--target", "苹果", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["targets_hit"] == ["苹果"]
    assert report["unknown_words"] == []


def test_check_plain_marks_unknown_words(tmp_path, capsys):
    passage = tmp_path / "p.txt"
    passage.write_text("周末我去超市。", encoding="utf-8")
    main(["check", str(passage), "--plain"])
    out = capsys.readouterr().out
    assert "【周末】" in out and "【超市】" in out


def test_eval_cases_lists_grid(capsys):
    main(["eval", "cases"])
    assert len(capsys.readouterr().out.splitlines()) == 30
