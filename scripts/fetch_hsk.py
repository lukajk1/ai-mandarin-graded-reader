"""Download HSK 3.0 word lists and write compact per-level TSV files to data/hsk/.

Source: https://github.com/drkameleon/complete-hsk-vocabulary (MIT). Uses the "exclusive"
lists, so each file holds only the words introduced at that level. Level 7 is the 7-9 band.

Output columns: word, pinyin, gloss (first definition), frequency rank.

Run with: uv run python scripts/fetch_hsk.py
"""

import json
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/drkameleon/complete-hsk-vocabulary/"
    "main/wordlists/exclusive/new/{level}.min.json"
)
LEVELS = range(1, 8)
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "hsk"


def pick_reading(forms: list[dict]) -> tuple[str, str]:
    """First (pinyin, meaning) that isn't a surname reading, e.g. 米 -> rice, not "surname Mi"."""
    candidates = [
        (form.get("i", {}).get("y", ""), meaning) for form in forms for meaning in form.get("m", [])
    ]
    for pinyin, meaning in candidates:
        if not meaning.startswith("surname"):
            return pinyin, meaning
    return candidates[0] if candidates else ("", "")


def convert(entries: list[dict]) -> list[str]:
    rows = []
    seen = set()
    for entry in entries:
        word = entry["s"]
        if word in seen:
            continue
        seen.add(word)
        pinyin, gloss = pick_reading(entry.get("f", []))
        rank = entry.get("q", "")
        rows.append("\t".join(str(x).replace("\t", " ") for x in (word, pinyin, gloss, rank)))
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for level in LEVELS:
        url = BASE_URL.format(level=level)
        with urllib.request.urlopen(url, timeout=60) as response:
            entries = json.loads(response.read().decode("utf-8"))
        rows = convert(entries)
        out = OUT_DIR / f"level{level}.tsv"
        out.write_text("word\tpinyin\tgloss\trank\n" + "\n".join(rows) + "\n", encoding="utf-8")
        print(f"level {level}: {len(rows)} words -> {out.name}")


if __name__ == "__main__":
    main()
