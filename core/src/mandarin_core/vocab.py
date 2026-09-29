"""HSK word lists and learner vocabulary profiles."""

import hashlib
import html
import os
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from mandarin_core.segment import HAN_RE

HSK_LEVELS = range(1, 8)  # level 7 is the HSK 3.0 7-9 band

_TAG_RE = re.compile(r"<[^>]+>")
_WORD_RE = re.compile(f"(?:{HAN_RE.pattern}){{1,8}}")


def data_dir() -> Path:
    override = os.environ.get("MANDARIN_DATA_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "data"


@dataclass(frozen=True)
class HskEntry:
    word: str
    level: int
    pinyin: str
    gloss: str
    rank: int | None


@cache
def load_hsk() -> dict[str, HskEntry]:
    """All HSK 3.0 words. A word listed at several levels keeps its lowest level."""
    entries: dict[str, HskEntry] = {}
    for level in HSK_LEVELS:
        path = data_dir() / "hsk" / f"level{level}.tsv"
        lines = path.read_text(encoding="utf-8").splitlines()[1:]
        for line in lines:
            word, pinyin, gloss, rank = line.split("\t")
            if word not in entries:
                entries[word] = HskEntry(word, level, pinyin, gloss, int(rank) if rank else None)
    return entries


def hsk_words(max_level: int, min_level: int = 1) -> frozenset[str]:
    return frozenset(e.word for e in load_hsk().values() if min_level <= e.level <= max_level)


def _clean_field(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text)).strip()


def _is_word(text: str) -> bool:
    return bool(_WORD_RE.fullmatch(text))


def detect_word_column(rows: list[list[str]]) -> int:
    """Index of the column that most often holds a single short Chinese word.

    Note types differ: some start with the word, others with a note number or sort field.
    Ties go to the leftmost column (simplified usually comes before traditional).
    """
    width = max((len(r) for r in rows), default=0)
    counts = [sum(_is_word(_clean_field(r[i])) for r in rows if i < len(r)) for i in range(width)]
    return max(range(width), key=lambda i: (counts[i], -i)) if width else 0


def parse_wordlist(text: str, column: int | None = None) -> list[str]:
    """Parse a word list: one word per line, or an Anki "Notes in Plain Text" export.

    Lines are split on tabs; the word column is detected unless `column` is given. HTML is
    stripped, and Anki's `#key:value` header lines and entries that aren't Chinese words are
    skipped.
    """
    rows = [line.split("\t") for line in text.splitlines() if line and not line.startswith("#")]
    if column is None:
        column = detect_word_column(rows)
    words = [_clean_field(r[column]) for r in rows if column < len(r)]
    return list(dict.fromkeys(w for w in words if _is_word(w)))


@dataclass(frozen=True)
class VocabProfile:
    name: str
    known: frozenset[str]
    learning: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_hsk(cls, max_level: int) -> "VocabProfile":
        name = "hsk1" if max_level == 1 else f"hsk1-{max_level}"
        return cls(name=name, known=hsk_words(max_level))

    @classmethod
    def from_wordlist(
        cls, path: Path, name: str | None = None, column: int | None = None
    ) -> "VocabProfile":
        words = parse_wordlist(path.read_text(encoding="utf-8-sig"), column)
        return cls(name=name or path.stem, known=frozenset(words))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for word in sorted(self.known):
            digest.update(word.encode("utf-8") + b"\n")
        digest.update(b"--learning--\n")
        for word in sorted(self.learning):
            digest.update(word.encode("utf-8") + b"\n")
        return digest.hexdigest()[:16]
