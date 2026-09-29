"""Word segmentation. See docs/segmentation-policy.md for the rules this implements.

Two passes:
- coarse: jieba with its full dictionary (HMM off, so it never invents words). This finds the
  units a reader perceives, e.g. 马上 as one word rather than 马 + 上.
- fine: coarse tokens that are not lexicon words (jieba's dictionary contains many phrases like
  电影吧 or 两本书) are split into lexicon words by a minimum-piece dynamic program.
"""

import logging
import re
from collections.abc import Collection, Iterable
from functools import cache
from typing import Literal

import jieba

Kind = Literal["han", "punct", "other", "space"]

HAN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
NUMERAL_CHARS = frozenset("零〇一二三四五六七八九十百千万亿两")
MAX_WORD_LEN = 8
MAX_MERGE_SPAN = 4


def is_han(text: str) -> bool:
    return bool(HAN_RE.search(text))


def kind_of(text: str) -> Kind:
    if is_han(text):
        return "han"
    if text.isspace():
        return "space"
    if any(ch.isalnum() for ch in text):
        return "other"
    return "punct"


def is_numeral_run(text: str) -> bool:
    return len(text) > 0 and all(ch in NUMERAL_CHARS for ch in text)


@cache
def _tokenizer() -> jieba.Tokenizer:
    jieba.setLogLevel(logging.ERROR)
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    return tokenizer


def coarse_segment(text: str) -> list[str]:
    return _tokenizer().lcut(text, HMM=False)


def merge_lexical(tokens: list[str], lexicon: Collection[str]) -> list[str]:
    """Greedily merge adjacent Han tokens whose concatenation is a lexicon word.

    Needed for learner words that jieba's dictionary doesn't contain (e.g. custom Anki entries).
    """
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        for span in range(min(MAX_MERGE_SPAN, len(tokens) - i), 1, -1):
            parts = tokens[i : i + span]
            if all(is_han(p) for p in parts) and "".join(parts) in lexicon:
                merged.append("".join(parts))
                i += span
                break
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def fine_segment(token: str, lexicon: Collection[str]) -> list[str]:
    """Split a token into the fewest pieces, preferring lexicon words over stray characters.

    Pieces are lexicon words, numeral runs (二十, 一百) or single characters. Minimizes
    (pieces that are not lexicon words or numerals, total pieces); ties prefer longer pieces on
    the left, like forward maximum matching (马路上 -> 马路 + 上, not 马 + 路上).
    """
    n = len(token)
    best: list[tuple[int, int] | None] = [None] * (n + 1)
    back = [0] * (n + 1)
    best[0] = (0, 0)
    for i in range(n):
        cost = best[i]
        if cost is None:
            continue
        for j in range(i + 1, min(n, i + MAX_WORD_LEN) + 1):
            piece = token[i:j]
            is_word = piece in lexicon or is_numeral_run(piece)
            if not is_word and j - i > 1:
                continue
            candidate = (cost[0] + (0 if is_word else 1), cost[1] + 1)
            current = best[j]
            if current is None or candidate < current or (candidate == current and i > back[j]):
                best[j] = candidate
                back[j] = i
    pieces = []
    j = n
    while j > 0:
        i = back[j]
        pieces.append(token[i:j])
        j = i
    return pieces[::-1]


def han_char_count(text: str) -> int:
    return len(HAN_RE.findall(text))


def segment(text: str, lexicon: Collection[str], protected: Iterable[str] = ()) -> list[str]:
    """Coarse segmentation followed by lexicon merging. Fine splitting happens in the checker.

    `protected` strings (character names) are cut out before jieba runs, so it can't glue them
    to neighbours: 小明说 would otherwise become 小 + 明说 ("to say outright").
    """
    names = sorted({p for p in protected if p}, key=len, reverse=True)
    if not names:
        return merge_lexical(coarse_segment(text), lexicon)
    tokens: list[str] = []
    for part in re.split(f"({'|'.join(map(re.escape, names))})", text):
        if part in names:
            tokens.append(part)
        elif part:
            tokens.extend(merge_lexical(coarse_segment(part), lexicon))
    return tokens


def tokens_of_kind(tokens: Iterable[str], kind: Kind) -> list[str]:
    return [t for t in tokens if kind_of(t) == kind]
