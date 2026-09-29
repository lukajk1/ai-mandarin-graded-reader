# Data

## hsk/

HSK 3.0 word lists, one file per level (`level7.tsv` is the 7–9 band). Each file holds only the
words introduced at that level. Columns: `word`, `pinyin`, `gloss` (first definition), `rank`
(frequency rank).

Derived from [drkameleon/complete-hsk-vocabulary](https://github.com/drkameleon/complete-hsk-vocabulary)
(MIT License). Its definitions come from CC-CEDICT (CC BY-SA 4.0). Regenerate with:

```
uv run python scripts/fetch_hsk.py
```

## private/ (not committed)

Your own vocabulary, e.g. an Anki "Notes in Plain Text" export saved as `private/anki.txt`.
It's gitignored.
