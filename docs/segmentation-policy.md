# Segmentation policy (v0)

The checker's numbers are only as honest as its segmentation. This document is the contract that
`core/src/mandarin_core/segment.py` and `checker.py` implement and `core/tests/test_policy.py` pins
down. Change the policy here first, then the tests, then the code.

## The problem

Knowing characters is not knowing words. A learner who knows 马 ("horse") and 上 ("up") does not
therefore know 马上 ("immediately"). A checker that splits text into the smallest known pieces
rewards exactly the wrong thing: text full of unfamiliar words built from familiar characters.

The opposite error also matters. jieba's dictionary contains many phrases (电影吧, 两本书, 说得对).
Treating those as single unknown words would flag perfectly readable text.

## Rules

1. **Lexicon.** The lexicon is all HSK 3.0 words (levels 1–7) plus the learner's own words, the
   request's target words and allowed names.
2. **Lexicon words are atomic.** A token that is a lexicon word is judged as a whole and never split.
   马上 is an HSK 1 word, so a learner who knows 马 and 上 but not 马上 gets 马上 marked unknown.
3. **Phrases are split and judged by their parts.** A multi-character token that jieba produced but
   that isn't a lexicon word is split into the fewest lexicon words (ties: longest piece first).
   Each piece is judged separately. The phrase is labelled by the pattern it follows:

   | Pattern | Examples |
   |---|---|
   | reduplication | 看看, 天天, 走走 |
   | numeral | 三个, 两本书, 二十岁 |
   | determiner (这那哪每几第) | 这个, 每天 |
   | negation (不没) | 不是, 不想 |
   | degree adverb (很太真最更都也) | 很多 |
   | particle (了的得地着过吧吗呢啊呀啦) | 电影吧, 说得对 |
   | localizer (上下里外中前后边) | 马路上 |
   | *unverified* | 喝茶, 北京大学, 吃完饭 |

4. **Unverified phrases are the documented loophole.** A phrase that matches no pattern might be
   compositional (喝茶 = 喝 + 茶) or might be a real word missing from HSK whose meaning isn't the
   sum of its parts. The report lists them under `unverified`, and `strict_coverage` counts their
   known pieces as unknown. True coverage lies between `strict_coverage` and `coverage`.
5. **Numbers.** A run of numeral characters (二十, 一百) is one word, known if every character is known.
   Arabic digits and Latin text are not counted as words.
6. **Not counted:** punctuation, whitespace, digits, Latin text.
7. **Names.** Names aren't in HSK. The generation prompt gives the model a short list of allowed names,
   which are passed to the checker as `free_words` and count as known. Any other name counts as unknown.
8. **Deterministic.** jieba runs with HMM off, so it never invents words from its statistical model.

## Metrics

- `coverage`: known words / all counted words. Target words count as not known.
- `strict_coverage`: the same, with known pieces of unverified phrases counted as unknown.
- `unknown_words`: distinct words that are neither known nor targets. These are the violations.
- `targets_hit` / `targets_missing`: which requested new words appear as whole words.

## Known gaps

- Word senses: 花 "flower" vs. 花 "spend". The checker cannot tell which sense the learner knows.
- Separable verbs (帮忙 → 帮他的忙) are split and judged by their parts.
- Traditional characters are not handled.
- The pattern lists are hand-picked. Review `unverified` in eval reports and extend them deliberately.
