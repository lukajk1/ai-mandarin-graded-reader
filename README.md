# Mandarin graded reader

Generates short Mandarin reading passages that fit a learner's vocabulary, introduce chosen new
words, and measure how well they did. The research question: can a small, adapted model do
this better and more cheaply than a frontier model with prompting? See [PLAN.md](PLAN.md).

Current stage: **Phase 1**, the vocabulary checker and evaluation grid, plus a prompted baseline loop.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest
```

## Usage

Check a passage against HSK 1 vocabulary, with 苹果 as an intended new word:

```
echo 我喜欢吃苹果，不过太贵了。 | uv run mandarin check --hsk 1 --target 苹果
```

Unknown words are shown in red, target words in green, and unverified phrases in yellow (see the
[segmentation policy](docs/segmentation-policy.md)). Use `--wordlist path/to/anki.txt` instead of
`--hsk` for your own vocabulary; an Anki "Notes in Plain Text" export works directly.

Generate with any OpenAI-compatible endpoint, e.g. llama.cpp's `llama-server` on its default port:

```
uv run mandarin generate --hsk 1 --topic "在超市买东西" --target 超市 --no-thinking
```

Run the 30-case evaluation grid and write a JSONL results file plus a Markdown summary to
`eval/results/`:

```
uv run mandarin eval cases                     # list the cases
uv run mandarin eval run --model qwen3.5-2b --no-thinking
uv run mandarin eval summarize eval/results/<file>.jsonl
```

Endpoint settings: `--base-url` (default `http://127.0.0.1:8080/v1`), `--model`, `--api-key`, or
the environment variables `MANDARIN_BASE_URL`, `MANDARIN_MODEL`, `MANDARIN_API_KEY`.

## Layout

```
core/     mandarin_core: segmentation, vocabulary profiles, checker, generation, evaluation, CLI
eval/     evaluation grid (cases.toml); results/ is gitignored
data/     HSK word lists (see data/README.md)
docs/     segmentation policy
scripts/  data fetching
```
