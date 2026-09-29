# Plan

**Core question:** can a small, adapted model write personalized Mandarin reading passages with better
vocabulary control, at lower inference cost, than a frontier model with prompting + checking?

## Decisions so far

- **Scope v1:** simplified Mandarin, ~100–150 character passages.
- **Vocabulary input:** Anki add-on (word-level notes; one word per note) is primary. HSK 3.0 levels are the fallback.
  - Card state gives three tiers: known (mature, interval >= 21d), learning (target-word candidates), unknown.
  - v1: add-on exports a profile JSON -> web app imports -> browser storage. No accounts.
  - v2: add-on pushes to the API with a sync token -> Postgres (Neon/Supabase free tier). Space disk isn't persistent.
  - Server-side in-memory cache of derived data (allowed-word set, compiled grammar) keyed by profile hash.
  - The learner's words also go into the segmenter's user dictionary.
- **Models:** Qwen3.5 family. 2B = main/fine-tune target, 4B = quality ceiling, 0.8B = speed floor.
  Frontier model + checker + one revision = reference baseline (evaluation only, never training data).
  Confirm license and how to disable thinking mode on the model cards.
- **Training data:** synthetic passages from a larger open Qwen model, filtered by our own checker.
- **Training compute:** Unsloth LoRA on Kaggle GPUs (local GPU is AMD, so no CUDA). Hourly rented GPU as fallback.
- **Hosting:** Hugging Face PRO ($9/mo) + Docker Space, CPU Basic first, CPU Upgrade ($0.03/hr) if benchmarks need it.
  Don't buy until Phase 3. Everything before that runs locally.

## Phases

0. **Setup:** repo, uv project, CI (lint + tests), data sources (HSK lists, CC-CEDICT, SUBTLEX-CH), segmentation policy doc.
   *Done (2026-09-27), except CC-CEDICT and SUBTLEX-CH, which aren't needed yet.*
1. **Checker + eval set:** `core/` checker, ~30 cases (3 profiles x 5 topics x 2 difficulties), segmentation edge-case tests.
   *Built (2026-09-27), along with a generate -> check -> revise loop and `mandarin eval run`. Next: manually review
   segmentation on real model output, and add your Anki export as a 4th profile.*
2. **Baselines:** frontier (prompt -> check -> revise) and small open models (prompted) on all cases. Results table.
3. **Thin end-to-end slice, deployed:** React/Vite -> FastAPI -> llama.cpp -> checker -> highlighted passage.
   Basic observability (per-generation log line) and resilience (timeouts, concurrency limit, saved-example fallback).
4. **Experiments:** constrained decoding (GBNF), then SFT (LoRA). Compare the 2x2 + frontier reference on held-out cases.
   - **4b (optional) — GRPO with the checker as reward.** Only if SFT leaves room for improvement.
     Reward: few off-list words, density close to target, target words used. Needs a repetition penalty and a
     fluency signal, or the model will exploit the reward (repetitive "baby talk" scores perfectly). Read outputs by hand.
     Compare base / SFT / SFT+GRPO. Unsloth supports GRPO on this model size.
5. **Presentation:** results page in the demo, README, short video.

## Later / maybe

- Japanese as a generalization test (different segmentation, kanji/kana readings).
- Mandarin tone coach (read passage aloud -> tone feedback). Separate audio-ML project.
- RAG only if a concrete feature needs it (e.g. "rewrite this article at my level").
- Use it daily for a few weeks; check whether learning words mature faster in Anki.

## Repo layout

```
core/  eval/  train/  api/  web/  anki-addon/  data/  .github/workflows/
```
Python: uv, ruff, pytest, pyright. `core/` is shared by eval, api, training-data filtering and CI.
