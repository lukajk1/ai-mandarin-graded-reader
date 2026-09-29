# Baselines, 2026-09-29

30 cases from `eval/cases.toml` (3 HSK profiles x 5 topics x 2 difficulties). Each system gets one
initial attempt and at most one revision with the checker's feedback. A case passes when coverage
meets the minimum (95% gentle, 90% stretch), every target word appears, and the length is 80–180
characters.

| System | Pass (initial) | Pass (final) | Coverage | Strict coverage | Unknown words / passage | Targets hit | Latency / case |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B Q4_K_M | 0% | 0% | 80.6% | 76.2% | 12.9 | 72/90 | 2.8 s |
| Qwen3.5-2B + 2 examples | 3% | 3% | 82.5% | 77.8% | 9.6 | 76/90 | 2.6 s |
| Qwen3.5-9B Q4_K_M | 0% | 10% | 85.9% | 79.8% | 8.6 | 84/90 | 7.0 s |
| Qwen3.5-9B + 2 examples | 0% | 7% | 86.3% | 79.9% | 8.1 | 86/90 | 7.4 s |
| Claude Opus 5.5 (manual) | 97% | 100% | 95.9% | 92.7% | 0.1 | 90/90 | n/a |

Qwen models ran locally through llama.cpp (Vulkan, RX 6800 XT) with thinking disabled. Examples
are `data/examples.toml`: two checked passages per profile on topics and target words outside the
grid.

## Findings

- **Prompting doesn't give open models vocabulary control.** Even with the full known-word list in
  the prompt, Qwen passages contain 8–13 off-list words each. The failure is almost entirely
  coverage (29 of 30 cases for the 2B model); lengths and target words are mostly fine.
- **Size helps a little; examples barely help.** 2B -> 9B raises coverage by about 5 points and the
  final pass rate to 10%. Adding examples moves coverage by under 2 points at either size.
- **Revision rarely rescues a small model.** The 2B model rewrote every failing passage and never
  fixed one. The 9B model fixed 3.
- **The frontier baseline is near the ceiling.** 29/30 on the first attempt.

## Caveats

- n = 30, one run each, sampling temperature 0.7. Differences of a case or two are noise.
- The Opus passages were written in a chat session by three sub-agents following the same protocol
  (initial passage written without the checker, one check, one unchecked revision). They could read
  the word-list files; the Qwen models had the same list in their prompt. No latency or cost figures.
- On strict coverage, 3 of the Opus hsk1-2 passages fall below their minimum because of unverified
  phrases (e.g. 老照片, 坐火车). Strict coverage is a pessimistic bound, not the pass criterion.
- Only `coverage` is measured. Nobody has yet rated fluency or naturalness.

## What this means for training data

Filtering 9B outputs by the checker keeps roughly 1 in 10 passages. A few thousand training examples
would take tens of thousands of generations. Cheaper routes to consider first:

1. **Constrained decoding** (llama.cpp grammar restricted to allowed words) attacks the failure mode
   directly, both at inference time and as a way to generate passing training data.
2. **A larger open teacher** (e.g. Qwen3.6-35B-A3B with partial CPU offload).
3. **More revision rounds** for the teacher when generating data (offline, latency doesn't matter).

Raw results: `eval/results/` (not committed). Opus passages: `eval/manual/opus-5.5/`.
