# Internal LLM Evaluation Pipeline

End-to-end local LLM evaluation pipeline built on **Ollama** (`llama3:8b`)
and **lm-evaluation-harness** (v0.4.11).

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
make serve          # starts Ollama + ensures model is pulled
make eval           # runs HellaSwag, MMLU, and custom benchmark
make perf           # load test + analysis notebook
make guardrails     # 5-layer determinism/validation suite
make improve        # Part E: full benchmark improvement pipeline
```

## Project Structure

```
serve/              Part A — Inference serving
  serve.py            Ollama server lifecycle (start/stop/health)
  client.py           Typed HTTP client with streaming TTFT support

eval_runner/        Part B — Evaluation harness integration
  model.py            Custom lm-eval wrapper (loglikelihood + generate_until)
  run_eval.py         Benchmark runner (HellaSwag, MMLU, code_output)
  tasks/              Custom task definitions (code_output YAML + JSONL)
  results/            Benchmark output JSONs and summary table

perf/               Part C — Performance & scaling
  load_test.py        Concurrent load generator (TTFT, tok/s, latency)
  metrics.csv         Raw per-request metrics
  analysis.ipynb      Plots and commentary (P50/P95/P99, queueing effects)

guardrails/         Part D — Determinism & validation
  validate.py         5-layer progressive validation suite
  README.md           Detailed test descriptions and nondeterminism analysis

improve/            Part E — Benchmark improvement (HellaSwag)
  prepare_data.py     TF-IDF semantic few-shot selection
  optimize_prompt.py  Ablation over few-shot/template configurations
  infer.py            Final evaluation with bootstrap CI and McNemar test
  eval.sh             One-command orchestration script
  report.md           Results, analysis, and reproducibility details
```

## Architecture Decisions

### Scoring Strategy (eval_runner/model.py)

Ollama v0.16.2 does not support `/v1/completions` with `echo=true`, so
teacher-forced loglikelihood scoring is unavailable.  We approximate it
with **token-by-token greedy scoring** via `/api/generate`:

1. Generate one token with `logprobs=true, top_logprobs=20`
2. Match against the continuation; record the logprob
3. On miss: two modes available (selectable via `scoring_mode`):
   - **`hard_floor`** (Part B): assign -100 and stop — fast but lossy
   - **`soft_floor`** (Part E, default): assign `min(top-20) - 1.0`
     and advance char-by-char — slower but scores full continuations

Both modes are preserved for cross-part comparison.  See
`eval_runner/model.py` docstring for full details.

### Caching

Two layers: (1) in-memory `_gen1_cache` deduplicates API calls within a
run (all MC options sharing a context hit this), (2) `CachingLM` SQLite
cache from lm-eval provides persistence across runs.

### Deterministic Mode

All evaluation uses `temperature=0, top_p=1, seed=42`.  Guardrails
validate this at five layers from config audit through harness-level
reproducibility.  Micro-nondeterminism (delta <= 6e-4) exists in logprob
values due to floating-point non-associativity but does not affect
rankings.  See `guardrails/README.md`.

## Key Limitations

- **Approximate loglikelihood**: Without `echo` support, absolute accuracy
  is lower than published benchmarks.  Relative rankings remain valid.
- **Top-20 truncation**: Ollama caps `top_logprobs` at 20; tokens outside
  this window receive an estimated score.
- **Char-by-char advance**: On token miss, prompts are sent at sub-token
  boundaries, which may subtly affect the model's next-token distribution.
- **Single-GPU throughput**: No tensor parallelism; concurrent requests
  serialize through Ollama's internal scheduler.

## Reproducibility

```
Model:    llama3:8b (Q4_0)
Ollama:   v0.16.2
lm-eval:  v0.4.11
Python:   3.14
Seed:     42
Platform: macOS (Apple Silicon)
```

## Summary: Best Improvements

The central challenge of this project was bridging Ollama's local inference
API to lm-eval's evaluation framework when the standard scoring path
(teacher-forced loglikelihood via `/v1/completions` with `echo=true`) is
unavailable in Ollama v0.16.2.

**The scoring problem.** lm-eval's multiple-choice benchmarks (HellaSwag,
MMLU) rank answer options by loglikelihood. Ollama only exposes logprobs
for *generated* tokens, not arbitrary prompt continuations. Our initial
approach (Part B) approximated this with token-by-token greedy scoring:
generate one token, look up the continuation token in the top-20 logprobs,
record it, and advance. When a token fell outside the top-20, we assigned a
hard floor of -100 and stopped scoring. This produced HellaSwag accuracy of
~36% — a single missed token dominated the entire score.

**The fix.** In Part E, we replaced the hard floor with a *soft floor*
(`min(top-20 logprobs) - 1.0`) and continued scoring character-by-character
on misses instead of stopping. This calibration change raised the 0-shot
HellaSwag baseline from 36% to 61% acc_norm — the single largest lever in
the entire project, requiring zero changes to the model, prompt, or Ollama
configuration. Combined with semantic few-shot selection, the final
acc_norm reaches 65%.

**Stacking few-shot on top.** With a calibrated scoring baseline, we added
TF-IDF semantic few-shot selection: for each validation question, the k
most similar training examples (by cosine similarity on activity
descriptions) are prepended to the prompt. We tested k=5 and k=10, and
found that **5-shot semantic outperforms 10-shot** — likely because shorter
prompts accumulate less scoring noise under the token-by-token method. The
best config (5-shot semantic) reaches **acc_norm=0.65**, a **+4.0 pp lift
that exceeds the +3.0 target**. Full ablation and 9 before/after examples
in `improve/report.md`.

**Interesting learnings**

The biggest surprise was that the most impactful change had nothing to do
with prompts. I spent time trying few-shot strategies and templates, but
the real problem was in my own scoring adapter — one bad constant (-100
hard floor) was destroying the signal for every evaluation downstream.
Once I fixed that, accuracy jumped 25 pp and suddenly the prompt-level
levers started working as expected. Few-shot went from hurting accuracy
to helping by +4 pp, just because the scorer could now tell correct from
incorrect continuations.

A few specific things I'd do differently or carry forward:

- I underestimated how much time the token-by-token scoring would take.
  Each question scores 4 options sequentially, each option requiring
  dozens of API round-trips. A proper `echo` endpoint would cut this by
  10–50×. Next time I'd verify the API surface more carefully before
  committing to an architecture.

- The determinism testing in Part D surfaced something I didn't expect:
  logprob values aren't perfectly reproducible even with fixed seeds.
  The text outputs are identical, but floating-point non-associativity
  causes deltas up to 6e-4 in logprob values. Small, but worth knowing.

- Semantic few-shot selection (TF-IDF nearest neighbors) with k=5
  outperformed k=10 by 3 pp. This was unexpected — I assumed more
  context would help. But the token-by-token scoring introduces more
  noise with longer prompts, so fewer well-chosen examples won out.

- The interaction between levers matters. Under the broken scorer,
  adding few-shot *hurt*. Under the fixed scorer, it helped. You can't
  evaluate one lever honestly without getting the layers below it right.
