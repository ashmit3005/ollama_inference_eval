# Guardrails & Determinism

## Quick start

```bash
python guardrails/validate.py              # all 5 layers
python guardrails/validate.py --rounds 5   # more stability rounds
python guardrails/validate.py --skip-harness  # skip slow harness-level test
```

## Validation layers

The guardrails are organized as **five progressive layers**, each testing a
deeper slice of the pipeline.  This structure was chosen deliberately: if a
shallow layer fails we can localize the issue quickly, and if all layers pass
we have high confidence in end-to-end determinism.

### 1. Deterministic config audit

Introspects `OllamaEvalModel.__init__` defaults via `inspect.signature` and
asserts:

| Parameter     | Required | Rationale |
|---------------|----------|-----------|
| `temperature` | `0.0`    | Greedy decoding — no sampling randomness |
| `top_p`       | `1.0`    | Full vocabulary (no nucleus cutoff) |
| `seed`        | any `int`| Ollama's internal RNG is seeded |

**Scope**: eval harness wrapper (`eval_runner/model.py`).

### 2. Adversarial stability

Sends 5 prompts **designed to invite randomness** — "give me a random number,"
"make up a word," "write a creative poem" — three rounds each, all with
`seed=42, temperature=0, top_p=1`.  Asserts byte-identical outputs across
rounds.

**Design rationale**: Testing determinism on factual prompts ("capital of
France") is trivially easy — the model's probability mass is concentrated on
one answer regardless of decoding parameters.  The interesting question is
whether `temperature=0 + seed` actually suppresses stochasticity when the
*prompt itself actively invites variation*.  These adversarial prompts create a
high-entropy output distribution that only collapses to a single answer under
correct deterministic decoding.

**Scope**: raw Ollama API via `OllamaClient.generate()`.

### 3. Scoring-path determinism

Tests `OllamaEvalModel._score_continuation` — the loglikelihood path that
MMLU and HellaSwag rely on for multiple-choice scoring.

Two sub-checks per question:

- **Exact repeat**: score the same (context, continuation) pairs twice in the
  same order, cache cleared between passes → asserts identical floats.
- **Order invariance**: score options in forward then reverse order → asserts
  rankings are identical and logprob values are within tolerance.

**Design rationale**: MMLU/HellaSwag correctness depends on *ranking* of option
logprobs, not their absolute values.  `lm-eval` may reorder requests for
batching, so order-invariance is a practical requirement.  We test both exact
repeatability (same order) and cross-order consistency (different order).

**Empirical finding**: Ollama exhibits micro-nondeterminism (δ ≤ 6e-4) in
logprob values even with identical seed/temperature/prompt, likely due to
floating-point non-associativity in the underlying inference engine's
KV-cache computations.  Despite this, *rankings are perfectly stable*, meaning
benchmark scores are deterministic.  See "Where nondeterminism persists" below.

**Scope**: `OllamaEvalModel._score_continuation` → Ollama `/api/generate`.

### 4. Harness-level determinism

Runs `lm_eval.evaluator.simple_evaluate` twice on `code_output` (limit=3, no
disk cache) with identical random seeds, then compares the per-task result
dictionaries for exact equality.

**Design rationale**: Layers 2 and 3 test necessary conditions (raw API is
deterministic, scoring path is deterministic).  This layer tests the *sufficient*
condition — that the full `lm-eval` pipeline (task loading, prompt
construction, filtering, metric computation) also produces identical results.
Without this, there could be hidden nondeterminism in `lm-eval`'s own
shuffling, batching, or postprocessing.

**Scope**: full path — `simple_evaluate` → `OllamaEvalModel` → Ollama API.

### 5. Output validation

**5a. Schema validation**: Every row in `data.jsonl` is checked for required
fields (`code: str`, `answer: str`) and answer format (regex patterns covering
numeric, boolean, list/dict repr, single-word, printable string).

**5b. Model output validation**: Runs the model on 5 `code_output` prompts
using the same prompt template and decoding settings as the harness.  Checks
format validity (regex) and exact match against ground truth.

**Scope**: data quality + raw API generation.

## Where nondeterminism persists

Even with `temperature=0`, `top_p=1`, and a fixed `seed`, these sources of
nondeterminism remain:

1. **Logprob micro-nondeterminism**.  Our scoring-path test (layer 3)
   empirically measured logprob deltas up to 6.35e-4 between identical calls.
   This is caused by floating-point non-associativity in GPU matrix
   multiplications and the KV-cache's internal state.  The effect is below the
   threshold needed to flip rankings on any benchmark question we tested, so
   evaluation *scores* remain deterministic even when exact logprob *values*
   are not bitwise reproducible.

2. **Cross-hardware divergence**.  GPU matrix multiplications are not
   bitwise-reproducible across different GPU architectures (or CPU vs GPU).
   Outputs for the same seed may differ on different machines.  Within a single
   machine, outputs are stable (confirmed by layer 2).

3. **Ollama version drift**.  Tokenizer, quantization, and sampling internals
   change between Ollama releases.  We tested on v0.16.2; results may differ on
   other versions.

4. **Model quantization artifacts**.  `llama3:8b` uses Q4_0 quantization.
   While deterministic for a given binary, it introduces approximation that
   may differ from full precision or other quantization levels.

5. **Concurrent request ordering**.  When concurrency > 1, request scheduling
   affects KV-cache eviction patterns and can alter internal state between
   requests.  Individual request outputs are deterministic (given their inputs),
   but timing-dependent cache behavior means the *order* of concurrent requests
   can indirectly affect logprob values (contributing to finding #1).

6. **Context-length truncation**.  Prompts exceeding the model's context window
   are silently truncated.  The truncation boundary depends on the tokenizer
   version and can cause different behavior for near-limit prompts.

## What we verified is stable

| Property | Evidence |
|----------|----------|
| Generation output (byte-identical) | Layer 2: 5/5 adversarial prompts, 3 rounds each |
| Logprob ranking (option ordering) | Layer 3: 2/2 MC questions, fwd vs rev |
| Harness aggregate scores | Layer 4: two independent `simple_evaluate` runs match |
| Custom task data quality | Layer 5a: 30/30 rows pass schema |
| Model output format | Layer 5b: 5/5 samples pass format + exact match |

## Integration with Parts A–E

- **Part A (serve/)**: `validate.py` imports `OllamaServer` and `OllamaClient`
  from `serve/`, using the same shared config constants (`DEFAULT_HOST`,
  `DEFAULT_PORT`, `DEFAULT_MODEL`).  If the server isn't running, it starts it.

- **Part B (eval_runner/)**: Layer 1 introspects `OllamaEvalModel` directly.
  Layer 3 calls its `_score_continuation` method.  Layer 4 runs `simple_evaluate`
  with the same task configuration used by `run_eval.py`.

- **Part C (perf/)**: `perf/load_test.py` uses the same deterministic settings
  (`temperature=0, top_p=1.0, seed=42`) as the eval pipeline, matching the
  config verified by layer 1.  The micro-nondeterminism in logprob values
  (finding #1) is consistent with the flat per-request tok/s observed in the
  performance analysis — the inference engine's behavior is stable even if not
  bitwise identical.

- **Part E (improve/)**: The scoring method in `OllamaEvalModel` was refined
  during Part E: the hard logprob floor (-100) was replaced with a soft
  floor (`min(top-20) - 1.0`) and char-by-char advance on token miss.  This
  does not affect layers 2 or 4 (generation-based), but layer 3 (scoring
  path) now exercises the updated `_score_continuation`.  The
  micro-nondeterminism finding (#1) still applies — the soft floor simply
  changes the magnitude, not the existence, of the variation.
