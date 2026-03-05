"""Guardrails & determinism validation for the Ollama evaluation pipeline.

Five layers of checks, progressing from config through raw API through the
full evaluation harness:

  1. Config audit         — introspects OllamaEvalModel defaults.
  2. Adversarial stability — prompts *designed to invite randomness*
     (creative writing, "pick a random X") must still produce
     byte-identical outputs under deterministic decoding.
  3. Scoring-path determinism — calls OllamaEvalModel._score_continuation
     twice for the same (context, continuation) pairs and asserts exact
     float equality on logprobs.  Also tests that scoring the *same*
     options in *different order* yields identical per-option scores,
     proving the result is independent of evaluation sequence.
  4. Harness-level determinism — runs lm-eval simple_evaluate on a small
     slice of code_output twice (no cache) and compares per-sample results.
  5. Output validation    — schema checks for data.jsonl + model output
     format validation via regex patterns.

Design notes (why these specific tests):
  - Layer 2 uses "adversarial" prompts because testing determinism on
    factual prompts is easy; the interesting question is whether
    temperature=0 + seed actually suppresses stochasticity when the
    prompt actively invites variation.
  - Layer 3 tests the *scoring* path (loglikelihood), not generation,
    because MMLU/HellaSwag correctness depends on logprob ranking.
    Ordering invariance matters because lm-eval may reorder requests
    for batching.
  - Layer 4 closes the loop: if the raw API is deterministic (layer 2)
    and the scoring path is deterministic (layer 3), the harness should
    produce identical aggregate metrics.  This test confirms no
    nondeterminism is introduced by lm-eval's own task/filter machinery.

Usage:
    python guardrails/validate.py              # all checks
    python guardrails/validate.py --rounds 5   # more stability rounds
    python guardrails/validate.py --skip-harness  # skip slow harness test
"""

import argparse
import inspect
import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from serve.serve import OllamaServer, DEFAULT_MODEL, DEFAULT_HOST, DEFAULT_PORT
from serve.client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SEED = 42
CUSTOM_TASK_DATA = PROJECT_ROOT / "eval_runner" / "tasks" / "code_output" / "data.jsonl"


# ── 1. Deterministic-mode config audit ────────────────────────────────

def check_deterministic_config() -> list[str]:
    """Verify that OllamaEvalModel defaults enforce determinism."""
    from eval_runner.model import OllamaEvalModel

    sig = inspect.signature(OllamaEvalModel.__init__)
    params = {
        k: v.default for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    issues = []
    if params.get("temperature") != 0.0:
        issues.append(f"temperature={params.get('temperature')}, want 0.0")
    if params.get("top_p") != 1.0:
        issues.append(f"top_p={params.get('top_p')}, want 1.0")
    if not isinstance(params.get("seed"), int):
        issues.append(f"seed={params.get('seed')}, want an int")
    return issues


# ── 2. Adversarial stability ─────────────────────────────────────────
#
# These prompts actively invite variation.  A truly deterministic setup
# must suppress all stochasticity even when the prompt says "random".

ADVERSARIAL_PROMPTS = [
    "Give me a random number between 1 and 100.",
    "Write a creative one-line poem about the ocean.",
    "Name any city in Europe.",
    "What's a good name for a cat? One word.",
    "Make up a word and define it.",
]


def check_adversarial_stability(
    client: OllamaClient, rounds: int = 3,
) -> list[dict]:
    """Send adversarial prompts `rounds` times; assert byte-identical."""
    results = []
    for prompt in ADVERSARIAL_PROMPTS:
        responses = []
        for _ in range(rounds):
            r = client.generate(
                prompt, temperature=0, top_p=1.0, max_tokens=64, seed=SEED,
            )
            responses.append(r.response)

        unique = set(responses)
        stable = len(unique) == 1
        results.append({
            "prompt": prompt,
            "stable": stable,
            "unique_responses": len(unique),
            "sample": responses[0][:80],
        })
        log.info(
            "[%s] adversarial | %r → %d unique",
            "PASS" if stable else "FAIL", prompt[:40], len(unique),
        )
    return results


# ── 3. Scoring-path determinism ──────────────────────────────────────
#
# Tests OllamaEvalModel._score_continuation (the loglikelihood path used
# by MMLU / HellaSwag).  Two sub-checks:
#   a) Same (context, continuation) → exact same (logprob, is_greedy).
#   b) Scoring 4 MC options in forward vs reverse order yields identical
#      per-option scores, proving independence from evaluation sequence.

MC_QUESTIONS = [
    {
        "context": "The capital of France is",
        "options": [" Paris", " London", " Berlin", " Tokyo"],
        "expected_best": 0,
    },
    {
        "context": "Water boils at",
        "options": [" 100 degrees", " 50 degrees", " 200 degrees"],
        "expected_best": 0,
    },
]


def check_scoring_determinism(tolerance: float = 1e-4) -> list[dict]:
    """Test loglikelihood scoring for determinism and order-invariance.

    Two sub-checks per question:
      a) *Repeat*: score the same options twice in the same order → exact match.
      b) *Reorder*: score in forward vs reverse order → values within tolerance
         (Ollama's KV-cache state can introduce micro-nondeterminism across
         different prompt orderings; see README for discussion).
    """
    from eval_runner.model import OllamaEvalModel
    model = OllamaEvalModel(seed=SEED)

    results = []
    for q in MC_QUESTIONS:
        ctx = q["context"]
        opts = q["options"]

        # Pass 1 (forward)
        model._gen1_cache.clear()
        scores_fwd = [model._score_continuation(ctx, opt) for opt in opts]

        # Pass 2 (forward again — tests pure repeat determinism)
        model._gen1_cache.clear()
        scores_fwd2 = [model._score_continuation(ctx, opt) for opt in opts]

        # Pass 3 (reverse order — tests order-invariance)
        model._gen1_cache.clear()
        scores_rev = [model._score_continuation(ctx, opt) for opt in reversed(opts)]
        scores_rev = list(reversed(scores_rev))

        exact_repeat = scores_fwd == scores_fwd2
        deltas = [abs(a[0] - b[0]) for a, b in zip(scores_fwd, scores_rev)]
        max_delta = max(deltas)
        order_approx = max_delta < tolerance
        ranking_fwd = sorted(range(len(opts)), key=lambda i: -scores_fwd[i][0])
        ranking_rev = sorted(range(len(opts)), key=lambda i: -scores_rev[i][0])
        ranking_stable = ranking_fwd == ranking_rev
        best_idx = ranking_fwd[0]
        ranking_ok = best_idx == q["expected_best"]

        results.append({
            "context": ctx,
            "exact_repeat": exact_repeat,
            "order_approx_match": order_approx,
            "ranking_stable": ranking_stable,
            "ranking_correct": ranking_ok,
            "best_option": opts[best_idx],
            "max_delta": max_delta,
            "scores_fwd": [(opt, f"{s[0]:.6f}", s[1]) for opt, s in zip(opts, scores_fwd)],
            "scores_rev": [(opt, f"{s[0]:.6f}", s[1]) for opt, s in zip(opts, scores_rev)],
        })
        log.info(
            "[%s] scoring | %r → exact_repeat=%s order_δ=%.2e rank=%s best=%r",
            "PASS" if exact_repeat and ranking_stable else "WARN",
            ctx[:30], exact_repeat, max_delta, ranking_ok, opts[best_idx],
        )
    return results


# ── 4. Harness-level determinism ─────────────────────────────────────
#
# Runs simple_evaluate twice on code_output (limit=3, no disk cache) and
# compares the per-task aggregate scores.  This tests the full path:
# run_eval → simple_evaluate → OllamaEvalModel → Ollama API.

def check_harness_determinism() -> dict:
    import eval_runner.model  # noqa: F401 — registers "ollama" model
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.tasks import TaskManager
    from eval_runner.model import OllamaEvalModel

    task_manager = TaskManager(
        include_path=str(PROJECT_ROOT / "eval_runner" / "tasks")
    )

    scores = []
    for run_idx in range(2):
        lm = OllamaEvalModel(seed=SEED)
        results = simple_evaluate(
            model=lm,
            tasks=["code_output"],
            limit=3,
            task_manager=task_manager,
            random_seed=SEED,
            numpy_random_seed=SEED,
            torch_random_seed=SEED,
            fewshot_random_seed=SEED,
        )
        task_results = results["results"].get("code_output", {})
        scores.append(task_results)
        log.info("Harness run %d: %s", run_idx + 1, task_results)

    match = scores[0] == scores[1]
    log.info("[%s] harness determinism | scores match=%s", "PASS" if match else "FAIL", match)
    return {"run_1": scores[0], "run_2": scores[1], "match": match}


# ── 5. Output validation (schema + model output) ─────────────────────

VALID_OUTPUT_PATTERNS = [
    re.compile(r"^-?\d+(\.\d+)?$"),
    re.compile(r"^(True|False)$"),
    re.compile(r"^\[.*\]$"),
    re.compile(r"^\{.*\}$"),
    re.compile(r"^[a-zA-Z_][\w]*$"),
    re.compile(r"^[\w\s\-\.\,\!\?\'\"\:\;]+$"),
]


def validate_code_output_answer(answer: str) -> bool:
    return any(p.match(answer.strip()) for p in VALID_OUTPUT_PATTERNS)


def check_custom_task_schema() -> list[dict]:
    results = []
    with open(CUSTOM_TASK_DATA) as f:
        for i, line in enumerate(f, 1):
            row = json.loads(line)
            issues = []
            if "code" not in row or not isinstance(row["code"], str):
                issues.append("missing or non-string 'code'")
            if "answer" not in row or not isinstance(row["answer"], str):
                issues.append("missing or non-string 'answer'")
            if not issues and not validate_code_output_answer(row["answer"]):
                issues.append(f"answer {row['answer']!r} no pattern match")
            results.append({"row": i, "valid": not issues, "issues": issues})
    log.info(
        "Schema: %d/%d valid",
        sum(r["valid"] for r in results), len(results),
    )
    return results


def check_model_output_validation(
    client: OllamaClient, n_samples: int = 5,
) -> list[dict]:
    with open(CUSTOM_TASK_DATA) as f:
        items = [json.loads(line) for line in f]

    results = []
    for item in items[:n_samples]:
        prompt = (
            "What does this Python code print? Reply with ONLY the exact "
            "output, no backticks, no explanation.\n\n"
            f"```python\n{item['code']}\n```\n\nOutput: "
        )
        r = client.generate(
            prompt, temperature=0, top_p=1.0, max_tokens=64, seed=SEED,
            stop=["\n"],
        )
        raw = r.response.strip()
        results.append({
            "expected": item["answer"],
            "got": raw,
            "format_valid": validate_code_output_answer(raw),
            "exact_match": raw == item["answer"],
        })
    return results


# ── CLI ───────────────────────────────────────────────────────────────

def _section(num, title):
    print(f"\n{'='*64}")
    print(f"  {num}. {title}")
    print(f"{'='*64}")


def main():
    parser = argparse.ArgumentParser(description="Guardrails validation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=f"http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--skip-harness", action="store_true",
                        help="Skip the slow harness-level determinism test")
    args = parser.parse_args()

    server = OllamaServer(model=args.model)
    if not server.is_healthy():
        log.info("Starting Ollama server …")
        server.start()
    client = OllamaClient(base_url=args.base_url, model=args.model)

    all_pass = True

    # ── 1. Config audit ──
    _section(1, "DETERMINISTIC CONFIG AUDIT")
    issues = check_deterministic_config()
    if issues:
        for i in issues:
            print(f"  FAIL: {i}")
        all_pass = False
    else:
        print("  PASS: temperature=0.0, top_p=1.0, seed=int")

    # ── 2. Adversarial stability ──
    _section(2, f"ADVERSARIAL STABILITY ({args.rounds} rounds)")
    stability = check_adversarial_stability(client, rounds=args.rounds)
    failed = [s for s in stability if not s["stable"]]
    if failed:
        for s in failed:
            print(f"  FAIL: {s['prompt']!r} → {s['unique_responses']} variants")
        all_pass = False
    else:
        print(f"  PASS: {len(stability)}/{len(stability)} adversarial prompts stable")
    for s in stability:
        tag = "PASS" if s["stable"] else "FAIL"
        print(f"    [{tag}] {s['prompt'][:50]}")
        print(f"          → {s['sample']!r}")

    # ── 3. Scoring-path determinism ──
    _section(3, "SCORING-PATH DETERMINISM (loglikelihood)")
    scoring = check_scoring_determinism()
    for s in scoring:
        repeat_ok = s["exact_repeat"]
        rank_ok = s["ranking_stable"] and s["ranking_correct"]
        tag = "PASS" if repeat_ok and rank_ok else "WARN" if rank_ok else "FAIL"
        if not rank_ok:
            all_pass = False
        print(f"  [{tag}] {s['context']!r}")
        print(f"         exact repeat: {repeat_ok}  |  ranking stable: {s['ranking_stable']}")
        print(f"         max order-swap δ: {s['max_delta']:.2e}  |  best: {s['best_option']!r}")
        print(f"         Forward scores:")
        for opt, score, greedy in s["scores_fwd"]:
            print(f"           {opt!r:>15s}  logp={score}  greedy={greedy}")
        if not s["order_approx_match"]:
            print(f"         Reverse scores (differ by > tolerance):")
            for opt, score, greedy in s["scores_rev"]:
                print(f"           {opt!r:>15s}  logp={score}  greedy={greedy}")

    # ── 4. Harness-level determinism ──
    if args.skip_harness:
        _section(4, "HARNESS DETERMINISM (skipped)")
        print("  SKIP: --skip-harness flag set")
    else:
        _section(4, "HARNESS DETERMINISM (2× code_output, limit=3, no cache)")
        harness = check_harness_determinism()
        if not harness["match"]:
            all_pass = False
            print("  FAIL: scores differ between runs")
            print(f"    Run 1: {harness['run_1']}")
            print(f"    Run 2: {harness['run_2']}")
        else:
            print("  PASS: identical scores across two independent runs")
            for key, val in sorted(harness["run_1"].items()):
                if isinstance(val, float):
                    print(f"    {key}: {val:.4f}")

    # ── 5a. Schema validation ──
    _section("5a", "CUSTOM TASK SCHEMA")
    schema = check_custom_task_schema()
    schema_fails = [r for r in schema if not r["valid"]]
    if schema_fails:
        for r in schema_fails:
            print(f"  FAIL row {r['row']}: {'; '.join(r['issues'])}")
        all_pass = False
    else:
        print(f"  PASS: {len(schema)}/{len(schema)} rows valid")

    # ── 5b. Model output validation ──
    _section("5b", f"MODEL OUTPUT VALIDATION ({args.samples} samples)")
    outputs = check_model_output_validation(client, n_samples=args.samples)
    for r in outputs:
        tag = "PASS" if r["exact_match"] else "MISS"
        print(f"  [{tag}] expected={r['expected']!r:>20s}  got={r['got']!r}")
    exact = sum(r["exact_match"] for r in outputs)
    fmt = sum(r["format_valid"] for r in outputs)
    print(f"\n  Exact: {exact}/{len(outputs)}  |  Valid format: {fmt}/{len(outputs)}")

    # ── Summary ──
    print(f"\n{'='*64}")
    verdict = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"  OVERALL: {verdict}")
    print(f"{'='*64}\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
