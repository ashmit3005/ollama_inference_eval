"""Benchmark sequential vs parallel choice scoring on HellaSwag.

Runs a small HellaSwag eval (--limit questions, default 10) in both modes
and reports wall time, accuracy, and speedup factor.

Usage:
    python perf/bench_parallel.py --limit 10
"""

import argparse
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lm_eval
from eval_runner.model import OllamaEvalModel


def run_eval(parallel: bool, limit: int, seed: int = 42) -> dict:
    mode_label = "parallel" if parallel else "sequential"
    print(f"\n{'='*60}")
    print(f"  Running {mode_label} mode  (limit={limit})")
    print(f"{'='*60}\n")

    lm = OllamaEvalModel(
        seed=seed,
        scoring_mode="soft_floor",
        parallel_choices=parallel,
    )

    t0 = time.time()
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["hellaswag"],
        num_fewshot=0,
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
    )
    elapsed = time.time() - t0

    task_results = results["results"]["hellaswag"]
    acc = task_results.get("acc,none", 0)
    acc_norm = task_results.get("acc_norm,none", 0)

    return {
        "mode": mode_label,
        "limit": limit,
        "acc": acc,
        "acc_norm": acc_norm,
        "elapsed_sec": round(elapsed, 1),
        "avg_sec_per_question": round(elapsed / limit, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seq_result = run_eval(parallel=False, limit=args.limit, seed=args.seed)

    OllamaEvalModel._gen1_cache = {}

    par_result = run_eval(parallel=True, limit=args.limit, seed=args.seed)

    speedup = seq_result["elapsed_sec"] / par_result["elapsed_sec"] if par_result["elapsed_sec"] > 0 else 0

    print(f"\n{'='*60}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Questions: {args.limit}")
    print(f"  Loglikelihood calls: {args.limit * 4}")
    print()
    print(f"  Sequential:  {seq_result['elapsed_sec']}s  "
          f"({seq_result['avg_sec_per_question']}s/q)  "
          f"acc_norm={seq_result['acc_norm']}")
    print(f"  Parallel:    {par_result['elapsed_sec']}s  "
          f"({par_result['avg_sec_per_question']}s/q)  "
          f"acc_norm={par_result['acc_norm']}")
    print(f"  Speedup:     {speedup:.2f}×")
    print(f"  Accuracy match: {seq_result['acc_norm'] == par_result['acc_norm']}")
    print(f"{'='*60}\n")

    report = {
        "sequential": seq_result,
        "parallel": par_result,
        "speedup": round(speedup, 2),
        "accuracy_match": seq_result["acc_norm"] == par_result["acc_norm"],
    }

    out_path = os.path.join("perf", "bench_parallel_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
