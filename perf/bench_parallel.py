"""Benchmark sequential vs parallel vs Go-proxy scoring on HellaSwag.

Runs a small HellaSwag eval (--limit questions, default 10) in each mode
and reports wall time, accuracy, and speedup factor.

Usage:
    python perf/bench_parallel.py --limit 10
    python perf/bench_parallel.py --limit 10 --modes sequential go
"""

import argparse
import json
import time
import sys
import os
import subprocess
import signal
import requests as req_lib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lm_eval
from eval_runner.model import OllamaEvalModel

SCORER_BIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scorer", "scorer"
)


def start_go_proxy():
    """Start the Go scoring proxy and wait for it to be healthy."""
    if not os.path.exists(SCORER_BIN):
        print("  Building Go scorer...")
        subprocess.run(
            ["go", "build", "-o", "scorer", "."],
            cwd=os.path.dirname(SCORER_BIN),
            check=True,
        )
    proc = subprocess.Popen(
        [SCORER_BIN],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for _ in range(30):
        time.sleep(0.5)
        try:
            r = req_lib.get("http://localhost:9090/health", timeout=2)
            if r.status_code == 200:
                print("  Go proxy is ready (pid=%d)" % proc.pid)
                return proc
        except Exception:
            pass
    proc.kill()
    raise RuntimeError("Go proxy failed to start")


def stop_go_proxy(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)


def run_eval(mode: str, limit: int, seed: int = 42) -> dict:
    print(f"\n{'='*60}")
    print(f"  Running {mode} mode  (limit={limit})")
    print(f"{'='*60}\n")

    parallel_flag = False
    if mode == "parallel":
        parallel_flag = True
    elif mode == "go":
        parallel_flag = "go"

    lm = OllamaEvalModel(
        seed=seed,
        scoring_mode="soft_floor",
        parallel_choices=parallel_flag,
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
        "mode": mode,
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
    parser.add_argument(
        "--modes", nargs="+",
        default=["sequential", "parallel", "go"],
        choices=["sequential", "parallel", "go"],
    )
    args = parser.parse_args()

    all_results = {}
    go_proc = None

    for mode in args.modes:
        if mode == "go" and go_proc is None:
            go_proc = start_go_proxy()

        result = run_eval(mode, args.limit, args.seed)
        all_results[mode] = result

    if go_proc:
        stop_go_proxy(go_proc)

    baseline_time = all_results.get("sequential", {}).get("elapsed_sec", 1)

    print(f"\n{'='*60}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Questions: {args.limit}")
    print(f"  Loglikelihood calls: {args.limit * 4}")
    print()
    for mode, r in all_results.items():
        speedup = baseline_time / r["elapsed_sec"] if r["elapsed_sec"] > 0 else 0
        tag = "" if mode == "sequential" else f"  speedup={speedup:.2f}×"
        print(f"  {mode:12s}: {r['elapsed_sec']:7.1f}s  "
              f"({r['avg_sec_per_question']}s/q)  "
              f"acc_norm={r['acc_norm']}{tag}")

    accs = [r["acc_norm"] for r in all_results.values()]
    print(f"\n  Accuracy match: {len(set(accs)) == 1}")
    print(f"{'='*60}\n")

    report = {
        "modes": all_results,
        "baseline_mode": "sequential",
    }
    if "sequential" in all_results:
        for mode, r in all_results.items():
            if mode != "sequential":
                report[f"speedup_{mode}"] = round(
                    baseline_time / r["elapsed_sec"], 2
                ) if r["elapsed_sec"] > 0 else 0

    out_path = os.path.join("perf", "bench_parallel_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
