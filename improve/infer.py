"""Run the final optimized HellaSwag evaluation and collect per-sample data.

Uses the best configuration found by optimize_prompt.py to run a full
evaluation, saving:
  - Per-sample predictions (question, options, predicted, correct, logprobs)
  - Aggregate accuracy with bootstrap 95% CI
  - McNemar test comparing baseline vs improved (statistical significance)
  - Before/after example pairs for the report

This script does NOT modify OllamaEvalModel or the serving layer — it
only varies evaluation-time parameters (few-shot, prompt template).

Usage:
    python improve/infer.py --config baseline --limit 200
    python improve/infer.py --config fewshot_semantic_10 --limit 200
    python improve/infer.py --compare baseline fewshot_semantic_10 --limit 200
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from serve.serve import OllamaServer, DEFAULT_MODEL, DEFAULT_HOST, DEFAULT_PORT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TASKS_DIR = Path(__file__).resolve().parent / "tasks"

SEED = 42


def run_with_samples(
    task_name: str,
    num_fewshot: int,
    limit: int | None,
    use_custom_tasks: bool = False,
    scoring_mode: str = "soft_floor",
) -> dict:
    """Run evaluation and capture per-sample log_likelihoods + predictions."""
    import eval_runner.model  # noqa: F401
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.tasks import TaskManager
    from eval_runner.model import OllamaEvalModel

    include_paths = [str(PROJECT_ROOT / "eval_runner" / "tasks")]
    if use_custom_tasks:
        include_paths.append(str(TASKS_DIR))

    task_manager = TaskManager(include_path=include_paths)
    lm = OllamaEvalModel(seed=SEED, scoring_mode=scoring_mode)

    t0 = time.time()
    results = simple_evaluate(
        model=lm,
        tasks=[task_name],
        num_fewshot=num_fewshot,
        limit=limit,
        task_manager=task_manager,
        random_seed=SEED,
        numpy_random_seed=SEED,
        torch_random_seed=SEED,
        fewshot_random_seed=SEED,
        log_samples=True,
    )
    elapsed = time.time() - t0

    task_results = results["results"].get(task_name, {})
    samples = results.get("samples", {}).get(task_name, [])

    per_sample = []
    for s in samples:
        doc = s.get("doc", {})
        pred_acc = None
        pred_norm = None
        logprobs = []
        correct_acc = None
        correct_norm = None

        resps = s.get("filtered_resps", s.get("resps", []))
        if resps:
            logprobs = [r[0] if isinstance(r, (list, tuple)) else r for r in resps]
            pred_acc = int(np.argmax(logprobs))

            # acc_norm: normalize by byte length of each choice
            choices = doc.get("choices", [])
            if choices and len(choices) == len(logprobs):
                byte_lens = [max(len(c.encode("utf-8")), 1) for c in choices]
                norm_logprobs = [lp / bl for lp, bl in zip(logprobs, byte_lens)]
                pred_norm = int(np.argmax(norm_logprobs))
            else:
                norm_logprobs = logprobs
                pred_norm = pred_acc

            gold = doc.get("gold", None)
            if gold is not None:
                correct_acc = pred_acc == int(gold)
                correct_norm = pred_norm == int(gold)

        per_sample.append({
            "idx": s.get("doc_id", len(per_sample)),
            "query": doc.get("query", "")[:200],
            "choices": doc.get("choices", []),
            "gold": doc.get("gold"),
            "pred_acc": pred_acc,
            "pred_norm": pred_norm,
            "correct_acc": correct_acc,
            "correct_norm": correct_norm,
            "logprobs": [round(lp, 4) if isinstance(lp, float) else lp for lp in logprobs],
            "activity": doc.get("activity_label", ""),
        })

    return {
        "task_results": task_results,
        "per_sample": per_sample,
        "elapsed_sec": round(elapsed, 1),
        "limit": limit,
        "num_fewshot": num_fewshot,
        "task_name": task_name,
    }


def bootstrap_ci(correct_flags: list[bool], n_boot: int = 10_000, alpha: float = 0.05) -> dict:
    """Compute bootstrap 95% CI for accuracy."""
    rng = np.random.RandomState(SEED)
    arr = np.array(correct_flags, dtype=float)
    n = len(arr)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(sample.mean())
    means = np.sort(means)
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return {"mean": float(arr.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def mcnemar_test(baseline_correct: list[bool], improved_correct: list[bool]) -> dict:
    """McNemar's test for paired nominal data."""
    assert len(baseline_correct) == len(improved_correct)
    b = sum(1 for a, b in zip(baseline_correct, improved_correct) if a and not b)
    c = sum(1 for a, b in zip(baseline_correct, improved_correct) if not a and b)
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0, "significant": False}
    chi2 = (b - c) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    return {
        "b_right_to_wrong": b,
        "c_wrong_to_right": c,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
    }


def find_flipped_examples(baseline_samples, improved_samples, n=12):
    """Find examples where baseline got wrong but improved got right (acc_norm)."""
    flipped = []
    for bs, imp in zip(baseline_samples, improved_samples):
        if bs["correct_norm"] is False and imp["correct_norm"] is True:
            flipped.append({
                "idx": bs["idx"],
                "query": bs["query"],
                "choices": bs["choices"],
                "gold": bs["gold"],
                "baseline_pred": bs["pred_norm"],
                "improved_pred": imp["pred_norm"],
                "baseline_logprobs": bs["logprobs"],
                "improved_logprobs": imp["logprobs"],
                "activity": bs.get("activity", ""),
            })
    return flipped[:n]


def main():
    parser = argparse.ArgumentParser(description="Run optimized HellaSwag inference")
    parser.add_argument("--config", type=str, help="Single config name to run")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "IMPROVED"),
                        help="Compare two configs")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=f"http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    parser.add_argument("--scoring-mode", choices=["soft_floor", "hard_floor"],
                        default="soft_floor",
                        help="Scoring strategy: soft_floor (Part E) or hard_floor (Part B)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    server = OllamaServer(model=args.model)
    if not server.is_healthy():
        log.info("Starting Ollama …")
        server.start()

    from improve.optimize_prompt import CONFIGS

    if args.config:
        cfg = CONFIGS[args.config]
        task_name = cfg.get("task", f"hellaswag_{args.config}")
        is_custom = cfg.get("custom", False)

        if is_custom and not (TASKS_DIR / args.config).exists():
            log.error("Custom task '%s' not built yet. Run optimize_prompt.py first.", args.config)
            sys.exit(1)

        log.info("Running config '%s' with limit=%d scoring=%s", args.config, args.limit, args.scoring_mode)
        result = run_with_samples(
            task_name=task_name,
            num_fewshot=cfg["num_fewshot"],
            limit=args.limit if not is_custom else None,
            use_custom_tasks=is_custom,
            scoring_mode=args.scoring_mode,
        )

        correct_flags = [s["correct_norm"] for s in result["per_sample"] if s["correct_norm"] is not None]
        ci = bootstrap_ci(correct_flags)
        result["bootstrap_ci"] = ci

        out_path = RESULTS_DIR / f"infer_{args.config}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info("Results → %s", out_path)
        log.info("acc=%.4f  95%% CI=[%.4f, %.4f]  n=%d  time=%.0fs",
                 ci["mean"], ci["ci_lo"], ci["ci_hi"], len(correct_flags), result["elapsed_sec"])

    elif args.compare:
        baseline_name, improved_name = args.compare
        results = {}

        for name in [baseline_name, improved_name]:
            cfg = CONFIGS[name]
            task_name = cfg.get("task", f"hellaswag_{name}")
            is_custom = cfg.get("custom", False)

            if is_custom and not (TASKS_DIR / name).exists():
                log.info("Building custom task '%s' …", name)
                from improve.optimize_prompt import (
                    build_custom_dataset, write_dataset_and_yaml,
                )
                with open(DATA_DIR / "train.json") as f:
                    train_data = json.load(f)
                with open(DATA_DIR / "val.json") as f:
                    val_data = json.load(f)
                with open(DATA_DIR / "fewshot_map.json") as f:
                    fewshot_map = json.load(f)
                build = cfg["build"]
                rows = build_custom_dataset(
                    val_data, train_data,
                    fewshot_map if build["semantic"] else None,
                    build["n_shots"], build["template"],
                    limit=args.limit,
                )
                task_name = write_dataset_and_yaml(name, rows)

            log.info("Running '%s' (scoring=%s) …", name, args.scoring_mode)
            result = run_with_samples(
                task_name=task_name,
                num_fewshot=cfg["num_fewshot"],
                limit=args.limit if not is_custom else None,
                use_custom_tasks=is_custom,
                scoring_mode=args.scoring_mode,
            )
            results[name] = result

        # Compute statistics
        base_correct = [s["correct_norm"] for s in results[baseline_name]["per_sample"]
                        if s["correct_norm"] is not None]
        imp_correct = [s["correct_norm"] for s in results[improved_name]["per_sample"]
                       if s["correct_norm"] is not None]

        n = min(len(base_correct), len(imp_correct))
        base_correct = base_correct[:n]
        imp_correct = imp_correct[:n]

        base_ci = bootstrap_ci(base_correct)
        imp_ci = bootstrap_ci(imp_correct)
        mcnemar = mcnemar_test(base_correct, imp_correct)

        flipped = find_flipped_examples(
            results[baseline_name]["per_sample"],
            results[improved_name]["per_sample"],
        )

        comparison = {
            "baseline": {
                "config": baseline_name,
                "acc": base_ci["mean"],
                "ci_95": [base_ci["ci_lo"], base_ci["ci_hi"]],
                "n": n,
                "elapsed_sec": results[baseline_name]["elapsed_sec"],
            },
            "improved": {
                "config": improved_name,
                "acc": imp_ci["mean"],
                "ci_95": [imp_ci["ci_lo"], imp_ci["ci_hi"]],
                "n": n,
                "elapsed_sec": results[improved_name]["elapsed_sec"],
            },
            "lift": round(imp_ci["mean"] - base_ci["mean"], 4),
            "lift_pct": round((imp_ci["mean"] - base_ci["mean"]) * 100, 2),
            "mcnemar": mcnemar,
            "flipped_examples": flipped,
        }

        out_path = RESULTS_DIR / f"compare_{baseline_name}_vs_{improved_name}.json"
        with open(out_path, "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        # Save per-sample data too
        for name, result in results.items():
            sample_path = RESULTS_DIR / f"infer_{name}.json"
            result["bootstrap_ci"] = bootstrap_ci(
                [s["correct_norm"] for s in result["per_sample"] if s["correct_norm"] is not None]
            )
            with open(sample_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"  COMPARISON: {baseline_name} vs {improved_name}")
        print(f"{'='*60}")
        print(f"  Baseline  acc: {base_ci['mean']:.4f}  95% CI [{base_ci['ci_lo']:.4f}, {base_ci['ci_hi']:.4f}]")
        print(f"  Improved  acc: {imp_ci['mean']:.4f}  95% CI [{imp_ci['ci_lo']:.4f}, {imp_ci['ci_hi']:.4f}]")
        print(f"  Lift: {comparison['lift_pct']:+.2f} percentage points")
        print(f"  McNemar: χ²={mcnemar['chi2']:.2f}  p={mcnemar['p_value']:.4f}  {'SIGNIFICANT' if mcnemar['significant'] else 'not significant'}")
        print(f"  Flipped (wrong→right): {mcnemar['c_wrong_to_right']}  (right→wrong): {mcnemar['b_right_to_wrong']}")
        print(f"  Saved: {out_path}")
        print(f"{'='*60}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
