"""Evaluation runner — runs benchmarks via lm-evaluation-harness with
the custom Ollama model wrapper and SQLite caching.

Integrates with Part A: uses OllamaServer to ensure the serving layer
is healthy before running evaluations.

Usage:
    # Run all three benchmarks (HellaSwag, MMLU, custom)
    python eval_runner/run_eval.py

    # Run a single benchmark
    python eval_runner/run_eval.py --tasks hellaswag

    # Quick smoke test (5 samples)
    python eval_runner/run_eval.py --limit 5
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "eval_runner" / "results"
CACHE_DIR = PROJECT_ROOT / "eval_runner" / "cache"
CUSTOM_TASKS_DIR = PROJECT_ROOT / "eval_runner" / "tasks"


def run_evaluation(
    tasks: list[str],
    model_name: str | None = None,
    base_url: str | None = None,
    limit: int | None = None,
    seed: int = 42,
    use_cache: bool = True,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """Run lm-eval benchmarks against the Ollama endpoint."""

    from serve.serve import OllamaServer, DEFAULT_MODEL, DEFAULT_HOST, DEFAULT_PORT
    import eval_runner.model  # noqa: F401 — registers the "ollama" model
    from lm_eval.tasks import TaskManager
    from lm_eval.evaluator import simple_evaluate

    model_name = model_name or DEFAULT_MODEL
    base_url = base_url or f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Ensure the serving layer from Part A is healthy
    server = OllamaServer(model=model_name)
    if not server.is_healthy():
        log.info("Ollama not running — starting via serve/serve.py")
        server.start()
    else:
        log.info("Ollama server already healthy at %s", base_url)

    log.info("Initializing Ollama eval model (model=%s)", model_name)
    from eval_runner.model import OllamaEvalModel

    lm = OllamaEvalModel(
        base_url=base_url,
        model=model_name,
        seed=seed,
    )

    cache_path = None
    if use_cache:
        cache_path = str(CACHE_DIR / f"{model_name.replace(':', '_')}")
        log.info("Caching enabled → %s", cache_path)

    task_manager = TaskManager(include_path=str(CUSTOM_TASKS_DIR))

    log.info("Running tasks: %s (limit=%s)", tasks, limit)
    t0 = time.time()

    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        task_manager=task_manager,
        use_cache=cache_path,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
    )

    elapsed = time.time() - t0
    log.info("Evaluation complete in %.1fs", elapsed)

    # Save raw results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results["results"], f, indent=2, default=str)
    log.info("Raw results → %s", results_file)

    # Rebuild summary from ALL results files so it's cumulative
    summary = rebuild_summary(output_dir)
    print(f"\n{summary}")

    return results["results"]


# MMLU group-level keys — show these but not the 57 individual subtasks
_MMLU_GROUPS = {"mmlu", "mmlu_humanities", "mmlu_stem", "mmlu_social_sciences", "mmlu_other"}


def _is_subtask(key: str) -> bool:
    """True for mmlu_anatomy, mmlu_formal_logic, etc. — skip in summary."""
    return key.startswith("mmlu_") and key not in _MMLU_GROUPS


def rebuild_summary(results_dir: Path = RESULTS_DIR) -> str:
    """Read every results_*.json in results_dir, merge, and write summary.txt.

    For each task key, the latest file (by filename timestamp) wins.
    MMLU subtasks are omitted from the summary — only the aggregate
    and four group rows are shown.
    """
    merged: dict[str, dict] = {}
    for p in sorted(results_dir.glob("results_*.json")):
        with open(p) as f:
            data = json.load(f)
        merged.update(data)

    lines = [
        "=" * 72,
        "  EVALUATION SUMMARY",
        "  (auto-generated from all results/*.json files)",
        "=" * 72,
        "",
        f"  {'Task':<30} {'Metric':<12} {'Value':>10} {'Stderr':>10}",
        f"  {'-'*30} {'-'*12} {'-'*10} {'-'*10}",
    ]

    shown_any = False
    for task_name, task_results in sorted(merged.items()):
        if _is_subtask(task_name):
            continue
        alias = task_results.get("alias", task_name)
        for metric_key, value in task_results.items():
            if "," not in metric_key:
                continue
            metric_name, filter_name = metric_key.rsplit(",", 1)
            if metric_name.endswith("_stderr") or metric_name == "alias":
                continue
            if not isinstance(value, (int, float)):
                continue
            stderr_key = f"{metric_name}_stderr,{filter_name}"
            stderr = task_results.get(stderr_key)
            stderr_str = f"±{stderr:.4f}" if isinstance(stderr, (int, float)) else ""
            display_name = task_name if task_name in _MMLU_GROUPS else alias.strip().lstrip("- ")
            lines.append(
                f"  {display_name:<30} {metric_name:<12} "
                f"{value:>10.4f} {stderr_str:>10}"
            )
            shown_any = True

    if not shown_any:
        lines.append("  (no results yet — run evaluations first)")

    lines += ["", "=" * 72, ""]

    summary = "\n".join(lines)
    summary_file = results_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(summary)
    log.info("Summary table → %s", summary_file)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluations")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["hellaswag", "mmlu", "code_output"],
        help="Tasks to evaluate",
    )
    parser.add_argument("--model", default=None, help="Defaults to serve.DEFAULT_MODEL")
    parser.add_argument("--base-url", default=None, help="Defaults to serve.DEFAULT_HOST:PORT")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        tasks=args.tasks,
        model_name=args.model,
        base_url=args.base_url,
        limit=args.limit,
        seed=args.seed,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
