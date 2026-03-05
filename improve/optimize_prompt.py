"""Optimize HellaSwag accuracy via inference-time levers.

Tests configurations along three axes, all using the existing
OllamaEvalModel without any model/wrapper changes:

  Axis 1 — Few-shot count:        0, 5, 10
  Axis 2 — Few-shot selection:    random (lm-eval default) vs semantic (TF-IDF)
  Axis 3 — Prompt template:       stock vs instruction-prefixed

For semantic few-shot and template changes, we build custom JSONL datasets
with the few-shot context baked into the query field, then register a
lightweight lm-eval task YAML that reads them.

Results are saved to improve/results/ as JSON files with per-configuration
accuracy, stderr, and timing.

Usage:
    python improve/optimize_prompt.py --limit 100          # exploration
    python improve/optimize_prompt.py --configs baseline    # single config
    python improve/optimize_prompt.py --configs all         # full sweep
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from serve.serve import OllamaServer, DEFAULT_MODEL, DEFAULT_HOST, DEFAULT_PORT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TASKS_DIR = Path(__file__).resolve().parent / "tasks"

SEED = 42

# ── Prompt templates ──────────────────────────────────────────────────

TEMPLATE_STOCK = "{query}"

TEMPLATE_INSTRUCTION = (
    "Choose the most natural continuation for the activity described below.\n\n"
    "{query}"
)


# ── Few-shot formatting ──────────────────────────────────────────────

def format_fewshot_example(example: dict) -> str:
    """Format one training example as a few-shot demonstration."""
    correct = example["choices"][example["gold"]]
    return f"{example['query']}{correct}"


def build_fewshot_prefix(examples: list[dict]) -> str:
    """Build the few-shot prefix from a list of examples."""
    if not examples:
        return ""
    parts = [format_fewshot_example(e) for e in examples]
    return "\n\n".join(parts) + "\n\n"


# ── Dataset builders ─────────────────────────────────────────────────

def build_custom_dataset(
    val_data: list[dict],
    train_data: list[dict],
    fewshot_map: dict[str, list[int]] | None,
    n_shots: int,
    template: str,
    limit: int | None = None,
) -> list[dict]:
    """Build a JSONL-ready dataset with few-shot context baked into query."""
    rng = random.Random(SEED)
    rows = []

    items = val_data[:limit] if limit else val_data
    for i, val_item in enumerate(items):
        # Select few-shot examples
        if n_shots == 0:
            fewshot_examples = []
        elif fewshot_map is not None:
            indices = fewshot_map.get(str(i), [])[:n_shots]
            fewshot_examples = [train_data[idx] for idx in indices]
        else:
            fewshot_examples = rng.sample(train_data, min(n_shots, len(train_data)))

        prefix = build_fewshot_prefix(fewshot_examples)
        query = template.format(query=val_item["query"])
        full_query = prefix + query

        rows.append({
            "query": full_query,
            "choices": val_item["choices"],
            "gold": val_item["gold"],
            "activity_label": val_item.get("activity_label", ""),
            "original_idx": i,
        })

    return rows


def write_dataset_and_yaml(
    config_name: str,
    rows: list[dict],
) -> str:
    """Write JSONL + task YAML for a custom configuration. Returns task name."""
    os.makedirs(TASKS_DIR / config_name, exist_ok=True)

    jsonl_path = TASKS_DIR / config_name / "data.jsonl"
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    task_name = f"hellaswag_{config_name}"
    yaml_content = f"""task: {task_name}
dataset_path: json
dataset_kwargs:
  data_files:
    validation: {jsonl_path}
output_type: multiple_choice
validation_split: validation
doc_to_text: "{{{{query}}}}"
doc_to_choice: "choices"
doc_to_target: "{{{{gold}}}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""
    yaml_path = TASKS_DIR / config_name / f"{task_name}.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    log.info("Config '%s': %d rows → %s", config_name, len(rows), jsonl_path)
    return task_name


# ── Evaluation ────────────────────────────────────────────────────────

def run_config(
    task_name: str,
    num_fewshot: int,
    limit: int | None,
    use_custom_tasks: bool = False,
) -> dict:
    """Run a single evaluation configuration and return results."""
    import eval_runner.model  # noqa: F401
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.tasks import TaskManager
    from eval_runner.model import OllamaEvalModel

    include_paths = [str(PROJECT_ROOT / "eval_runner" / "tasks")]
    if use_custom_tasks:
        include_paths.append(str(TASKS_DIR))

    task_manager = TaskManager(include_path=include_paths)
    lm = OllamaEvalModel(seed=SEED)

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
    )
    elapsed = time.time() - t0

    task_results = results["results"].get(task_name, {})
    task_results["_elapsed_sec"] = round(elapsed, 1)
    task_results["_limit"] = limit
    task_results["_num_fewshot"] = num_fewshot

    return task_results


# ── Configuration definitions ─────────────────────────────────────────

CONFIGS = {
    "baseline": {
        "description": "0-shot, stock template, stock hellaswag task",
        "task": "hellaswag",
        "num_fewshot": 0,
        "custom": False,
    },
    "fewshot_random_5": {
        "description": "5-shot random, stock template",
        "task": "hellaswag",
        "num_fewshot": 5,
        "custom": False,
    },
    "fewshot_random_10": {
        "description": "10-shot random, stock template",
        "task": "hellaswag",
        "num_fewshot": 10,
        "custom": False,
    },
    "fewshot_semantic_5": {
        "description": "5-shot semantic (TF-IDF), stock template",
        "build": {"n_shots": 5, "semantic": True, "template": TEMPLATE_STOCK},
        "num_fewshot": 0,
        "custom": True,
    },
    "fewshot_semantic_10": {
        "description": "10-shot semantic (TF-IDF), stock template",
        "build": {"n_shots": 10, "semantic": True, "template": TEMPLATE_STOCK},
        "num_fewshot": 0,
        "custom": True,
    },
    "template_v1": {
        "description": "0-shot, instruction template",
        "build": {"n_shots": 0, "semantic": False, "template": TEMPLATE_INSTRUCTION},
        "num_fewshot": 0,
        "custom": True,
    },
    "template_v1_semantic_10": {
        "description": "10-shot semantic + instruction template (full combo)",
        "build": {"n_shots": 10, "semantic": True, "template": TEMPLATE_INSTRUCTION},
        "num_fewshot": 0,
        "custom": True,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Optimize HellaSwag prompt configuration")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--configs", nargs="+", default=["all"],
                        help="Config names to run, or 'all'")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=f"http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Ensure server is running
    server = OllamaServer(model=args.model)
    if not server.is_healthy():
        log.info("Starting Ollama …")
        server.start()

    # Load prepared data (for custom configs)
    train_data, val_data, fewshot_map = None, None, None
    train_path = DATA_DIR / "train.json"
    val_path = DATA_DIR / "val.json"
    map_path = DATA_DIR / "fewshot_map.json"

    if train_path.exists():
        with open(train_path) as f:
            train_data = json.load(f)
        with open(val_path) as f:
            val_data = json.load(f)
        with open(map_path) as f:
            fewshot_map = json.load(f)
        log.info("Loaded prepared data: %d train, %d val, %d fewshot entries",
                 len(train_data), len(val_data), len(fewshot_map))

    # Determine which configs to run
    if "all" in args.configs:
        config_names = list(CONFIGS.keys())
    else:
        config_names = args.configs

    all_results = {}

    for name in config_names:
        cfg = CONFIGS[name]
        log.info("=" * 60)
        log.info("Running config: %s — %s", name, cfg["description"])
        log.info("=" * 60)

        task_name = cfg.get("task")

        # Build custom dataset if needed
        if cfg.get("custom") and cfg.get("build"):
            if val_data is None:
                log.error("Need prepared data for custom configs. Run prepare_data.py first.")
                sys.exit(1)

            build = cfg["build"]
            rows = build_custom_dataset(
                val_data=val_data,
                train_data=train_data,
                fewshot_map=fewshot_map if build["semantic"] else None,
                n_shots=build["n_shots"],
                template=build["template"],
                limit=args.limit,
            )
            task_name = write_dataset_and_yaml(name, rows)

        result = run_config(
            task_name=task_name,
            num_fewshot=cfg["num_fewshot"],
            limit=args.limit if not cfg.get("custom") else None,
            use_custom_tasks=cfg.get("custom", False),
        )

        result["_config"] = name
        result["_description"] = cfg["description"]
        all_results[name] = result

        acc = result.get("acc,none", result.get("acc,get_answer", "?"))
        acc_norm = result.get("acc_norm,none", "n/a")
        elapsed = result.get("_elapsed_sec", "?")
        log.info("  → acc=%.4f  acc_norm=%s  elapsed=%ss", acc, acc_norm, elapsed)

        # Save incrementally
        out_path = RESULTS_DIR / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*72}")
    print(f"  {'Config':<30} {'acc':>8} {'acc_norm':>10} {'time':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8}")
    for name, r in all_results.items():
        acc = r.get("acc,none", r.get("acc,get_answer", 0))
        acc_norm = r.get("acc_norm,none", None)
        elapsed = r.get("_elapsed_sec", 0)
        norm_str = f"{acc_norm:.4f}" if isinstance(acc_norm, float) else "n/a"
        print(f"  {name:<30} {acc:>8.4f} {norm_str:>10} {elapsed:>7.0f}s")
    print(f"{'='*72}\n")

    # Save combined results
    combined_path = RESULTS_DIR / "optimization_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("All results → %s", combined_path)


if __name__ == "__main__":
    main()
