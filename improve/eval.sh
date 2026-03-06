#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Part E: HellaSwag benchmark improvement pipeline
#
# Orchestrates: data preparation → prompt optimization →
# final inference with statistical comparison.
#
# All inference goes through the existing OllamaEvalModel (Part B).
# No model or wrapper changes — only evaluation-time parameters.
#
# Usage:
#   bash improve/eval.sh              # full pipeline
#   bash improve/eval.sh prepare      # data prep only
#   bash improve/eval.sh optimize     # exploration sweep (limit=100)
#   bash improve/eval.sh infer        # final comparison (limit=100)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."
VENV="venv/bin/activate"
if [ -f "$VENV" ]; then source "$VENV"; fi

LIMIT_EXPLORE=100
LIMIT_FINAL=100
SEED=42

step="${1:-all}"

# ── Step 1: Prepare data ──
if [[ "$step" == "all" || "$step" == "prepare" ]]; then
    echo "═══ Step 1: Preparing data (TF-IDF few-shot selection) ═══"
    python improve/prepare_data.py --k 10
fi

# ── Step 2: Exploration sweep ──
if [[ "$step" == "all" || "$step" == "optimize" ]]; then
    echo "═══ Step 2: Exploration sweep (limit=$LIMIT_EXPLORE) ═══"
    python improve/optimize_prompt.py \
        --limit "$LIMIT_EXPLORE" \
        --configs baseline fewshot_semantic_5 fewshot_semantic_10 \
                 fewshot_random_10 template_v1 template_v1_semantic_10
fi

# ── Step 3: Final comparison with statistical tests ──
if [[ "$step" == "all" || "$step" == "infer" ]]; then
    echo "═══ Step 3: Final comparison (limit=$LIMIT_FINAL) ═══"
    # Compare baseline vs best config with per-sample predictions,
    # bootstrap CI, McNemar test, confidence calibration, and
    # flipped-example extraction.
    python improve/infer.py \
        --compare baseline fewshot_semantic_5 \
        --limit "$LIMIT_FINAL"
fi

echo ""
echo "═══ Pipeline complete. Results in improve/results/ ═══"
