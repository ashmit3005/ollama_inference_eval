"""Prepare data for HellaSwag benchmark improvement.

Downloads the HellaSwag dataset, computes TF-IDF vectors over training
examples, and for each validation example selects the k most similar
training examples for semantic few-shot selection.

Outputs:
  improve/data/fewshot_map.json  — {val_idx: [train_idx, ...]} mapping
  improve/data/train.json        — training examples (for prompt construction)

Usage:
    python improve/prepare_data.py
    python improve/prepare_data.py --k 10 --limit 200
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"


def preprocess(text: str) -> str:
    """Match lm-eval's HellaSwag preprocessing."""
    import re
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def build_query(doc: dict) -> str:
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return preprocess(doc["activity_label"] + ": " + ctx)


def load_hellaswag():
    """Load HellaSwag train + validation splits via HuggingFace datasets."""
    from datasets import load_dataset
    log.info("Loading HellaSwag dataset …")
    ds = load_dataset("Rowan/hellaswag", trust_remote_code=True)
    return ds["train"], ds["validation"]


def build_tfidf_index(train_docs: list[dict]):
    """Build TF-IDF matrix over training queries."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    queries = [build_query(d) for d in train_docs]
    log.info("Fitting TF-IDF on %d training examples …", len(queries))
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(queries)
    return vectorizer, train_matrix


def select_fewshot(
    vectorizer,
    train_matrix,
    val_docs: list[dict],
    k: int = 10,
    limit: int | None = None,
) -> dict[int, list[int]]:
    """For each validation example, find the k most similar training examples."""
    from sklearn.metrics.pairwise import cosine_similarity

    val_queries = [build_query(d) for d in val_docs]
    if limit:
        val_queries = val_queries[:limit]

    log.info("Computing similarity for %d validation examples (k=%d) …", len(val_queries), k)
    val_matrix = vectorizer.transform(val_queries)

    fewshot_map = {}
    batch_size = 500
    for start in range(0, val_matrix.shape[0], batch_size):
        end = min(start + batch_size, val_matrix.shape[0])
        sims = cosine_similarity(val_matrix[start:end], train_matrix)
        for i, row in enumerate(sims):
            top_k = row.argsort()[-k:][::-1].tolist()
            fewshot_map[start + i] = top_k

    return fewshot_map


def main():
    parser = argparse.ArgumentParser(description="Prepare HellaSwag data for improvement")
    parser.add_argument("--k", type=int, default=10, help="Number of few-shot examples per question")
    parser.add_argument("--limit", type=int, default=None, help="Limit validation examples")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    train_split, val_split = load_hellaswag()

    train_docs = [dict(row) for row in train_split]
    val_docs = [dict(row) for row in val_split]
    log.info("Train: %d  |  Validation: %d", len(train_docs), len(val_docs))

    # Save compact training data (needed later for building few-shot prompts)
    train_compact = []
    for d in train_docs:
        train_compact.append({
            "query": build_query(d),
            "choices": [preprocess(e) for e in d["endings"]],
            "gold": int(d["label"]),
            "activity_label": d["activity_label"],
        })
    train_path = DATA_DIR / "train.json"
    with open(train_path, "w") as f:
        json.dump(train_compact, f)
    log.info("Training data → %s (%d examples)", train_path, len(train_compact))

    # Save compact validation data
    val_compact = []
    for d in val_docs:
        val_compact.append({
            "query": build_query(d),
            "choices": [preprocess(e) for e in d["endings"]],
            "gold": int(d["label"]),
            "activity_label": d["activity_label"],
        })
    val_path = DATA_DIR / "val.json"
    with open(val_path, "w") as f:
        json.dump(val_compact, f)
    log.info("Validation data → %s (%d examples)", val_path, len(val_compact))

    # Build TF-IDF index and compute few-shot selections
    vectorizer, train_matrix = build_tfidf_index(train_docs)

    fewshot_map = select_fewshot(
        vectorizer, train_matrix, val_docs,
        k=args.k, limit=args.limit,
    )

    map_path = DATA_DIR / "fewshot_map.json"
    with open(map_path, "w") as f:
        json.dump(fewshot_map, f)
    log.info("Few-shot map → %s (%d entries, k=%d)", map_path, len(fewshot_map), args.k)

    # Quick sanity check: show first example's selected few-shot
    if fewshot_map:
        idx = 0
        neighbors = fewshot_map[str(idx)] if str(idx) in fewshot_map else fewshot_map.get(idx, [])
        if not neighbors:
            neighbors = list(fewshot_map.values())[0]
            idx = list(fewshot_map.keys())[0]
        log.info("Sample — val[%s] query: %s", idx, val_compact[int(idx)]["query"][:80])
        for ni in neighbors[:3]:
            log.info("  neighbor[%d]: %s", ni, train_compact[ni]["query"][:80])

    log.info("Done. Data ready in %s", DATA_DIR)


if __name__ == "__main__":
    main()
