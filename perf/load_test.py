"""Performance load generator for the Ollama endpoint.

Scope: tests the *raw Ollama serving layer* (Part A), not the evaluation
harness.  This is intentional — we want to isolate inference latency and
throughput from lm-eval's task-processing overhead.  Deterministic decoding
settings (temperature=0, top_p=1, seed=42) match those verified by
guardrails/validate.py (layer 1) and used by eval_runner/model.py.

Sends concurrent streaming requests with varying prompt lengths,
concurrency levels, cache conditions, and stop-sequence settings.
For every request it records:
    - TTFT  (time-to-first-token, measured via streaming)
    - Total end-to-end latency
    - Tokens generated / tokens per second
    - Prompt token count

Results are written to metrics.csv (one row per request) and a
summary table with P50 / P95 / P99 is printed to stdout.

Usage:
    python perf/load_test.py                     # full matrix
    python perf/load_test.py --concurrency 1 4   # specific levels
    python perf/load_test.py --requests 4        # fewer per config
"""

import argparse
import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from itertools import groupby
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from serve.serve import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_PORT, OllamaServer
from serve.client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Prompt pools ──────────────────────────────────────────────────────

SHORT_PROMPTS = [
    "What is 2+2?",
    "Name three primary colors.",
    "What is the capital of France?",
    "Define gravity in one sentence.",
    "Is water wet? Answer yes or no.",
    "What day comes after Monday?",
    "Name a mammal that can fly.",
    "What is the boiling point of water in Celsius?",
]

LONG_PROMPTS = [
    (
        "Explain the theory of general relativity in detail, covering how "
        "massive objects curve spacetime and how this curvature affects the "
        "motion of other objects. Include examples from everyday life and "
        "astronomical observations that confirm the theory."
    ),
    (
        "Write a comprehensive overview of how the human immune system works, "
        "including the roles of white blood cells, antibodies, the difference "
        "between innate and adaptive immunity, and how vaccines leverage these "
        "mechanisms to protect against disease."
    ),
    (
        "Describe the complete process of photosynthesis from the absorption "
        "of light energy to the production of glucose, including the light-"
        "dependent reactions, the Calvin cycle, and the role of chlorophyll "
        "and other pigments in capturing photons."
    ),
    (
        "Explain the history and evolution of the Internet from ARPANET to "
        "the modern World Wide Web, covering key milestones like the adoption "
        "of TCP/IP, the creation of DNS, the invention of HTTP, and the "
        "emergence of social media platforms."
    ),
    (
        "Provide a detailed comparison of functional programming and object-"
        "oriented programming paradigms, discussing their core principles, "
        "advantages, disadvantages, and use cases. Give specific examples "
        "from languages like Haskell, Python, Java, and Scala."
    ),
    (
        "Describe the architecture and key innovations of transformer models "
        "in natural language processing, covering self-attention, positional "
        "encoding, multi-head attention, and how they differ fundamentally "
        "from recurrent neural networks and LSTMs."
    ),
    (
        "Explain how neural networks learn through the backpropagation "
        "algorithm, covering forward passes, loss function computation, "
        "gradient calculation via the chain rule, weight updates, and common "
        "challenges like vanishing gradients and overfitting."
    ),
    (
        "Discuss the causes and consequences of the Industrial Revolution, "
        "covering the major technological innovations in textiles and steam "
        "power, the social changes including urbanization and labor movements, "
        "and the long-term environmental impacts across Europe."
    ),
]


# ── Per-request metric ────────────────────────────────────────────────

@dataclass
class RequestMetric:
    prompt_type: str       # "short" | "long"
    concurrency: int
    cache_mode: str        # "cold" (unique prompts) | "warm" (same prompt repeated)
    stop_seq: str          # "none" | "newline"
    request_id: int
    prompt_tokens: int
    tokens_generated: int
    ttft_ms: float
    total_latency_ms: float
    tokens_per_second: float


# ── Streaming request with TTFT measurement ───────────────────────────

def _send_streaming(
    client: OllamaClient,
    prompt: str,
    max_tokens: int,
    seed: int,
    stop: list[str] | None,
) -> dict:
    """Send one streaming request via OllamaClient, return timing metrics."""
    r = client.generate_stream_timed(
        prompt,
        temperature=0,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=seed,
        stop=stop,
    )
    return {
        "ttft_ms": round(r.time_to_first_token_ms, 2),
        "total_latency_ms": round(r.total_duration_ms, 2),
        "tokens_generated": r.eval_count,
        "prompt_tokens": r.prompt_eval_count,
        "tokens_per_second": round(r.tokens_per_second, 2),
    }


# ── Run a batch at a given concurrency ────────────────────────────────

def _run_batch(
    client: OllamaClient,
    prompts: list[str],
    prompt_type: str,
    concurrency: int,
    cache_mode: str,
    stop: list[str] | None,
    stop_label: str,
    max_tokens: int,
    n_requests: int,
    seed: int,
) -> list[RequestMetric]:
    if cache_mode == "warm":
        work = [prompts[0]] * n_requests
    else:
        work = [prompts[i % len(prompts)] for i in range(n_requests)]

    metrics: list[RequestMetric] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _send_streaming, client, p, max_tokens, seed, stop
            ): i
            for i, p in enumerate(work)
        }
        for f in as_completed(futures):
            idx = futures[f]
            try:
                r = f.result()
            except Exception as exc:
                log.error("Request %d failed: %s", idx, exc)
                continue
            metrics.append(
                RequestMetric(
                    prompt_type=prompt_type,
                    concurrency=concurrency,
                    cache_mode=cache_mode,
                    stop_seq=stop_label,
                    request_id=idx,
                    **r,
                )
            )
    return metrics


# ── Percentile helper ─────────────────────────────────────────────────

def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ollama performance load test")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--base-url", default=f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    )
    parser.add_argument(
        "--concurrency", nargs="+", type=int, default=[1, 2, 4, 8],
    )
    parser.add_argument(
        "--requests", type=int, default=8,
        help="Number of requests per configuration",
    )
    parser.add_argument("--max-tokens-short", type=int, default=32)
    parser.add_argument("--max-tokens-long", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "perf" / "metrics.csv"),
    )
    args = parser.parse_args()

    # ── ensure server ──
    server = OllamaServer(model=args.model)
    if not server.is_healthy():
        log.info("Starting Ollama server …")
        server.start()
    else:
        log.info("Ollama healthy at %s", args.base_url)

    client = OllamaClient(base_url=args.base_url, model=args.model)

    # ── build config matrix ──
    configs: list[dict] = []
    for pt in ("short", "long"):
        prompts = SHORT_PROMPTS if pt == "short" else LONG_PROMPTS
        mt = args.max_tokens_short if pt == "short" else args.max_tokens_long
        for c in args.concurrency:
            for cm in ("cold", "warm"):
                for stop, sl in [(None, "none"), (["\n"], "newline")]:
                    configs.append(
                        dict(
                            prompts=prompts,
                            prompt_type=pt,
                            concurrency=c,
                            cache_mode=cm,
                            stop=stop,
                            stop_label=sl,
                            max_tokens=mt,
                        )
                    )

    all_metrics: list[RequestMetric] = []
    total = len(configs)
    t_start = time.perf_counter()

    for i, cfg in enumerate(configs, 1):
        tag = (
            f"{cfg['prompt_type']}/c={cfg['concurrency']}/"
            f"{cfg['cache_mode']}/stop={cfg['stop_label']}"
        )
        log.info("[%d/%d] %s  (%d reqs)", i, total, tag, args.requests)

        batch = _run_batch(
            client=client,
            n_requests=args.requests,
            seed=args.seed,
            **cfg,
        )
        all_metrics.extend(batch)

        lats = [m.total_latency_ms for m in batch]
        ttfts = [m.ttft_ms for m in batch]
        log.info(
            "  → P50 lat=%.0fms  TTFT=%.0fms  tok/s=%.1f",
            pct(lats, 50),
            pct(ttfts, 50),
            sum(m.tokens_per_second for m in batch) / max(len(batch), 1),
        )

    wall = time.perf_counter() - t_start
    log.info(
        "Load test finished: %d requests across %d configs in %.1fs",
        len(all_metrics), total, wall,
    )

    # ── write CSV ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = [f.name for f in fields(RequestMetric)]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(asdict(m))
    log.info("Wrote %d rows → %s", len(all_metrics), args.output)

    # ── print summary table ──
    print(f"\n{'=' * 92}")
    print("  PERFORMANCE SUMMARY")
    print(f"{'=' * 92}")
    print(
        f"  {'Config':<42} {'P50 lat':>8} {'P95 lat':>8} {'P99 lat':>8}"
        f" {'P50 TTFT':>9} {'avg tok/s':>10}"
    )
    print(
        f"  {'-' * 42} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 10}"
    )

    def _key(m: RequestMetric):
        return (m.prompt_type, m.concurrency, m.cache_mode, m.stop_seq)

    all_metrics.sort(key=_key)
    for k, grp in groupby(all_metrics, key=_key):
        items = list(grp)
        pt, c, cm, ss = k
        tag = f"{pt}/c={c}/{cm}/stop={ss}"
        lats = [m.total_latency_ms for m in items]
        ttfts = [m.ttft_ms for m in items]
        tps = [m.tokens_per_second for m in items]
        avg_tps = sum(tps) / len(tps) if tps else 0
        print(
            f"  {tag:<42} {pct(lats, 50):>6.0f}ms"
            f" {pct(lats, 95):>6.0f}ms {pct(lats, 99):>6.0f}ms"
            f" {pct(ttfts, 50):>7.0f}ms {avg_tps:>9.1f}"
        )

    print(f"{'=' * 92}\n")


if __name__ == "__main__":
    main()
