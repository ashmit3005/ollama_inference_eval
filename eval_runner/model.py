"""Custom lm-evaluation-harness model wrapper for an Ollama endpoint.

Bridges Part A (serve/) with lm-eval by implementing the three abstract
methods required by lm_eval.api.model.LM:
  - loglikelihood        (MMLU, HellaSwag, ARC, etc.)
  - loglikelihood_rolling (perplexity benchmarks)
  - generate_until       (generative benchmarks like code_output)

Scoring strategy — loglikelihood:
  Ollama (v0.16.2) exposes logprobs only for *generated* tokens via
  /api/generate.  The OpenAI-compatible /v1/completions does not support
  ``echo`` or prompt-token logprobs, so teacher-forced scoring is not
  available.  We approximate it with token-by-token greedy scoring:

    1. Send prompt=context with num_predict=1, logprobs=true, top_logprobs=20.
    2. In the returned top-20 logprobs, find the token that is a prefix
       of the remaining continuation.
    3. If found: record its logprob, append it to the prompt, advance.
    4. If NOT found: assign a *soft floor* (min logprob in top-20 minus 1.0)
       and advance by one character of the continuation.
    5. Repeat until the full continuation is consumed.

  The soft floor avoids the failure mode where a single token outside
  top-20 dominates the total score (the earlier hard floor of -100 did
  this, causing artificially low accuracy on HellaSwag/MMLU).

  Limitations of this approach vs true teacher-forced scoring:
    - top-20 truncation: tokens ranked below 20th get an estimated logprob
    - char-by-char advance on miss: sends mid-token prompts, which may
      produce slightly different logprobs than token-boundary prompts
    - slower than single-call /v1/completions echo (N calls per option
      instead of 1), but Ollama does not support the single-call path
    - absolute accuracy is lower than published benchmarks; relative
      rankings (and therefore improvement deltas) remain valid

  Because all options in a multiple-choice question share the same context,
  the first API call (from context alone) is cached and reused via
  _gen1_cache, amortizing the most expensive call across all options.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Optional

import aiohttp
import requests
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from serve.serve import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_MODEL
from serve.client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

LOGPROB_FLOOR = -100.0


@register_model("ollama")
class OllamaEvalModel(LM):
    """lm-eval wrapper that queries a running Ollama server.

    Uses shared configuration from serve/ (DEFAULT_HOST, DEFAULT_PORT,
    DEFAULT_MODEL) and OllamaClient for health checks, ensuring the
    eval pipeline is not disjoint from the serving layer.
    """

    def __init__(
        self,
        base_url: str = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}",
        model: str = DEFAULT_MODEL,
        max_tokens_generate: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        top_logprobs: int = 20,
        batch_size: int = 1,
        max_score_tokens: int = 50,
        scoring_mode: str = "soft_floor",
        parallel_choices: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._max_gen = max_tokens_generate
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.top_logprobs_k = top_logprobs
        self._batch_size = batch_size
        self._max_score_tokens = max_score_tokens
        self.scoring_mode = scoring_mode
        self.parallel_choices = parallel_choices
        self.session = requests.Session()
        self._gen1_cache: dict[tuple, dict] = {}
        self._client = OllamaClient(base_url=self.base_url, model=self.model)
        self._verify_connection()
        log.info("Scoring mode: %s", self.scoring_mode)
        log.info("Parallel choices: %s", self.parallel_choices)

    def _verify_connection(self) -> None:
        if not self._client.is_healthy():
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Start with: python serve/serve.py"
            )
        available = self._client.list_models()
        if self.model not in available:
            raise RuntimeError(
                f"Model '{self.model}' not found. Available: {available}"
            )
        log.info("Connected to Ollama — model '%s' ready.", self.model)

    # ------------------------------------------------------------------
    # Native API: generate 1 token with logprobs
    # ------------------------------------------------------------------

    def _generate_one(self, prompt: str) -> dict:
        """Call /api/generate for 1 token with logprobs.

        Cached in-memory by (model, seed, prompt) so that all MC options
        sharing the same context hit this cache on the first call.
        """
        cache_key = (self.model, self.seed, prompt)
        if cache_key in self._gen1_cache:
            return self._gen1_cache[cache_key]

        resp = self.session.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "raw": True,
                "logprobs": True,
                "top_logprobs": self.top_logprobs_k,
                "options": {
                    "temperature": 0,
                    "top_p": 1.0,
                    "num_predict": 1,
                    "seed": self.seed,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        self._gen1_cache[cache_key] = data
        return data

    # ------------------------------------------------------------------
    # Async API: generate 1 token with logprobs (for parallel scoring)
    # ------------------------------------------------------------------

    async def _async_generate_one(
        self, session: aiohttp.ClientSession, prompt: str
    ) -> dict:
        cache_key = (self.model, self.seed, prompt)
        if cache_key in self._gen1_cache:
            return self._gen1_cache[cache_key]

        async with session.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "raw": True,
                "logprobs": True,
                "top_logprobs": self.top_logprobs_k,
                "options": {
                    "temperature": 0,
                    "top_p": 1.0,
                    "num_predict": 1,
                    "seed": self.seed,
                },
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            self._gen1_cache[cache_key] = data
            return data

    async def _async_score_continuation(
        self, session: aiohttp.ClientSession, context: str, continuation: str
    ) -> tuple[float, bool]:
        if not continuation.strip():
            return (0.0, True)

        total_logprob = 0.0
        is_greedy = True
        remaining = continuation
        current_prompt = context
        tokens_scored = 0
        use_soft_floor = self.scoring_mode == "soft_floor"

        while remaining and tokens_scored < self._max_score_tokens:
            data = await self._async_generate_one(session, current_prompt)
            logprobs_list = data.get("logprobs", [])

            if not logprobs_list:
                total_logprob += LOGPROB_FLOOR
                is_greedy = False
                break

            token_info = logprobs_list[0]
            top_probs = token_info.get("top_logprobs", [token_info])
            min_lp = min(tp["logprob"] for tp in top_probs) if top_probs else LOGPROB_FLOOR
            soft_floor = min_lp - 1.0

            best_token_text: Optional[str] = None
            best_logprob = soft_floor
            for tp in top_probs:
                tok_text = tp["token"]
                if remaining.startswith(tok_text) and len(tok_text) > 0:
                    if best_token_text is None or len(tok_text) > len(best_token_text):
                        best_token_text = tok_text
                        best_logprob = tp["logprob"]

            if best_token_text is None:
                if use_soft_floor:
                    advance = remaining[0]
                    total_logprob += soft_floor
                    is_greedy = False
                    current_prompt += advance
                    remaining = remaining[len(advance):]
                    tokens_scored += 1
                    continue
                else:
                    total_logprob += LOGPROB_FLOOR
                    is_greedy = False
                    break

            greedy_token = top_probs[0]["token"]
            if greedy_token != best_token_text:
                is_greedy = False

            total_logprob += best_logprob
            current_prompt += best_token_text
            remaining = remaining[len(best_token_text):]
            tokens_scored += 1

        return (total_logprob, is_greedy)

    async def _score_group_parallel(
        self,
        session: aiohttp.ClientSession,
        group: list[tuple[int, str, str]],
    ) -> list[tuple[int, tuple[float, bool]]]:
        """Score a group of (index, context, continuation) concurrently."""
        tasks = [
            self._async_score_continuation(session, ctx, cont)
            for _, ctx, cont in group
        ]
        scores = await asyncio.gather(*tasks)
        return [(idx, score) for (idx, _, _), score in zip(group, scores)]

    def _run_parallel_loglikelihood(
        self, requests_list
    ) -> list[tuple[float, bool]]:
        """Group requests by context, score each group's choices in parallel."""
        groups: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        for i, req in enumerate(requests_list):
            context, continuation = req.args
            groups[context].append((i, context, continuation))

        results_map: dict[int, tuple[float, bool]] = {}
        pbar = tqdm(total=len(requests_list), desc="loglikelihood(parallel)")

        async def _run():
            connector = aiohttp.TCPConnector(limit=8)
            async with aiohttp.ClientSession(connector=connector) as session:
                for context, group in groups.items():
                    scored = await self._score_group_parallel(session, group)
                    for idx, score in scored:
                        results_map[idx] = score
                    pbar.update(len(group))

        asyncio.run(_run())
        pbar.close()
        return [results_map[i] for i in range(len(requests_list))]

    # ------------------------------------------------------------------
    # Go proxy: batch scoring via the Go scoring proxy (perf/scorer)
    # ------------------------------------------------------------------

    def _run_go_proxy_loglikelihood(
        self, requests_list
    ) -> list[tuple[float, bool]]:
        """Send all requests to the Go scoring proxy in one batch."""
        items = []
        for i, req in enumerate(requests_list):
            context, continuation = req.args
            items.append({
                "index": i,
                "context": context,
                "continuation": continuation,
            })

        payload = {
            "items": items,
            "config": {
                "model": self.model,
                "seed": self.seed,
                "top_logprobs": self.top_logprobs_k,
                "max_score_tokens": self._max_score_tokens,
                "scoring_mode": self.scoring_mode,
            },
        }

        go_url = getattr(self, "go_proxy_url", "http://localhost:9090")
        resp = self.session.post(
            f"{go_url}/score",
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()

        results_by_idx = {r["index"]: r for r in data["results"]}
        log.info(
            "Go proxy scored %d items in %.1fs",
            len(items), data.get("elapsed_sec", 0),
        )
        return [
            (results_by_idx[i]["logprob"], results_by_idx[i]["is_greedy"])
            for i in range(len(requests_list))
        ]

    # ------------------------------------------------------------------
    # loglikelihood  (context, continuation) → (logprob, is_greedy)
    # ------------------------------------------------------------------

    def loglikelihood(self, requests_list) -> list[tuple[float, bool]]:
        if self.parallel_choices == "go":
            return self._run_go_proxy_loglikelihood(requests_list)
        if self.parallel_choices:
            return self._run_parallel_loglikelihood(requests_list)

        results = []
        for req in tqdm(requests_list, desc="loglikelihood"):
            context, continuation = req.args
            result = self._score_continuation(context, continuation)
            results.append(result)
        return results

    def _score_continuation(
        self, context: str, continuation: str
    ) -> tuple[float, bool]:
        """Score continuation by accumulating real token logprobs.

        Two scoring modes (selectable via ``scoring_mode`` init parameter):

        ``"hard_floor"`` (Part B original):
          On token miss (not in top-k), assign LOGPROB_FLOOR (-100) and
          stop scoring.  Fast, but a single miss dominates the total score
          and depresses accuracy well below published benchmarks.

        ``"soft_floor"`` (Part E improved, default):
          On token miss, assign ``min(top-k logprobs) - 1.0`` and advance
          by one character.  Scores the full continuation and produces
          rankings that more faithfully track ground truth, at the cost of
          more API calls per continuation.

        Both modes are kept for cross-part comparison (Part B vs Part E).
        """
        if not continuation.strip():
            return (0.0, True)

        total_logprob = 0.0
        is_greedy = True
        remaining = continuation
        current_prompt = context
        tokens_scored = 0
        use_soft_floor = self.scoring_mode == "soft_floor"

        while remaining and tokens_scored < self._max_score_tokens:
            data = self._generate_one(current_prompt)
            logprobs_list = data.get("logprobs", [])

            if not logprobs_list:
                total_logprob += LOGPROB_FLOOR
                is_greedy = False
                break

            token_info = logprobs_list[0]
            top_probs = token_info.get("top_logprobs", [token_info])

            min_lp = min(tp["logprob"] for tp in top_probs) if top_probs else LOGPROB_FLOOR
            soft_floor = min_lp - 1.0

            best_token_text: Optional[str] = None
            best_logprob = soft_floor
            for tp in top_probs:
                tok_text = tp["token"]
                if remaining.startswith(tok_text) and len(tok_text) > 0:
                    if best_token_text is None or len(tok_text) > len(best_token_text):
                        best_token_text = tok_text
                        best_logprob = tp["logprob"]

            if best_token_text is None:
                if use_soft_floor:
                    advance = remaining[0]
                    total_logprob += soft_floor
                    is_greedy = False
                    current_prompt += advance
                    remaining = remaining[len(advance):]
                    tokens_scored += 1
                    continue
                else:
                    total_logprob += LOGPROB_FLOOR
                    is_greedy = False
                    break

            greedy_token = top_probs[0]["token"]
            if greedy_token != best_token_text:
                is_greedy = False

            total_logprob += best_logprob
            current_prompt += best_token_text
            remaining = remaining[len(best_token_text):]
            tokens_scored += 1

        return (total_logprob, is_greedy)

    # ------------------------------------------------------------------
    # loglikelihood_rolling  (string) → logprob
    # ------------------------------------------------------------------

    def loglikelihood_rolling(self, requests_list) -> list[float]:
        results = []
        for req in tqdm(requests_list, desc="loglikelihood_rolling"):
            (text,) = req.args
            score, _ = self._score_continuation("", text)
            results.append(score)
        return results

    # ------------------------------------------------------------------
    # generate_until  (context, gen_kwargs) → generated string
    # ------------------------------------------------------------------

    def generate_until(self, requests_list) -> list[str]:
        results = []
        for req in tqdm(requests_list, desc="generate_until"):
            context, gen_kwargs = req.args

            until = gen_kwargs.get("until", [])
            max_tokens = gen_kwargs.get("max_gen_toks", self._max_gen)
            stop = until if isinstance(until, list) else [until]

            payload = {
                "model": self.model,
                "prompt": context,
                "stream": False,
                "raw": True,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": max_tokens,
                    "seed": self.seed,
                },
            }
            if stop:
                payload["options"]["stop"] = stop

            resp = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")

            for s in stop:
                if s in text:
                    text = text[: text.index(s)]

            results.append(text.strip())

        return results

    # ------------------------------------------------------------------
    # Properties required by the harness
    # ------------------------------------------------------------------

    @property
    def eot_token_id(self):
        return None

    @property
    def max_length(self):
        return 4096

    @property
    def max_gen_toks(self):
        return self._max_gen

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cpu"

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return list(range(len(string.split())))

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return ""
