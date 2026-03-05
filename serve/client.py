"""Typed client for the Ollama HTTP API.

Provides both streaming and non-streaming generation, plus helpers for
chat, embeddings, and model listing.  All methods include structured
logging so every call is observable.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Generator, Optional

import requests

from serve.serve import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"


@dataclass
class GenerationResult:
    """Structured response from a /api/generate call."""

    prompt: str
    response: str
    model: str
    total_duration_ms: float
    prompt_eval_count: int  # tokens in prompt
    eval_count: int  # tokens generated
    eval_duration_ms: float  # generation wall-time
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0

    def __post_init__(self):
        if self.eval_duration_ms > 0:
            self.tokens_per_second = (
                self.eval_count / (self.eval_duration_ms / 1000)
            )


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatResult:
    messages: list[ChatMessage]
    response: str
    model: str
    total_duration_ms: float
    eval_count: int
    tokens_per_second: float = 0.0


class OllamaClient:
    """Synchronous client for Ollama's REST API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Raw generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        seed: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Send a prompt and return a structured result."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if seed is not None:
            payload["options"]["seed"] = seed
        if stop:
            payload["options"]["stop"] = stop

        t0 = time.perf_counter()
        resp = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        data = resp.json()
        elapsed = (time.perf_counter() - t0) * 1000

        result = GenerationResult(
            prompt=prompt,
            response=data.get("response", ""),
            model=data.get("model", self.model),
            total_duration_ms=data.get("total_duration", 0) / 1e6,
            prompt_eval_count=data.get("prompt_eval_count", 0),
            eval_count=data.get("eval_count", 0),
            eval_duration_ms=data.get("eval_duration", 0) / 1e6,
        )

        log.info(
            "generate | model=%s prompt_tokens=%d gen_tokens=%d "
            "tok/s=%.1f wall=%.0fms",
            result.model,
            result.prompt_eval_count,
            result.eval_count,
            result.tokens_per_second,
            elapsed,
        )
        return result

    def generate_stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        seed: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Yield tokens as they arrive (streaming mode)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if seed is not None:
            payload["options"]["seed"] = seed

        resp = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    return

    def generate_stream_timed(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        seed: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Stream a generation and return a result with TTFT measured."""
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if seed is not None:
            payload["options"]["seed"] = seed
        if stop:
            payload["options"]["stop"] = stop

        t0 = time.perf_counter()
        resp = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        ttft: Optional[float] = None
        tokens: list[str] = []
        final: dict = {}
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            tok = chunk.get("response", "")
            if tok:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                tokens.append(tok)
            if chunk.get("done"):
                final = chunk
                break

        t_end = time.perf_counter()
        wall_ms = (t_end - t0) * 1000
        eval_count = final.get("eval_count", len(tokens))
        eval_dur_ns = final.get("eval_duration", 0)
        eval_dur_ms = eval_dur_ns / 1e6

        result = GenerationResult(
            prompt=prompt,
            response="".join(tokens),
            model=final.get("model", self.model),
            total_duration_ms=wall_ms,
            prompt_eval_count=final.get("prompt_eval_count", 0),
            eval_count=eval_count,
            eval_duration_ms=eval_dur_ms,
            time_to_first_token_ms=ttft or wall_ms,
        )

        log.info(
            "generate_stream_timed | model=%s prompt_tokens=%d gen_tokens=%d "
            "tok/s=%.1f ttft=%.0fms wall=%.0fms",
            result.model, result.prompt_eval_count, result.eval_count,
            result.tokens_per_second, result.time_to_first_token_ms, wall_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        seed: Optional[int] = None,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        if seed is not None:
            payload["options"]["seed"] = seed

        resp = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        assistant_text = data["message"]["content"]
        tps = 0.0
        eval_dur = data.get("eval_duration", 0)
        eval_count = data.get("eval_count", 0)
        if eval_dur > 0:
            tps = eval_count / (eval_dur / 1e9)

        return ChatResult(
            messages=messages,
            response=assistant_text,
            model=data.get("model", self.model),
            total_duration_ms=data.get("total_duration", 0) / 1e6,
            eval_count=eval_count,
            tokens_per_second=tps,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        resp = self.session.get(
            f"{self.base_url}/api/tags", timeout=10
        )
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def is_healthy(self) -> bool:
        try:
            r = self.session.get(
                f"{self.base_url}/api/tags", timeout=2
            )
            return r.status_code == 200
        except requests.ConnectionError:
            return False


# ======================================================================
# Sample script — run a few prompt generations
# ======================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample generations from the Ollama server"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    client = OllamaClient(base_url=args.base_url, model=args.model)

    if not client.is_healthy():
        print("ERROR: Ollama server is not running. Start with: python serve/serve.py")
        return

    print(f"\n{'='*60}")
    print(f"  Ollama Sample Generations  |  model: {args.model}")
    print(f"{'='*60}\n")

    prompts = [
        "Explain quantum entanglement in two sentences.",
        "Write a Python function that checks if a number is prime.",
        "What are three benefits of test-driven development?",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"--- Prompt {i} ---")
        print(f"  {prompt}\n")

        result = client.generate(
            prompt,
            temperature=0.7,
            max_tokens=200,
            seed=42,
        )

        print(f"  Response ({result.eval_count} tokens, "
              f"{result.tokens_per_second:.1f} tok/s):")
        for line in result.response.strip().splitlines():
            print(f"    {line}")
        print()

    # Streaming example
    print("--- Streaming Example ---")
    print("  Prompt: Count from 1 to 5 with a word for each number.\n")
    print("  Response: ", end="", flush=True)
    for token in client.generate_stream(
        "Count from 1 to 5 with a word for each number.",
        temperature=0.3,
        max_tokens=100,
        seed=42,
    ):
        print(token, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()
