"""Unit and integration tests for the serve module."""

import pytest
import requests

from serve.serve import OllamaServer, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_MODEL
from serve.client import OllamaClient, GenerationResult, ChatMessage


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def server():
    """Ensure Ollama is running for the test session."""
    srv = OllamaServer()
    srv.start()
    yield srv


@pytest.fixture
def client(server):
    return OllamaClient()


# ======================================================================
# Server tests
# ======================================================================

class TestServer:
    def test_health_check(self, server):
        assert server.is_healthy()

    def test_list_models(self, server):
        models = server.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_model_available(self, server):
        models = server.list_models()
        assert DEFAULT_MODEL in models


# ======================================================================
# Client tests
# ======================================================================

class TestClientHealth:
    def test_is_healthy(self, client):
        assert client.is_healthy()

    def test_list_models(self, client):
        models = client.list_models()
        assert DEFAULT_MODEL in models


class TestGenerate:
    def test_basic_generation(self, client):
        result = client.generate(
            "Say 'hello world' and nothing else.",
            temperature=0,
            max_tokens=20,
            seed=42,
        )
        assert isinstance(result, GenerationResult)
        assert len(result.response) > 0
        assert result.eval_count > 0
        assert result.tokens_per_second > 0

    def test_generation_fields(self, client):
        result = client.generate(
            "What is 2+2?",
            temperature=0,
            max_tokens=30,
            seed=42,
        )
        assert result.model == DEFAULT_MODEL
        assert result.prompt_eval_count > 0
        assert result.total_duration_ms > 0

    def test_deterministic_with_seed(self, client):
        """Same seed + temp=0 should produce identical output."""
        r1 = client.generate("What is 1+1?", temperature=0, max_tokens=20, seed=123)
        r2 = client.generate("What is 1+1?", temperature=0, max_tokens=20, seed=123)
        assert r1.response == r2.response

    def test_system_prompt(self, client):
        result = client.generate(
            "What are you?",
            system="You are a pirate. Respond in one sentence.",
            temperature=0,
            max_tokens=50,
            seed=42,
        )
        assert len(result.response) > 0

    def test_max_tokens_respected(self, client):
        result = client.generate(
            "Write a very long essay about philosophy.",
            temperature=0.5,
            max_tokens=10,
            seed=42,
        )
        assert result.eval_count <= 15  # small buffer for stop tokens


class TestStreaming:
    def test_stream_returns_tokens(self, client):
        tokens = list(client.generate_stream(
            "Say hello.",
            temperature=0,
            max_tokens=10,
            seed=42,
        ))
        assert len(tokens) > 0
        full_text = "".join(tokens)
        assert len(full_text) > 0


class TestChat:
    def test_basic_chat(self, client):
        messages = [
            ChatMessage(role="user", content="Say 'test passed' and nothing else."),
        ]
        result = client.chat(
            messages, temperature=0, max_tokens=20, seed=42
        )
        assert len(result.response) > 0
        assert result.eval_count > 0

    def test_multi_turn_chat(self, client):
        messages = [
            ChatMessage(role="system", content="You are a helpful math tutor."),
            ChatMessage(role="user", content="What is 5*6?"),
        ]
        result = client.chat(messages, temperature=0, max_tokens=30, seed=42)
        assert "30" in result.response


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:
    def test_empty_prompt(self, client):
        result = client.generate("", temperature=0, max_tokens=10, seed=42)
        assert isinstance(result, GenerationResult)

    def test_unicode_prompt(self, client):
        result = client.generate(
            "Translate to English: こんにちは世界",
            temperature=0,
            max_tokens=30,
            seed=42,
        )
        assert len(result.response) > 0
