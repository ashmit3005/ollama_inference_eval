"""Ollama inference server manager.

Wraps the ollama CLI to provide a programmatic start/stop/health-check
interface.  Running this file directly starts the server and blocks until
interrupted.
"""

import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_MODEL = "llama3:8b"
HEALTH_ENDPOINT = "/api/tags"
STARTUP_TIMEOUT = 30  # seconds


class OllamaServer:
    """Manage an Ollama server subprocess."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        model: str = DEFAULT_MODEL,
    ):
        self.host = host
        self.port = port
        self.model = model
        self.base_url = f"http://{host}:{port}"
        self._proc: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout: int = STARTUP_TIMEOUT) -> "OllamaServer":
        """Start the Ollama server if it isn't already running."""
        if self.is_healthy():
            log.info("Ollama already running at %s", self.base_url)
            self._ensure_model()
            return self

        ollama_bin = shutil.which("ollama")
        if ollama_bin is None:
            raise RuntimeError(
                "ollama binary not found on PATH. "
                "Install from https://ollama.com"
            )

        log.info("Starting Ollama server on %s …", self.base_url)
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{self.host}:{self.port}"

        self._log_path = os.path.join(
            os.path.dirname(__file__), "..", "logs", "ollama.log"
        )
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        self._log_file = open(self._log_path, "a")
        log.info("Ollama logs → %s", os.path.abspath(self._log_path))

        self._proc = subprocess.Popen(
            [ollama_bin, "serve"],
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        atexit.register(self.stop)

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_healthy():
                log.info("Ollama server ready (pid=%d)", self._proc.pid)
                self._ensure_model()
                return self
            time.sleep(0.5)

        self.stop()
        raise TimeoutError(
            f"Ollama did not become healthy within {timeout}s"
        )

    def stop(self) -> None:
        """Gracefully terminate the server subprocess."""
        if self._proc and self._proc.poll() is None:
            log.info("Stopping Ollama server (pid=%d) …", self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if hasattr(self, "_log_file") and self._log_file:
            self._log_file.close()
            self._log_file = None

    def is_healthy(self) -> bool:
        """Return True if the server responds to a health probe."""
        try:
            r = requests.get(
                f"{self.base_url}{HEALTH_ENDPOINT}", timeout=2
            )
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Pull the configured model if it's not already present."""
        r = requests.get(f"{self.base_url}/api/tags", timeout=10)
        models = {m["name"] for m in r.json().get("models", [])}
        if self.model not in models:
            log.info("Pulling model %s (this may take a while) …", self.model)
            resp = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                stream=True,
                timeout=600,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    log.debug(line.decode())
            log.info("Model %s ready.", self.model)
        else:
            log.info("Model %s already available.", self.model)

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.base_url}/api/tags", timeout=10)
        return [m["name"] for m in r.json().get("models", [])]


# ======================================================================
# CLI entry‑point
# ======================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Start the Ollama server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    server = OllamaServer(host=args.host, port=args.port, model=args.model)
    server.start()

    def _shutdown(sig, frame):
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("Server running. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
