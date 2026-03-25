from __future__ import annotations

import subprocess
from typing import Any

import httpx

from .base import BaseCollector


class EnvironmentCollector(BaseCollector):
    """Collects static environment info once at benchmark startup.

    Sources
    -------
    * ``nvidia-smi`` subprocess  → GPU name, VRAM total / used / free (GB).
    * vLLM ``GET /v1/models``    → served model name.
    * vLLM ``GET /metrics``      → KV-cache blocks allocated (Prometheus).
    """

    def __init__(self, base_url: str, gpu_index: int = 0) -> None:
        self._base_url = base_url.rstrip("/")
        self._gpu_index = gpu_index
        self._data: dict[str, Any] = {}

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._data = {}
        self._collect_gpu()
        self._collect_model()
        self._collect_kv_cache()

    def stop(self) -> None:
        pass

    def results(self) -> dict[str, Any]:
        return dict(self._data)

    # -- gpu via nvidia-smi -----------------------------------------------

    def _collect_gpu(self) -> None:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self._gpu_index}",
                    "--query-gpu=name,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                return

            parts = [p.strip() for p in proc.stdout.strip().split(",")]
            self._data["gpu"] = parts[0]
            self._data["environment"] = {
                "vram_total_gb": round(int(parts[1]) / 1024, 2),
                "vram_used_gb": round(int(parts[2]) / 1024, 2),
                "vram_free_gb": round(int(parts[3]) / 1024, 2),
            }
        except (OSError, IndexError, ValueError):
            pass

    # -- model via vllm openai api ----------------------------------------

    def _collect_model(self) -> None:
        try:
            resp = httpx.get(f"{self._base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if data:
                self._data["model"] = data[0]["id"]
        except (httpx.HTTPError, KeyError, IndexError):
            pass

    # -- kv cache blocks via prometheus metrics ----------------------------

    def _collect_kv_cache(self) -> None:
        try:
            resp = httpx.get(f"{self._base_url}/metrics", timeout=10)
            resp.raise_for_status()

            for line in resp.text.splitlines():
                if line.startswith("vllm:num_gpu_blocks"):
                    blocks = int(float(line.split()[-1]))
                    self._data.setdefault("environment", {})["kv_cache_blocks"] = blocks
                    return
        except (httpx.HTTPError, ValueError):
            pass
