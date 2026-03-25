from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from openai import OpenAI

from .base import BaseCollector, RequestRecord, RequestStore


class MetricsCollector:
    """Orchestrates all collectors, assembles the final results dict, and
    provides helpers for saving to JSON and printing a terminal summary.

    Parameters
    ----------
    engine:
        Inference engine name (``"vllm"``, ``"sglang"``, ``"lmdeploy"``).
    store:
        Shared :class:`RequestStore` that the benchmark loop writes into.
    """

    def __init__(self, engine: str, store: RequestStore) -> None:
        self._engine = engine
        self._store = store
        self._collectors: list[BaseCollector] = []
        self._timestamp: str = ""
        self._start_time: float = 0.0
        self._end_time: float = 0.0

    def add(self, collector: BaseCollector) -> MetricsCollector:
        """Register a collector. Returns *self* so calls can be chained."""
        self._collectors.append(collector)
        return self

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Call before the benchmark loop begins."""
        self._timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._start_time = time.perf_counter()
        for collector in self._collectors:
            collector.start()

    def stop(self) -> None:
        """Call after the benchmark loop ends."""
        self._end_time = time.perf_counter()
        for collector in self._collectors:
            collector.stop()

    # -- benchmark runner -------------------------------------------------

    def run(
        self,
        base_url: str,
        prompts: list[str],
        warmup_prompts: list[str] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        """Run the full benchmark: warmup → start → stream each prompt → stop.

        Sends *warmup_prompts* as throwaway requests first to let the server
        JIT-compile kernels and fill CUDA caches, then records real measurements.
        """
        client = OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="dummy")

        models = client.models.list()
        model_name = models.data[0].id
        print(f"Server model: {model_name}")

        if warmup_prompts:
            print(f"Warming up with {len(warmup_prompts)} requests...")
            for i, wp in enumerate(warmup_prompts):
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": wp}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                for _ in stream:
                    pass
                print(f"  warmup [{i + 1}/{len(warmup_prompts)}] done")
            print()

        print(f"Running {len(prompts)} prompts (max_tokens={max_tokens}, temperature={temperature})\n")

        self.start()

        for i, prompt in enumerate(prompts):
            record = RequestRecord(request_id=str(uuid4()), prompt=prompt)
            record.t_start = time.perf_counter()

            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            token_count = 0
            chunks: list[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    if record.t_first_token == 0:
                        record.t_first_token = time.perf_counter()
                    token_count += 1
                    chunks.append(delta.content)

            record.t_last_token = time.perf_counter()
            record.num_output_tokens = token_count
            record.output_text = "".join(chunks)
            self._store.add(record)

            print(
                f"  [{i + 1}/{len(prompts)}]  "
                f"ttft={record.ttft:.4f}s  "
                f"tpot={record.tpot:.4f}s  "
                f"tok/s={record.throughput:.1f}"
            )

        self.stop()

    # -- results ----------------------------------------------------------

    def results(self) -> dict[str, Any]:
        """Assemble the complete benchmark results dict.

        The output layout matches the agreed-upon JSON schema::

            {
              "timestamp", "model", "gpu", "engine",
              "total_requests", "duration_s",
              "environment": { vram + kv },
              "latency": { ttft, tpot, throughput },
              "raw": [ per-request records ]
            }
        """
        merged: dict[str, Any] = {}
        for collector in self._collectors:
            merged.update(collector.results())

        output: dict[str, Any] = {
            "timestamp": self._timestamp,
            "model": merged.pop("model", ""),
            "gpu": merged.pop("gpu", ""),
            "engine": self._engine,
            "total_requests": len(self._store),
            "duration_s": round(self._end_time - self._start_time, 2),
        }

        if "environment" in merged:
            output["environment"] = merged.pop("environment")

        if "latency" in merged:
            output["latency"] = merged.pop("latency")

        output["raw"] = [r.to_dict() for r in self._store.records]

        return output

    # -- persistence ------------------------------------------------------

    def save(self, output_dir: str | Path) -> Path:
        """Auto-generate path and write results to JSON.

        Output structure::

            {output_dir}/{gpu}/{engine}/{model}_{timestamp}.json

        GPU and model names are sanitised for filesystem safety.
        Creates parent dirs if needed.
        """
        r = self.results()

        gpu_slug = _slugify(r.get("gpu", "unknown-gpu"))
        engine_slug = self._engine
        model_slug = _slugify(r.get("model", "unknown-model").split("/")[-1])
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

        path = Path(output_dir) / gpu_slug / engine_slug / f"{model_slug}_{ts}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=2)
        print(f"Results saved → {path}")
        return path

    # -- terminal summary -------------------------------------------------

    def summary(self) -> None:
        """Print a human-readable summary to stdout."""
        r = self.results()

        print(f"\n{'=' * 60}")
        print("  Benchmark Summary")
        print(f"{'=' * 60}")
        print(f"  Timestamp:   {r.get('timestamp', 'n/a')}")
        print(f"  Model:       {r.get('model', 'n/a')}")
        print(f"  GPU:         {r.get('gpu', 'n/a')}")
        print(f"  Engine:      {r.get('engine', 'n/a')}")
        print(f"  Requests:    {r.get('total_requests', 0)}")
        print(f"  Duration:    {r.get('duration_s', 0):.2f}s")

        env = r.get("environment", {})
        if env:
            print(f"\n  VRAM total:  {env.get('vram_total_gb', 'n/a')} GB")
            print(f"  VRAM used:   {env.get('vram_used_gb', 'n/a')} GB")
            print(f"  VRAM free:   {env.get('vram_free_gb', 'n/a')} GB")
            print(f"  KV blocks:   {env.get('kv_cache_blocks', 'n/a')}")

        latency = r.get("latency", {})
        for metric in ("ttft", "tpot", "throughput"):
            stats = latency.get(metric, {})
            if not stats:
                continue
            unit = stats.get("unit", "")
            print(f"\n  {metric.upper()} ({unit}):")
            print(
                f"    mean={stats.get('mean', 0):.4f}  "
                f"min={stats.get('min', 0):.4f}  "
                f"max={stats.get('max', 0):.4f}"
            )
            pcts = "    " + "  ".join(
                f"p{p}={stats.get(f'p{p}', 0):.4f}" for p in (50, 75, 90, 95, 99)
            )
            print(pcts)

        print(f"{'=' * 60}\n")


def _slugify(name: str) -> str:
    """Convert a display name to a filesystem-safe slug.

    ``"NVIDIA A100-SXM4-80GB"`` → ``"a100-sxm4-80gb"``
    ``"nilay-samora/gemma3-27b-fp8"`` → ``"gemma3-27b-fp8"``
    """
    name = name.lower().replace("nvidia ", "").strip()
    return re.sub(r"[^\w\-.]", "-", name).strip("-")
