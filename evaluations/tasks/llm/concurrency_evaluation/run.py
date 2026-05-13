"""Concurrency capacity evaluation for vLLM.

For each context-length bucket in the HF dataset, this script:
  1. starts a vLLM server with max_model_len sized for that bucket
  2. fires all prompts simultaneously via asyncio.gather
  3. polls /metrics at high frequency to capture peak num_requests_running
  4. snapshots /metrics at start and end to compute counter/histogram deltas
  5. writes per-bucket summary.json + requests.jsonl + metrics_timeseries.jsonl
  6. shuts the server down before moving to the next bucket

Usage:
    python -m evaluations.tasks.llm.concurrency_evaluation.run \
        configs/evaluations/tasks/llm/models/qwen/qwen_3/32b/fp8.yaml \
        configs/evaluations/tasks/llm/concurrency.yaml

    # smoke test on a single bucket
    python -m evaluations.tasks.llm.concurrency_evaluation.run \
        configs/evaluations/tasks/llm/models/qwen/qwen_3/32b/fp8.yaml \
        configs/evaluations/tasks/llm/concurrency.yaml \
        --buckets 01k

    # dry run — print serve commands without starting servers
    python -m evaluations.tasks.llm.concurrency_evaluation.run \
        configs/evaluations/tasks/llm/models/qwen/qwen_3/32b/fp8.yaml \
        configs/evaluations/tasks/llm/concurrency.yaml \
        --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from datasets import load_dataset


# =============================================================================
# Config
# =============================================================================

ENGINE_YAML = Path("configs/inference/common/engines/vllm.yaml")
DEFAULT_PORT = 8000

BUCKET_TOKENS: dict[str, int] = {
    "01k": 1024,
    "02k": 2048,
    "04k": 4096,
    "08k": 8192,
    "16k": 16384,
    "32k": 32768,
}


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def slugify(name: str) -> str:
    name = name.lower().replace("/", "-").strip()
    return re.sub(r"[^\w\-.]", "-", name).strip("-")


# =============================================================================
# Prometheus exposition parser
# =============================================================================

_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>\S+)"
)
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_metrics(text: str) -> dict[tuple[str, frozenset], float]:
    """Parse Prometheus exposition text into a flat dict.

    Key   = (full_metric_name_including_suffix, frozenset of label items)
    Value = float
    """
    out: dict[tuple[str, frozenset], float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        labels = dict(_LABEL_RE.findall(m.group("labels") or ""))
        out[(m.group("name"), frozenset(labels.items()))] = value
    return out


def get_gauge(metrics: dict, name: str) -> float | None:
    """Return the first matching gauge value (single-engine assumption)."""
    for (n, _), v in metrics.items():
        if n == name:
            return v
    return None


def get_counter(metrics: dict, name: str) -> float:
    """Sum a counter named `<name>_total` across all label sets (per-engine)."""
    full = f"{name}_total"
    total = 0.0
    for (n, _), v in metrics.items():
        if n == full:
            total += v
    return total


def get_counter_by_label(
    metrics: dict, name: str, label_key: str
) -> dict[str, float]:
    """Return {label_value: summed value} for `<name>_total{label_key=...}`."""
    full = f"{name}_total"
    out: dict[str, float] = {}
    for (n, lbls), v in metrics.items():
        if n != full:
            continue
        d = dict(lbls)
        if label_key in d:
            out[d[label_key]] = out.get(d[label_key], 0.0) + v
    return out


def get_histogram(metrics: dict, name: str) -> dict | None:
    """Aggregate histogram across label sets (per-engine).

    Returns {"sum": float, "count": float, "buckets": [(le, cum_count)]}.
    """
    sum_v: float | None = None
    count_v: float | None = None
    bucket_acc: dict[float, float] = {}
    for (n, lbls), v in metrics.items():
        if n == f"{name}_sum":
            sum_v = (sum_v or 0.0) + v
        elif n == f"{name}_count":
            count_v = (count_v or 0.0) + v
        elif n == f"{name}_bucket":
            le_str = dict(lbls).get("le")
            if le_str is None:
                continue
            le = float("inf") if le_str == "+Inf" else float(le_str)
            bucket_acc[le] = bucket_acc.get(le, 0.0) + v
    if sum_v is None and count_v is None and not bucket_acc:
        return None
    return {
        "sum": sum_v or 0.0,
        "count": count_v or 0.0,
        "buckets": sorted(bucket_acc.items(), key=lambda x: x[0]),
    }


def histogram_diff(end: dict | None, start: dict | None) -> dict | None:
    if end is None or start is None:
        return None
    start_map = dict(start.get("buckets", []))
    diff_buckets = [
        (le, max(0.0, c - start_map.get(le, 0.0)))
        for le, c in end.get("buckets", [])
    ]
    return {
        "sum": max(0.0, end["sum"] - start["sum"]),
        "count": max(0.0, end["count"] - start["count"]),
        "buckets": diff_buckets,
    }


def histogram_quantile(hist: dict | None, q: float) -> float | None:
    """Linear interpolation across cumulative bucket counts."""
    if hist is None or not hist.get("buckets"):
        return None
    buckets = hist["buckets"]
    total = buckets[-1][1] if buckets else 0
    if total <= 0:
        return None
    target = q * total
    prev_le = 0.0
    prev_count = 0.0
    for le, count in buckets:
        if count >= target:
            if le == float("inf"):
                return prev_le
            if count == prev_count:
                return prev_le if prev_le > 0 else le
            frac = (target - prev_count) / (count - prev_count)
            return prev_le + frac * (le - prev_le)
        prev_le = le
        prev_count = count
    return prev_le


def hist_percentiles(hist: dict | None, qs=(0.5, 0.9, 0.95, 0.99)) -> dict:
    if hist is None:
        return {f"p{int(q * 100)}": None for q in qs}
    return {f"p{int(q * 100)}": histogram_quantile(hist, q) for q in qs}


# =============================================================================
# vLLM server lifecycle
# =============================================================================

def build_serve_cmd(
    model_cfg: dict,
    engine_cfg: dict,
    max_model_len: int,
    port: int,
) -> list[str]:
    m = model_cfg["model"]
    srv = engine_cfg["server"]
    mem = engine_cfg["memory"]
    perf = engine_cfg["performance"]
    par = engine_cfg["parallelism"]

    parts: list[str] = [
        "vllm", "serve", m["name"],
        "--dtype", str(m.get("dtype", "auto")),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(perf["max_num_seqs"]),
        "--max-num-batched-tokens", str(perf["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(mem["gpu_memory_utilization"]),
        "--kv-cache-dtype", str(mem["kv_cache_dtype"]),
        "--tensor-parallel-size", str(par["tensor_parallel_size"]),
        "--pipeline-parallel-size", str(par["pipeline_parallel_size"]),
        "--host", str(srv["host"]),
        "--port", str(port),
        "--seed", str(srv["seed"]),
    ]

    if perf.get("enable_prefix_caching"):
        parts += ["--enable-prefix-caching"]
    if perf.get("enable_chunked_prefill"):
        parts += ["--enable-chunked-prefill"]
    if perf.get("async_scheduling"):
        parts += ["--async-scheduling"]
    if srv.get("trust_remote_code"):
        parts += ["--trust-remote-code"]
    if mem.get("cpu_offload_gb", 0):
        parts += ["--cpu-offload-gb", str(mem["cpu_offload_gb"])]

    spec = engine_cfg.get("speculative_decoding") or {}
    if spec.get("enabled"):
        method = spec.get("method", "ngram_gpu")
        spec_cfg: dict[str, Any] = {
            "method": method,
            "num_speculative_tokens": int(spec.get("num_speculative_tokens", 5)),
        }
        if method in ("ngram", "ngram_gpu"):
            spec_cfg["prompt_lookup_max"] = int(spec.get("prompt_lookup_max", 4))
            spec_cfg["prompt_lookup_min"] = int(spec.get("prompt_lookup_min", 2))
        elif method in ("eagle", "eagle3"):
            if spec.get("eagle_model_path"):
                spec_cfg["model"] = spec["eagle_model_path"]
        parts += ["--speculative-config", json.dumps(spec_cfg)]

    return parts


def _ensure_port_free(port: int, max_wait_s: float = 30.0) -> None:
    """Kill anything bound to `port` and wait until the port is free.

    Stops stale vLLM processes from previous runs hijacking the new server's
    /v1/models endpoint and producing fake "server ready in 0.0s" results.
    """
    import socket as _socket

    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
            sock.close()
            return
        except OSError:
            sock.close()
            killed = False
            for cmd in (
                ["fuser", "-k", "-n", "tcp", str(port)],
                ["lsof", "-ti", f"tcp:{port}"],
            ):
                try:
                    out = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=5
                    )
                    if cmd[0] == "lsof" and out.stdout.strip():
                        for pid in out.stdout.strip().splitlines():
                            try:
                                os.kill(int(pid), signal.SIGKILL)
                                killed = True
                            except (ProcessLookupError, ValueError, OSError):
                                pass
                    if cmd[0] == "fuser" and out.returncode == 0:
                        killed = True
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            if killed:
                time.sleep(1.0)
            else:
                time.sleep(0.5)


async def wait_for_ready(
    base_url: str,
    timeout_s: float,
    interval_s: float,
    proc: subprocess.Popen | None = None,
    expected_model: str | None = None,
) -> bool:
    """Poll /v1/models until OUR process is serving the EXPECTED model.

    Guards against stale leftover servers responding instantly with the wrong
    model or wrong max_model_len.
    """
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                return False
            try:
                r = await client.get(f"{base_url}/v1/models", timeout=5.0)
                if r.status_code == 200:
                    if expected_model is None:
                        return True
                    try:
                        served = {
                            m.get("id") for m in r.json().get("data", [])
                        }
                    except Exception:
                        served = set()
                    if expected_model in served:
                        return True
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
            ):
                pass
            await asyncio.sleep(interval_s)
    return False


def stop_server(proc: subprocess.Popen, timeout_s: float = 30.0) -> None:
    """SIGTERM the process group; escalate to SIGKILL if it overstays."""
    if proc.poll() is not None:
        return
    pgid: int | None = None
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        pgid = None

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=10.0)
        except Exception:
            pass
    except (ProcessLookupError, OSError):
        pass


# =============================================================================
# Async load generator
# =============================================================================

async def send_one(
    client: httpx.AsyncClient,
    model_name: str,
    messages: list[dict],
    request_id: str,
    bucket: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    chat_template_kwargs: dict | None = None,
) -> dict:
    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs

    t_send = time.monotonic()
    t_first: float | None = None
    t_last: float | None = None
    finish_reason: str | None = None
    completion_tokens = 0
    prompt_tokens = 0
    output_chars = 0
    status = "ok"
    error: str | None = None

    try:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=30.0),
        ) as resp:
            if resp.status_code != 200:
                status = "http_error"
                try:
                    body_text = await resp.aread()
                    error = (
                        f"HTTP {resp.status_code}: "
                        f"{body_text.decode(errors='replace')[:200]}"
                    )
                except Exception:
                    error = f"HTTP {resp.status_code}"
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = int(usage.get("prompt_tokens") or 0)
                        completion_tokens = int(
                            usage.get("completion_tokens") or 0
                        )
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            if t_first is None:
                                t_first = time.monotonic()
                            output_chars += len(content)
                        fr = choice.get("finish_reason")
                        if fr is not None:
                            finish_reason = fr
                t_last = time.monotonic()
    except (httpx.TimeoutException, asyncio.TimeoutError):
        status = "timeout"
        error = f"timeout after {timeout_s}s"
    except (
        httpx.ConnectError,
        httpx.ReadError,
        httpx.RemoteProtocolError,
    ) as e:
        status = "transport_error"
        error = repr(e)
    except Exception as e:
        status = "error"
        error = repr(e)

    if status == "ok":
        if completion_tokens == 0:
            status = "empty_completion"
        elif finish_reason not in ("stop", "length"):
            status = "aborted_by_server"

    e2e_s = (t_last - t_send) if t_last else None
    ttft_s = (t_first - t_send) if t_first else None
    tpot_s = None
    if t_first is not None and t_last is not None and completion_tokens > 1:
        tpot_s = (t_last - t_first) / (completion_tokens - 1)

    return {
        "request_id": request_id,
        "bucket": bucket,
        "status": status,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_chars": output_chars,
        "ttft_s": ttft_s,
        "e2e_s": e2e_s,
        "tpot_s": tpot_s,
        "t_send": t_send,
        "t_first_token": t_first,
        "t_last_token": t_last,
        "error": error,
    }


# =============================================================================
# Async metrics poller
# =============================================================================

async def metrics_poller(
    client: httpx.AsyncClient,
    interval_s: float,
    samples: list[dict],
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        t_poll = time.monotonic()
        try:
            r = await client.get("/metrics", timeout=2.0)
            if r.status_code == 200:
                m = parse_metrics(r.text)
                samples.append({
                    "t": t_poll,
                    "num_requests_running": (
                        get_gauge(m, "vllm:num_requests_running") or 0
                    ),
                    "num_requests_waiting": (
                        get_gauge(m, "vllm:num_requests_waiting") or 0
                    ),
                    "kv_cache_usage_perc": (
                        get_gauge(m, "vllm:kv_cache_usage_perc") or 0
                    ),
                })
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def fetch_metrics(client: httpx.AsyncClient) -> dict | None:
    try:
        r = await client.get("/metrics", timeout=10.0)
        if r.status_code != 200:
            return None
        return parse_metrics(r.text)
    except Exception:
        return None


# =============================================================================
# Aggregation
# =============================================================================

def _peak(samples: list[dict], key: str) -> float:
    return max((s[key] for s in samples), default=0.0)


def _mean(samples: list[dict], key: str) -> float:
    if not samples:
        return 0.0
    return sum(s[key] for s in samples) / len(samples)


def aggregate(
    per_request_rows: list[dict],
    gauge_samples: list[dict],
    snap_start: dict | None,
    snap_end: dict | None,
    model_name: str,
    bucket: str,
    max_model_len: int,
    max_num_seqs: int,
    num_prompts_fired: int,
    wall_seconds: float,
) -> dict:
    fulfilled_client = sum(1 for r in per_request_rows if r["status"] == "ok")
    by_status: dict[str, int] = {}
    for r in per_request_rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    server: dict[str, Any] = {}
    if snap_start is not None and snap_end is not None:
        success_start = get_counter_by_label(
            snap_start, "vllm:request_success", "finished_reason"
        )
        success_end = get_counter_by_label(
            snap_end, "vllm:request_success", "finished_reason"
        )
        success_delta = {
            k: max(0.0, success_end.get(k, 0.0) - success_start.get(k, 0.0))
            for k in set(success_start) | set(success_end)
        }

        def cd(name: str) -> float:
            return max(
                0.0,
                get_counter(snap_end, name) - get_counter(snap_start, name),
            )

        server = {
            "request_success_delta": success_delta,
            "preemptions_delta": cd("vllm:num_preemptions"),
            "prompt_tokens_delta": cd("vllm:prompt_tokens"),
            "generation_tokens_delta": cd("vllm:generation_tokens"),
            "prompt_tokens_recomputed_delta": cd(
                "vllm:prompt_tokens_recomputed"
            ),
            "prefix_cache_hits_delta": cd("vllm:prefix_cache_hits"),
            "prefix_cache_queries_delta": cd("vllm:prefix_cache_queries"),
            "corrupted_requests_delta": cd("vllm:corrupted_requests"),
        }

        hist_specs = [
            ("e2e", "vllm:e2e_request_latency_seconds"),
            ("queue_time", "vllm:request_queue_time_seconds"),
            ("prefill_time", "vllm:request_prefill_time_seconds"),
            ("decode_time", "vllm:request_decode_time_seconds"),
            ("inference_time", "vllm:request_inference_time_seconds"),
            ("ttft", "vllm:time_to_first_token_seconds"),
            ("inter_token_latency", "vllm:inter_token_latency_seconds"),
            (
                "time_per_output_token",
                "vllm:request_time_per_output_token_seconds",
            ),
        ]
        latency: dict[str, dict] = {}
        for label, full in hist_specs:
            d = histogram_diff(
                get_histogram(snap_end, full),
                get_histogram(snap_start, full),
            )
            entry = {
                "count": d["count"] if d else 0.0,
                "sum_s": d["sum"] if d else 0.0,
            }
            entry.update(hist_percentiles(d))
            latency[label] = entry
        server["latency_seconds"] = latency

    server_fulfilled = 0
    if "request_success_delta" in server:
        server_fulfilled = int(
            server["request_success_delta"].get("stop", 0.0)
            + server["request_success_delta"].get("length", 0.0)
        )
    match = "ok" if fulfilled_client == server_fulfilled else "mismatch"

    peak_running = int(_peak(gauge_samples, "num_requests_running"))
    waves_estimate = (
        math.ceil(num_prompts_fired / peak_running) if peak_running > 0 else 0
    )
    throughput_req_per_s = (
        round(fulfilled_client / wall_seconds, 2) if wall_seconds > 0 else 0.0
    )
    avg_wave_seconds = (
        round(wall_seconds / waves_estimate, 2) if waves_estimate > 0 else 0.0
    )

    return {
        "bucket": bucket,
        "model": model_name,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "num_prompts_fired": num_prompts_fired,
        "wall_seconds": round(wall_seconds, 3),
        "throughput": {
            "req_per_s": throughput_req_per_s,
            "waves_estimate": waves_estimate,
            "avg_wave_seconds": avg_wave_seconds,
        },
        "concurrency": {
            "peak_running": peak_running,
            "mean_running": round(
                _mean(gauge_samples, "num_requests_running"), 2
            ),
            "peak_waiting": int(_peak(gauge_samples, "num_requests_waiting")),
            "mean_waiting": round(
                _mean(gauge_samples, "num_requests_waiting"), 2
            ),
            "samples": len(gauge_samples),
        },
        "memory": {
            "peak_kv_cache_usage": round(
                _peak(gauge_samples, "kv_cache_usage_perc"), 4
            ),
            "mean_kv_cache_usage": round(
                _mean(gauge_samples, "kv_cache_usage_perc"), 4
            ),
        },
        "outcomes": {
            "fulfilled_client": fulfilled_client,
            "fulfilled_server_total": server_fulfilled,
            "match_check": match,
            "by_status": by_status,
            "request_success_by_finish_reason": server.get(
                "request_success_delta", {}
            ),
            "preemptions": int(server.get("preemptions_delta", 0)),
        },
        "sanity": {
            "prefix_cache_hits": int(server.get("prefix_cache_hits_delta", 0)),
            "prefix_cache_queries": int(
                server.get("prefix_cache_queries_delta", 0)
            ),
            "prompt_tokens_recomputed": int(
                server.get("prompt_tokens_recomputed_delta", 0)
            ),
            "corrupted_requests": int(
                server.get("corrupted_requests_delta", 0)
            ),
        },
        "tokens": {
            "prompt_tokens": int(server.get("prompt_tokens_delta", 0)),
            "generation_tokens": int(server.get("generation_tokens_delta", 0)),
        },
        "latency_server_seconds": server.get("latency_seconds", {}),
    }


# =============================================================================
# Per-bucket runner
# =============================================================================

async def run_bucket(
    bucket: str,
    model_cfg: dict,
    engine_cfg: dict,
    concurrency_cfg: dict,
    output_dir: Path,
    port: int,
    dry_run: bool,
) -> dict | None:
    model_name = model_cfg["model"]["name"]
    base_url = f"http://localhost:{port}"

    if bucket not in BUCKET_TOKENS:
        print(f"  ERROR: unknown bucket '{bucket}', valid={list(BUCKET_TOKENS)}")
        return None

    max_model_len = int(concurrency_cfg["server"]["max_model_len"])

    serve_cmd = build_serve_cmd(model_cfg, engine_cfg, max_model_len, port)
    print(f"\n[bucket={bucket}] max_model_len={max_model_len}")

    if dry_run:
        print(f"  serve cmd: {shlex.join(serve_cmd)}")
        return None

    print(f"  loading dataset bucket {bucket} from HuggingFace...")
    ds = load_dataset(
        concurrency_cfg["dataset"]["hf_repo"], bucket, split="train"
    )
    num_prompts = int(concurrency_cfg["load"]["num_prompts"])
    rows = list(ds.select(range(min(num_prompts, len(ds)))))
    print(f"  loaded {len(rows)} prompts")

    bucket_dir = output_dir / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)

    print("  starting vllm serve...")
    _ensure_port_free(port)
    log_path = bucket_dir / "vllm_server.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        serve_cmd,
        env=os.environ.copy(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    try:
        startup_timeout = float(concurrency_cfg["server"]["startup_timeout_s"])
        ready_interval = float(
            concurrency_cfg["server"]["ready_poll_interval_s"]
        )
        t_start = time.monotonic()
        ready = await wait_for_ready(
            base_url,
            startup_timeout,
            ready_interval,
            proc=proc,
            expected_model=model_name,
        )
        if not ready:
            if proc.poll() is not None:
                print(
                    f"  ERROR: vllm serve exited with code {proc.returncode} "
                    f"during startup — see {log_path}"
                )
            else:
                print(
                    f"  ERROR: server failed to become ready in "
                    f"{startup_timeout}s — see {log_path}"
                )
            return None
        print(f"  server ready in {time.monotonic() - t_start:.1f}s")

        chat_template_cfg = engine_cfg.get("chat_template") or {}
        ctk: dict | None = None
        if "enable_thinking" in chat_template_cfg:
            ctk = {"enable_thinking": chat_template_cfg["enable_thinking"]}

        req_cfg = concurrency_cfg["request"]
        max_tokens = int(req_cfg["max_completion_tokens"])
        temperature = float(req_cfg.get("temperature", 0.0))
        timeout_s = float(req_cfg["per_request_timeout_s"])

        limits = httpx.Limits(
            max_connections=max(num_prompts + 16, 256),
            max_keepalive_connections=max(num_prompts + 16, 256),
        )
        async with httpx.AsyncClient(base_url=base_url, limits=limits) as client:
            warmup_count = int(concurrency_cfg["load"].get("warmup_requests", 1))
            for i in range(warmup_count):
                row = rows[i % len(rows)]
                print(f"  warmup {i + 1}/{warmup_count}...")
                wres = await send_one(
                    client,
                    model_name=model_name,
                    messages=row["messages"],
                    request_id=f"warmup_{i}",
                    bucket=bucket,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    chat_template_kwargs=ctk,
                )
                print(
                    f"    status={wres['status']} "
                    f"ttft={wres['ttft_s']} e2e={wres['e2e_s']}"
                )

            print("  scraping initial /metrics snapshot...")
            snap_start = await fetch_metrics(client)

            gauge_samples: list[dict] = []
            stop_event = asyncio.Event()
            poll_interval = float(
                concurrency_cfg["server"]["metrics_poll_interval_s"]
            )
            poller_task = asyncio.create_task(
                metrics_poller(client, poll_interval, gauge_samples, stop_event)
            )

            print(f"  firing {len(rows)} concurrent requests...")
            t_fire = time.monotonic()
            tasks = [
                send_one(
                    client,
                    model_name=model_name,
                    messages=row["messages"],
                    request_id=row.get("id", f"{bucket}_{i:03d}"),
                    bucket=bucket,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    chat_template_kwargs=ctk,
                )
                for i, row in enumerate(rows)
            ]
            per_request_rows = await asyncio.gather(*tasks)
            wall_s = time.monotonic() - t_fire

            stop_event.set()
            await poller_task

            snap_end = await fetch_metrics(client)
            print(
                f"  completed in {wall_s:.1f}s, captured "
                f"{len(gauge_samples)} gauge samples"
            )

            summary = aggregate(
                per_request_rows=per_request_rows,
                gauge_samples=gauge_samples,
                snap_start=snap_start,
                snap_end=snap_end,
                model_name=model_name,
                bucket=bucket,
                max_model_len=max_model_len,
                max_num_seqs=int(engine_cfg["performance"]["max_num_seqs"]),
                num_prompts_fired=len(rows),
                wall_seconds=wall_s,
            )

            (bucket_dir / "summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            with (bucket_dir / "requests.jsonl").open("w") as f:
                for r in per_request_rows:
                    f.write(json.dumps(r) + "\n")
            with (bucket_dir / "metrics_timeseries.jsonl").open("w") as f:
                for s in gauge_samples:
                    f.write(json.dumps(s) + "\n")

            c = summary["concurrency"]
            o = summary["outcomes"]
            mm = summary["memory"]
            tp = summary["throughput"]
            print(
                f"  RESULT: peak_running={c['peak_running']} "
                f"peak_waiting={c['peak_waiting']} "
                f"peak_kv={mm['peak_kv_cache_usage']:.2f} "
                f"fulfilled={o['fulfilled_client']}/"
                f"{summary['num_prompts_fired']} "
                f"preempt={o['preemptions']} "
                f"match={o['match_check']}"
            )
            print(
                f"  THROUGHPUT: wall={summary['wall_seconds']}s "
                f"throughput={tp['req_per_s']} req/s "
                f"waves~{tp['waves_estimate']} "
                f"avg_wave={tp['avg_wave_seconds']}s"
            )

            return summary
    finally:
        print("  stopping vllm serve...")
        t_stop = time.monotonic()
        stop_server(
            proc,
            timeout_s=float(
                concurrency_cfg["server"].get("shutdown_timeout_s", 30)
            ),
        )
        _ensure_port_free(port)
        try:
            log_file.close()
        except Exception:
            pass
        print(f"  server stopped ({time.monotonic() - t_stop:.1f}s)")


# =============================================================================
# Top-level summary
# =============================================================================

def write_top_summary(output_dir: Path, summaries: list[dict]) -> None:
    if not summaries:
        return
    csv_path = output_dir / "summary.csv"
    headers = [
        "bucket", "max_model_len", "max_num_seqs", "num_prompts_fired",
        "peak_running", "mean_running", "peak_waiting", "mean_waiting",
        "peak_kv_cache_usage", "fulfilled_client", "fulfilled_server_total",
        "match_check", "preemptions", "prefix_cache_hits",
        "prompt_tokens_recomputed", "wall_seconds",
        "throughput_req_per_s", "waves_estimate", "avg_wave_seconds",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for s in summaries:
            w.writerow([
                s["bucket"],
                s["max_model_len"],
                s["max_num_seqs"],
                s["num_prompts_fired"],
                s["concurrency"]["peak_running"],
                s["concurrency"]["mean_running"],
                s["concurrency"]["peak_waiting"],
                s["concurrency"]["mean_waiting"],
                s["memory"]["peak_kv_cache_usage"],
                s["outcomes"]["fulfilled_client"],
                s["outcomes"]["fulfilled_server_total"],
                s["outcomes"]["match_check"],
                s["outcomes"]["preemptions"],
                s["sanity"]["prefix_cache_hits"],
                s["sanity"]["prompt_tokens_recomputed"],
                s["wall_seconds"],
                s["throughput"]["req_per_s"],
                s["throughput"]["waves_estimate"],
                s["throughput"]["avg_wave_seconds"],
            ])
    print(f"\n  Top-level summary: {csv_path}")


# =============================================================================
# Main
# =============================================================================

async def amain(args: argparse.Namespace) -> int:
    model_cfg = load_yaml(Path(args.model_yaml))
    concurrency_cfg = load_yaml(Path(args.concurrency_yaml))
    engine_cfg = load_yaml(ENGINE_YAML)

    model_name = model_cfg["model"]["name"]
    model_slug = slugify(model_name)

    all_buckets = list(concurrency_cfg["dataset"]["buckets"])
    if args.buckets:
        wanted = [b.strip() for b in args.buckets.split(",") if b.strip()]
        unknown = [b for b in wanted if b not in all_buckets]
        if unknown:
            print(
                f"ERROR: unknown buckets {unknown}. "
                f"Valid: {all_buckets}",
                file=sys.stderr,
            )
            return 2
        buckets = wanted
    else:
        buckets = all_buckets

    base_output = Path(args.output_dir or concurrency_cfg["output"]["dir"])
    output_dir = base_output / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  Concurrency Capacity Evaluation")
    print(f"  Model:    {model_name}")
    print(f"  Buckets:  {buckets}")
    print(
        f"  Prompts:  {concurrency_cfg['load']['num_prompts']} fired "
        "concurrently per bucket"
    )
    print(f"  Output:   {output_dir}/")
    print(f"  Dry run:  {args.dry_run}")
    print(f"{'=' * 60}")

    summaries: list[dict] = []
    failed: list[str] = []

    for i, bucket in enumerate(buckets, 1):
        print(f"\n[{i}/{len(buckets)}]", end=" ")
        try:
            summary = await run_bucket(
                bucket=bucket,
                model_cfg=model_cfg,
                engine_cfg=engine_cfg,
                concurrency_cfg=concurrency_cfg,
                output_dir=output_dir,
                port=args.port,
                dry_run=args.dry_run,
            )
            if summary is None and not args.dry_run:
                failed.append(bucket)
            elif summary is not None:
                summaries.append(summary)
        except KeyboardInterrupt:
            print("\n  INTERRUPTED — exiting cleanly")
            return 130
        except Exception as e:
            print(f"\n  ERROR running bucket {bucket}: {e!r}")
            failed.append(bucket)

    if not args.dry_run:
        write_top_summary(output_dir, summaries)

    print(f"\n{'=' * 60}")
    print("  DONE")
    if failed:
        print(f"  Failed buckets: {failed}")
    print(f"  Results: {output_dir}/")
    print(f"{'=' * 60}\n")

    return 0 if not failed else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="vLLM concurrency capacity evaluation"
    )
    parser.add_argument("model_yaml", help="Path to model YAML config")
    parser.add_argument("concurrency_yaml", help="Path to concurrency YAML config")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="vLLM serve port"
    )
    parser.add_argument(
        "--buckets",
        default=None,
        help="Comma-separated bucket names to run (default: all from yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print serve commands without starting servers",
    )
    args = parser.parse_args()

    try:
        rc = asyncio.run(amain(args))
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
