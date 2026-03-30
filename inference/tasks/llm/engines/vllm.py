"""
vLLM inference engine — launches a vLLM server from model + engine YAMLs.

Usage:
    python -m inference.tasks.llm.engines.vllm \
        configs/inference/tasks/llm/qwen/qwen3.5/4b.yaml
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml


ENGINE_YAML = Path("configs/inference/common/engines/vllm.yaml")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_command(model_cfg: dict, engine_cfg: dict) -> list[str]:
    m = model_cfg["model"]
    srv = engine_cfg["server"]
    mem = engine_cfg["memory"]
    perf = engine_cfg["performance"]
    par = engine_cfg["parallelism"]
    log = engine_cfg["logging"]

    cmd = ["vllm", "serve", m["name"]]

    cmd += ["--dtype", m["dtype"]]

    if m.get("quantization"):
        cmd += ["--quantization", m["quantization"]]

    if m.get("revision"):
        cmd += ["--revision", m["revision"]]

    if m.get("max_model_len"):
        cmd += ["--max-model-len", str(m["max_model_len"])]

    cmd += ["--gpu-memory-utilization", str(mem["gpu_memory_utilization"])]
    cmd += ["--kv-cache-dtype", mem["kv_cache_dtype"]]

    if mem.get("cpu_offload_gb", 0) > 0:
        cmd += ["--cpu-offload-gb", str(mem["cpu_offload_gb"])]

    if perf.get("enable_prefix_caching"):
        cmd += ["--enable-prefix-caching"]

    if perf.get("enable_chunked_prefill"):
        cmd += ["--enable-chunked-prefill"]

    cmd += ["--max-num-seqs", str(perf["max_num_seqs"])]
    cmd += ["--max-num-batched-tokens", str(perf["max_num_batched_tokens"])]

    cmd += ["--tensor-parallel-size", str(par["tensor_parallel_size"])]

    if par.get("pipeline_parallel_size", 1) > 1:
        cmd += ["--pipeline-parallel-size", str(par["pipeline_parallel_size"])]

    cmd += ["--host", srv["host"]]
    cmd += ["--port", str(srv["port"])]
    cmd += ["--seed", str(srv["seed"])]

    if srv.get("trust_remote_code"):
        cmd += ["--trust-remote-code"]

    tools = engine_cfg.get("tool_calling", {})
    if tools.get("enable_auto_tool_choice"):
        cmd += ["--enable-auto-tool-choice"]
    if tools.get("tool_call_parser"):
        cmd += ["--tool-call-parser", tools["tool_call_parser"]]

    chat_tpl = engine_cfg.get("chat_template", {})
    if chat_tpl:
        import json
        cmd += ["--default-chat-template-kwargs", json.dumps(chat_tpl)]

    if log.get("disable_log_stats"):
        cmd += ["--disable-log-stats"]

    return cmd


def wait_for_healthy(host: str, port: int, timeout: int = 300) -> None:
    url = f"http://{host}:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(3)
    raise TimeoutError(f"vLLM server not healthy after {timeout}s at {url}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m inference.tasks.llm.engines.vllm <model.yaml>")
        sys.exit(1)

    model_cfg = load_yaml(Path(sys.argv[1]))
    engine_cfg = load_yaml(ENGINE_YAML)

    cmd = build_command(model_cfg, engine_cfg)

    host = engine_cfg["server"]["host"]
    port = engine_cfg["server"]["port"]

    print(f"\n{'='*60}")
    print(f"  vLLM Server")
    print(f"  Model:   {model_cfg['model']['name']}")
    print(f"  Dtype:   {model_cfg['model']['dtype']}")
    print(f"  Address: http://{host}:{port}")
    print(f"{'='*60}")
    print(f"\n  Command:\n  {' '.join(cmd)}\n")

    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    def shutdown(sig, frame):
        print(f"\nShutting down vLLM server (pid={process.pid})...")
        process.terminate()
        process.wait(timeout=15)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Waiting for server to be healthy...")
    try:
        wait_for_healthy(host, port)
    except TimeoutError as e:
        print(f"\nERROR: {e}")
        process.terminate()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Server ready at http://{host}:{port}")
    print(f"  OpenAI-compatible endpoint: http://{host}:{port}/v1")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    process.wait()


if __name__ == "__main__":
    main()
