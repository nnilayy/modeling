"""
SGLang inference engine — launches an SGLang server from model + engine YAMLs.

Usage:
    python -m inference.tasks.llm.engines.sglang \
        configs/inference/tasks/llm/qwen/qwen_3.5/4b/fp8.yaml
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml


ENGINE_YAML = Path("configs/inference/common/engines/sglang.yaml")


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

    cmd = ["python3", "-m", "sglang.launch_server"]

    cmd += ["--model-path", m["name"]]
    cmd += ["--dtype", m["dtype"]]

    if m.get("quantization"):
        cmd += ["--quantization", m["quantization"]]

    if m.get("revision"):
        cmd += ["--revision", m["revision"]]

    cmd += ["--mem-fraction-static", str(mem["mem_fraction_static"])]

    if perf.get("chunked_prefill_size"):
        cmd += ["--chunked-prefill-size", str(perf["chunked_prefill_size"])]

    if perf.get("schedule_policy"):
        cmd += ["--schedule-policy", perf["schedule_policy"]]

    cmd += ["--tp", str(par["tensor_parallel_size"])]

    cmd += ["--host", srv["host"]]
    cmd += ["--port", str(srv["port"])]

    if srv.get("trust_remote_code"):
        cmd += ["--trust-remote-code"]

    if log.get("log_level"):
        cmd += ["--log-level", log["log_level"]]

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
    raise TimeoutError(f"SGLang server not healthy after {timeout}s at {url}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m inference.tasks.llm.engines.sglang <model.yaml>")
        sys.exit(1)

    model_cfg = load_yaml(Path(sys.argv[1]))
    engine_cfg = load_yaml(ENGINE_YAML)

    cmd = build_command(model_cfg, engine_cfg)

    host = engine_cfg["server"]["host"]
    port = engine_cfg["server"]["port"]

    print(f"\n{'='*60}")
    print(f"  SGLang Server")
    print(f"  Model:   {model_cfg['model']['name']}")
    print(f"  Dtype:   {model_cfg['model']['dtype']}")
    print(f"  Address: http://{host}:{port}")
    print(f"{'='*60}")
    print(f"\n  Command:\n  {' '.join(cmd)}\n")

    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    def shutdown(sig, frame):
        print(f"\nShutting down SGLang server (pid={process.pid})...")
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
