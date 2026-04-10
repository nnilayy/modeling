"""
vLLM-Omni TTS engine — launches a vLLM-Omni server from model + engine YAMLs.

Usage:
    python -m inference.tasks.tts.engines.vllm_omni \
        configs/inference/tasks/tts/fish_audio/s2_pro/base.yaml

    python -m inference.tasks.tts.engines.vllm_omni \
        configs/inference/tasks/tts/fish_audio/s2_pro/base.yaml --port 8092
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml


ENGINE_YAML = Path("configs/inference/common/engines/vllm_omni.yaml")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_command(model_cfg: dict, engine_cfg: dict) -> list[str]:
    m = model_cfg["model"]
    srv = engine_cfg["server"]
    mem = engine_cfg["memory"]
    omni = engine_cfg["omni"]
    par = engine_cfg["parallelism"]
    log = engine_cfg["logging"]

    cmd = ["vllm-omni", "serve", m["name"]]

    if m.get("revision"):
        cmd += ["--revision", m["revision"]]

    cmd += ["--omni"]

    if omni.get("enforce_eager"):
        cmd += ["--enforce-eager"]

    if omni.get("stage_configs_path"):
        cmd += ["--stage-configs-path", omni["stage_configs_path"]]

    if omni.get("stage_init_timeout"):
        cmd += ["--stage-init-timeout", str(omni["stage_init_timeout"])]

    if omni.get("tts_max_instructions_length"):
        cmd += ["--tts-max-instructions-length", str(omni["tts_max_instructions_length"])]

    cmd += ["--gpu-memory-utilization", str(mem["gpu_memory_utilization"])]

    cmd += ["--tensor-parallel-size", str(par["tensor_parallel_size"])]

    cmd += ["--host", srv["host"]]
    cmd += ["--port", str(srv["port"])]
    cmd += ["--seed", str(srv["seed"])]

    if srv.get("trust_remote_code"):
        cmd += ["--trust-remote-code"]

    if log.get("disable_log_stats"):
        cmd += ["--disable-log-stats"]

    return cmd


def wait_for_healthy(host: str, port: int, timeout: int = 600) -> None:
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
    raise TimeoutError(f"vLLM-Omni server not healthy after {timeout}s at {url}")


def main():
    parser = argparse.ArgumentParser(description="Launch vLLM-Omni TTS server")
    parser.add_argument("model_yaml", help="Path to model YAML config")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    args = parser.parse_args()

    model_cfg = load_yaml(Path(args.model_yaml))
    engine_cfg = load_yaml(ENGINE_YAML)

    if args.port is not None:
        engine_cfg["server"]["port"] = args.port

    cmd = build_command(model_cfg, engine_cfg)

    host = engine_cfg["server"]["host"]
    port = engine_cfg["server"]["port"]

    print(f"\n{'='*60}")
    print(f"  vLLM-Omni TTS Server")
    print(f"  Model:   {model_cfg['model']['name']}")
    print(f"  Arch:    {model_cfg['metadata']['architecture']}")
    print(f"  Address: http://{host}:{port}")
    print(f"{'='*60}")
    print(f"\n  Command:\n  {' '.join(cmd)}\n")

    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    def shutdown(sig, frame):
        print(f"\nShutting down vLLM-Omni server (pid={process.pid})...")
        process.terminate()
        process.wait(timeout=15)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Waiting for server to be healthy (TTS stages take longer to init)...")
    try:
        wait_for_healthy(host, port)
    except TimeoutError as e:
        print(f"\nERROR: {e}")
        process.terminate()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Server ready at http://{host}:{port}")
    print(f"  TTS endpoint:   http://{host}:{port}/v1/audio/speech")
    print(f"  Voice upload:   http://{host}:{port}/v1/audio/voices")
    print(f"  Health check:   http://{host}:{port}/health")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    process.wait()


if __name__ == "__main__":
    main()
