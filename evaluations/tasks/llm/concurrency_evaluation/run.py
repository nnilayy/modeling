"""
Concurrency evaluation — finds optimal max_concurrency per context length.
Uses vLLM's built-in sweep tool to automatically explore concurrency levels.

Usage:
    python -m evaluations.tasks.llm.concurrency_evaluation.run \
        configs/evaluations/tasks/llm/models/qwen/qwen_3/32b/fp8.yaml \
        configs/evaluations/tasks/llm/concurrency.yaml

    python -m evaluations.tasks.llm.concurrency_evaluation.run \
        configs/evaluations/tasks/llm/models/qwen/qwen_3/32b/fp8.yaml \
        configs/evaluations/tasks/llm/concurrency.yaml --port 8001
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


ENGINE_YAML = Path("configs/inference/common/engines/vllm.yaml")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def slugify(name: str) -> str:
    name = name.lower().replace("/", "_").strip()
    return re.sub(r"[^\w\-.]", "-", name).strip("-")


def build_serve_cmd(
    model_cfg: dict, engine_cfg: dict, context_length: int, port: int
) -> str:
    """Build the `vllm serve ...` command string.

    max-model-len is set to the current context_length being tested,
    NOT the model's full 131072, so the KV cache is sized correctly
    for each sweep iteration.
    """
    m = model_cfg["model"]
    mem = engine_cfg["memory"]
    srv = engine_cfg["server"]
    par = engine_cfg["parallelism"]

    parts = [
        "vllm", "serve", m["name"],
        "--dtype", m.get("dtype", "auto"),
        "--max-model-len", str(context_length),
        "--gpu-memory-utilization", str(mem["gpu_memory_utilization"]),
        "--kv-cache-dtype", mem["kv_cache_dtype"],
        "--tensor-parallel-size", str(par["tensor_parallel_size"]),
        "--port", str(port),
        "--seed", str(srv["seed"]),
    ]

    if m.get("rope_scaling"):
        hf_overrides = {
            "rope_scaling": m["rope_scaling"],
            "max_position_embeddings": context_length,
        }
        parts += ["--hf-overrides", json.dumps(hf_overrides)]
        parts += ["--enforce-eager"]

    if engine_cfg["performance"].get("enable_prefix_caching"):
        parts += ["--enable-prefix-caching"]

    if srv.get("trust_remote_code"):
        parts += ["--trust-remote-code"]

    return shlex.join(parts)


def build_bench_cmd(
    model_name: str,
    context_length: int,
    num_prompts: int,
    output_tokens: int,
    port: int,
) -> str:
    """Build the `vllm bench serve ...` command string.

    Uses the 'random' dataset so no external data is needed.
    vLLM treats max-model-len as the TOTAL budget (input + output),
    so random-input-len = context_length - output_tokens to leave
    room for the requested output. output_tokens is kept minimal
    (1) since we only care about concurrency capacity, not
    generation throughput.
    """
    parts = [
        "vllm", "bench", "serve",
        "--model", model_name,
        "--base-url", f"http://localhost:{port}",
        "--dataset-name", "random",
        "--random-input-len", str(context_length - output_tokens),
        "--random-output-len", str(output_tokens),
        "--num-prompts", str(num_prompts),
    ]

    return shlex.join(parts)


def run_sweep(
    serve_cmd: str,
    bench_cmd: str,
    output_dir: str,
    workload_var: str,
    workload_iters: int,
    num_runs: int,
) -> int:
    """Invoke `vllm bench sweep serve_workload`.

    The sweep tool handles the full lifecycle:
      1. Starts the vLLM server using serve_cmd
      2. Runs bench_cmd at varying max_concurrency levels
      3. Shuts down the server
      4. Writes per-iteration JSONs + summary.csv to output_dir
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "vllm", "bench", "sweep", "serve_workload",
        "--serve-cmd", serve_cmd,
        "--bench-cmd", bench_cmd,
        "--workload-var", workload_var,
        "--workload-iters", str(workload_iters),
        "--num-runs", str(num_runs),
        "--output-dir", output_dir,
    ]

    env = None
    if "enforce-eager" in serve_cmd or "hf-overrides" in serve_cmd:
        import os
        env = os.environ.copy()
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    result = subprocess.run(cmd, env=env)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal max_concurrency per context length"
    )
    parser.add_argument("model_yaml", help="Path to model YAML config")
    parser.add_argument("concurrency_yaml", help="Path to concurrency YAML config")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    model_cfg = load_yaml(Path(args.model_yaml))
    concurrency_cfg = load_yaml(Path(args.concurrency_yaml))
    engine_cfg = load_yaml(ENGINE_YAML)

    model_name = model_cfg["model"]["name"]
    model_slug = slugify(model_name)

    evaluation = concurrency_cfg["evaluation"]
    context_lengths = evaluation["context_lengths"]
    num_prompts = evaluation["num_prompts"]
    output_tokens = evaluation["output_tokens"]
    workload_var = evaluation["workload_var"]
    workload_iters = evaluation["workload_iters"]
    num_runs = evaluation["num_runs"]
    base_output_dir = concurrency_cfg["output"]["dir"]

    print(f"\n{'='*60}")
    print(f"  Concurrency Evaluation")
    print(f"  Model:           {model_name}")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Num prompts:     {num_prompts}")
    print(f"  Output tokens:   {output_tokens}")
    print(f"  Workload var:    {workload_var}")
    print(f"  Workload iters:  {workload_iters}")
    print(f"  Num runs:        {num_runs}")
    print(f"  Output:          {base_output_dir}/{model_slug}/")
    print(f"{'='*60}\n")

    failed = []

    for i, ctx in enumerate(context_lengths, 1):
        print(f"\n[{i}/{len(context_lengths)}] Context length: {ctx}")
        print(f"{'-'*40}")

        serve_cmd = build_serve_cmd(model_cfg, engine_cfg, ctx, args.port)
        bench_cmd = build_bench_cmd(
            model_name, ctx, num_prompts, output_tokens, args.port
        )
        output_dir = str(Path(base_output_dir) / model_slug / f"ctx_{ctx}")

        print(f"  Serve:  {serve_cmd[:100]}...")
        print(f"  Bench:  {bench_cmd[:100]}...")
        print(f"  Output: {output_dir}")
        print()

        rc = run_sweep(
            serve_cmd, bench_cmd, output_dir,
            workload_var, workload_iters, num_runs,
        )

        if rc != 0:
            print(f"  WARNING: Sweep exited with code {rc} for ctx={ctx}")
            failed.append(ctx)

    print(f"\n{'='*60}")
    print(f"  Concurrency evaluation complete.")
    print(f"  Results: {base_output_dir}/{model_slug}/")
    if failed:
        print(f"  Failed context lengths: {failed}")
    else:
        print(f"  All {len(context_lengths)} context lengths succeeded.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
