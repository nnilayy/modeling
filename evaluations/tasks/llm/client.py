"""
Benchmark client — streams prompts against a running inference server and
collects TTFT, TPOT, and throughput metrics.

Usage:
    python -m evaluations.tasks.llm.client \
        configs/evaluations/tasks/llm/benchmark.yaml \
        --engine vllm
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from evaluations.tasks.llm.metrics import (
    EnvironmentCollector,
    LatencyCollector,
    MetricsCollector,
    RequestStore,
)

config = yaml.safe_load(open(sys.argv[1]))

parser = argparse.ArgumentParser(description="LLM benchmark client")
parser.add_argument("config", help="Path to benchmark YAML config")
parser.add_argument("--engine", required=True, help="Engine name (vllm, sglang, lmdeploy)")
args = parser.parse_args()

base_url = config["server"]["base_url"]
prompt_data = json.load(open(config["data"]["prompts_file"]))
warmup_prompts = prompt_data["warmup"]
prompts = prompt_data["prompts"]

# ── components ──────────────────────────────────────────
store = RequestStore()
env_collector = EnvironmentCollector(base_url=base_url)
latency_collector = LatencyCollector(store)

mc = MetricsCollector(engine=args.engine, store=store)
mc.add(env_collector)
mc.add(latency_collector)

# ── run ─────────────────────────────────────────────────
mc.run(
    base_url=base_url,
    prompts=prompts,
    warmup_prompts=warmup_prompts,
    max_tokens=config["generation"]["max_tokens"],
    temperature=config["generation"].get("temperature", 0.0),
)

# ── output ──────────────────────────────────────────────
mc.summary()
mc.save(config["output"]["dir"])
