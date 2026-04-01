"""
Benchmark runner — loads model + benchmark YAMLs and dispatches to the
configured evaluation framework.

Usage:
    python -m evaluations.tasks.llm.benchmarks.run \
        configs/evaluations/tasks/llm/models/qwen/qwen_3/1.7b/base.yaml \
        configs/evaluations/tasks/llm/bfcl_v4.yaml

    python -m evaluations.tasks.llm.benchmarks.run \
        configs/evaluations/tasks/llm/models/qwen/qwen_3/1.7b/base.yaml \
        configs/evaluations/tasks/llm/bfcl_v4.yaml --port 8001
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def slugify(name: str) -> str:
    name = name.lower().replace("/", "_").strip()
    return re.sub(r"[^\w\-.]", "-", name).strip("-")


def build_evalscope_config(model_cfg: dict, benchmark_cfg: dict) -> dict:
    model = model_cfg["model"]
    server = benchmark_cfg["server"]
    dataset = benchmark_cfg["dataset"]
    evaluation = benchmark_cfg.get("evaluation", {})

    model_name = model["name"]
    temperature = model.get("temperature", 0.0)
    enable_thinking = model.get("enable_thinking", None)

    generation_config: dict = {"temperature": temperature}
    if enable_thinking is not None:
        generation_config["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }

    model_slug = slugify(model_name)
    output_dir = benchmark_cfg.get("output", {}).get("dir", "results/evaluations/llm")
    work_dir = str(Path(output_dir) / model_slug)

    task_cfg: dict = {
        "model": model_name,
        "api_url": server["base_url"],
        "api_key": server.get("api_key", "dummy"),
        "eval_type": "openai_api",
        "datasets": [dataset["name"]],
        "eval_batch_size": evaluation.get("batch_size", 10),
        "generation_config": generation_config,
        "work_dir": work_dir,
    }

    subset = dataset.get("subset")
    if subset:
        task_cfg["dataset_args"] = {
            dataset["name"]: {"subset_list": subset}
        }

    return task_cfg


def run_evalscope(model_cfg: dict, benchmark_cfg: dict) -> None:
    from evalscope.config import TaskConfig
    from evalscope.run import run_task

    task_dict = build_evalscope_config(model_cfg, benchmark_cfg)

    model_name = model_cfg["model"]["name"]
    benchmark_name = benchmark_cfg["benchmark"]["name"]

    print(f"\n{'='*60}")
    print(f"  Benchmark Evaluation")
    print(f"  Model:     {model_name}")
    print(f"  Benchmark: {benchmark_name}")
    print(f"  Framework: evalscope")
    print(f"  Output:    {task_dict['work_dir']}")
    print(f"{'='*60}\n")

    task_cfg = TaskConfig(**task_dict)
    run_task(task_cfg=task_cfg)


FRAMEWORK_DISPATCH = {
    "evalscope": run_evalscope,
}


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("model_yaml", help="Path to model YAML config")
    parser.add_argument("benchmark_yaml", help="Path to benchmark YAML config")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    args = parser.parse_args()

    model_cfg = load_yaml(Path(args.model_yaml))
    benchmark_cfg = load_yaml(Path(args.benchmark_yaml))

    if args.port is not None:
        base_url = benchmark_cfg["server"]["base_url"]
        parts = base_url.rsplit(":", 1)
        path_suffix = ""
        if "/" in parts[-1]:
            port_and_path = parts[-1].split("/", 1)
            path_suffix = "/" + port_and_path[1]
        benchmark_cfg["server"]["base_url"] = f"{parts[0]}:{args.port}{path_suffix}"

    framework = benchmark_cfg["benchmark"]["framework"]
    runner = FRAMEWORK_DISPATCH.get(framework)

    if runner is None:
        print(f"Unknown framework: {framework}")
        print(f"Supported: {list(FRAMEWORK_DISPATCH.keys())}")
        sys.exit(1)

    runner(model_cfg, benchmark_cfg)


if __name__ == "__main__":
    main()
