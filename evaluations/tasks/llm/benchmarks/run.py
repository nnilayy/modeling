"""
Benchmark runner — starts vLLM server, runs evaluation, shuts down server.
End to end, one command.

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
import atexit
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml


ENGINE_YAML = Path("configs/inference/common/engines/vllm.yaml")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def slugify(name: str) -> str:
    name = name.lower().replace("/", "_").strip()
    return re.sub(r"[^\w\-.]", "-", name).strip("-")


def build_vllm_command(
    model_cfg: dict, engine_cfg: dict, port: int, benchmark_cfg: dict | None = None
) -> list[str]:
    m = model_cfg["model"]
    srv = engine_cfg["server"]
    mem = engine_cfg["memory"]
    perf = engine_cfg["performance"]
    par = engine_cfg["parallelism"]
    overrides = (benchmark_cfg or {}).get("vllm_overrides", {})

    cmd = ["vllm", "serve", m["name"], "--dtype", m.get("dtype", "auto")]

    if m.get("max_model_len"):
        cmd += ["--max-model-len", str(m["max_model_len"])]

    if m.get("rope_scaling"):
        rs = m["rope_scaling"]
        for key, val in rs.items():
            cmd += [f"--rope-scaling.{key}", str(val)]

    cmd += ["--gpu-memory-utilization", str(mem["gpu_memory_utilization"])]
    cmd += ["--kv-cache-dtype", mem["kv_cache_dtype"]]

    if perf.get("enable_prefix_caching"):
        cmd += ["--enable-prefix-caching"]
    if perf.get("enable_chunked_prefill"):
        cmd += ["--enable-chunked-prefill"]

    max_num_seqs = overrides.get("max_num_seqs", perf["max_num_seqs"])
    cmd += ["--max-num-seqs", str(max_num_seqs)]
    cmd += ["--max-num-batched-tokens", str(perf["max_num_batched_tokens"])]
    cmd += ["--tensor-parallel-size", str(par["tensor_parallel_size"])]

    cmd += ["--host", srv["host"], "--port", str(port)]
    cmd += ["--seed", str(srv["seed"])]

    if srv.get("trust_remote_code"):
        cmd += ["--trust-remote-code"]

    tools = engine_cfg.get("tool_calling", {})
    if tools.get("enable_auto_tool_choice"):
        cmd += ["--enable-auto-tool-choice"]
    if tools.get("tool_call_parser"):
        cmd += ["--tool-call-parser", tools["tool_call_parser"]]

    return cmd


def wait_for_healthy(host: str, port: int, timeout: int = 1200) -> None:
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


def start_server(
    model_cfg: dict, port: int, benchmark_cfg: dict | None = None
) -> tuple[subprocess.Popen, Path]:
    engine_cfg = load_yaml(ENGINE_YAML)
    cmd = build_vllm_command(model_cfg, engine_cfg, port, benchmark_cfg)
    host = engine_cfg["server"]["host"]

    log_path = Path(f"logs/vllm_server_{port}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")

    print(f"  Starting vLLM server on port {port}...")
    print(f"  Server logs → {log_path}")

    process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)

    _server_state["process"] = process
    _server_state["log_file"] = log_file
    atexit.register(_atexit_cleanup)

    try:
        wait_for_healthy(host, port)
    except TimeoutError:
        print(f"\n  ERROR: vLLM server failed to start. Check {log_path}")
        shutdown_server(process, log_file)
        sys.exit(1)

    print(f"  Server ready at http://{host}:{port}\n")
    return process, log_path


_server_state: dict = {}


def _atexit_cleanup() -> None:
    proc = _server_state.get("process")
    lf = _server_state.get("log_file")
    if proc and lf:
        shutdown_server(proc, lf)


def shutdown_server(process: subprocess.Popen, log_file) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
    if not log_file.closed:
        log_file.close()
    _server_state.clear()



def build_evalscope_config(
    model_cfg: dict, benchmark_cfg: dict, api_url: str
) -> dict:
    model = model_cfg["model"]
    dataset = benchmark_cfg["dataset"]
    evaluation = benchmark_cfg.get("evaluation", {})

    model_name = model["name"]
    temperature = model.get("temperature", 0.0)
    enable_thinking = model.get("enable_thinking", None)

    generation_config: dict = {"temperature": temperature}
    max_tokens = evaluation.get("max_tokens")
    if max_tokens:
        generation_config["max_tokens"] = max_tokens
    if enable_thinking is not None:
        generation_config["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }

    model_slug = slugify(model_name)
    output_dir = benchmark_cfg.get("output", {}).get("dir", "results/evaluations/llm")
    work_dir = str(Path(output_dir) / model_slug)

    task_cfg: dict = {
        "model": model_name,
        "api_url": api_url,
        "api_key": model_cfg.get("server", {}).get("api_key", "dummy"),
        "eval_type": "openai_api",
        "datasets": [dataset["name"]],
        "eval_batch_size": evaluation.get("batch_size", 10),
        "generation_config": generation_config,
        "work_dir": work_dir,
        "no_timestamp": True,
    }

    if Path(work_dir).exists():
        task_cfg["use_cache"] = work_dir

    subset = dataset.get("subset")
    if subset:
        task_cfg["dataset_args"] = {
            dataset["name"]: {"subset_list": subset}
        }

    return task_cfg


def run_evalscope(model_cfg: dict, benchmark_cfg: dict, api_url: str) -> None:
    from evalscope.config import TaskConfig
    from evalscope.run import run_task

    task_dict = build_evalscope_config(model_cfg, benchmark_cfg, api_url)

    model_name = model_cfg["model"]["name"]
    benchmark_name = benchmark_cfg["benchmark"]["name"]

    print(f"  Running {benchmark_name} evaluation...")
    print(f"  Output → {task_dict['work_dir']}")
    if task_dict.get("use_cache"):
        print(f"  Cache  → {task_dict['use_cache']}")
    print()

    task_cfg = TaskConfig(**task_dict)
    run_task(task_cfg=task_cfg)


class _TiktokenWrapper:
    """Wraps tiktoken to match the HuggingFace tokenizer(text).input_ids interface
    that RULER expects for token counting during prompt generation."""

    def __init__(self, encoding_name: str = "o200k_base"):
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding_name)

    def __call__(self, text: str):
        ids = self._enc.encode(text)

        class _Result:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        return _Result(ids)


def _patch_ruler_tokenizer_for_openai() -> None:
    """Replace RULER's get_tokenizer with a tiktoken-backed wrapper so that
    OpenAI models (which have no HuggingFace tokenizer) can be evaluated."""
    from lm_eval.tasks.ruler import common_utils

    wrapper = _TiktokenWrapper("o200k_base")
    _replacement = lambda *args, **kwargs: wrapper
    common_utils.get_tokenizer = _replacement

    # qa_utils imports get_tokenizer at module level, so we must also patch
    # it there (the name is already bound in that module's namespace).
    try:
        from lm_eval.tasks.ruler import qa_utils
        qa_utils.get_tokenizer = _replacement
    except (ImportError, AttributeError):
        pass


def run_lm_eval(model_cfg: dict, benchmark_cfg: dict, api_url: str) -> None:
    model_name = model_cfg["model"]["name"]
    evaluation = benchmark_cfg.get("evaluation", {})
    is_hosted = model_cfg["model"].get("hosted", False)
    provider = model_cfg.get("metadata", {}).get("provider", "")

    if provider == "openai":
        _patch_ruler_tokenizer_for_openai()
        model_type = "local-chat-completions"
        completions_url = api_url.rstrip("/") + "/chat/completions"
    else:
        model_type = "local-completions"
        completions_url = api_url.rstrip("/") + "/completions"

    model_args = (
        f"model={model_name}"
        f",base_url={completions_url}"
        f",num_concurrent=16"
        f",max_retries=5"
        f",tokenized_requests=False"
    )

    if provider != "openai":
        model_args += f",tokenizer={model_name}"

    api_key = model_cfg.get("server", {}).get("api_key", "")
    if api_key and api_key != "dummy":
        model_args += f",api_key={api_key}"

    tasks = evaluation.get("tasks", "ruler")
    batch_size = str(evaluation.get("batch_size", 32))
    max_seq_lengths = evaluation.get("max_seq_lengths", [4096])
    metadata = json.dumps({"max_seq_lengths": max_seq_lengths})

    output_dir = benchmark_cfg.get("output", {}).get("dir", "results/evaluations/llm/ruler")
    output_path = str(Path(output_dir) / slugify(model_name))

    benchmark_name = benchmark_cfg["benchmark"]["name"]
    print(f"  Running {benchmark_name} evaluation...")
    print(f"  Tasks:  {tasks}")
    print(f"  Lengths: {max_seq_lengths}")
    print(f"  Output: {output_path}")
    print()

    cli_args = [
        "--model", model_type,
        "--model_args", model_args,
        "--tasks", tasks,
        "--metadata", metadata,
        "--batch_size", batch_size,
        "--output_path", output_path,
    ]

    if provider == "openai":
        cli_args += ["--apply_chat_template"]

    if provider == "openai" and "ruler" in tasks:
        # lm_eval runs as a subprocess, so in-memory patches don't survive.
        # Use the launcher shim that patches RULER's tokenizer then calls lm_eval.
        shim = str(Path(__file__).parent / "_lm_eval_ruler_shim.py")
        cmd = [sys.executable, shim] + cli_args
        result = subprocess.run(cmd)
    else:
        cmd = [sys.executable, "-m", "lm_eval"] + cli_args
        result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  ERROR: lm_eval exited with code {result.returncode}")
        sys.exit(result.returncode)


FRAMEWORK_DISPATCH = {
    "evalscope": run_evalscope,
    "lm_eval": run_lm_eval,
}


def parse_port_from_url(url: str) -> int:
    parsed = urlparse(url)
    return parsed.port or 8000


def replace_port_in_url(url: str, port: int) -> str:
    parsed = urlparse(url)
    replaced = parsed._replace(netloc=f"{parsed.hostname}:{port}")
    return replaced.geturl()


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("model_yaml", help="Path to model YAML config")
    parser.add_argument("benchmark_yaml", help="Path to benchmark YAML config")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    parser.add_argument("--api-key", type=str, default=None, help="Override API key (for hosted models)")
    args = parser.parse_args()

    model_cfg = load_yaml(Path(args.model_yaml))
    benchmark_cfg = load_yaml(Path(args.benchmark_yaml))

    server_cfg = model_cfg.get("server", {})
    base_url = server_cfg.get("base_url", "http://0.0.0.0:8000/v1")
    is_hosted = model_cfg["model"].get("hosted", False)

    if is_hosted:
        api_url = base_url
        port = None
    else:
        port = args.port or parse_port_from_url(base_url)
        api_url = replace_port_in_url(base_url, port)

    if args.api_key:
        server_cfg["api_key"] = args.api_key

    framework = benchmark_cfg["benchmark"]["framework"]
    runner = FRAMEWORK_DISPATCH.get(framework)
    if runner is None:
        print(f"Unknown framework: {framework}")
        print(f"Supported: {list(FRAMEWORK_DISPATCH.keys())}")
        sys.exit(1)

    model_name = model_cfg["model"]["name"]
    benchmark_name = benchmark_cfg["benchmark"]["name"]

    print(f"\n{'='*60}")
    print(f"  Benchmark Evaluation")
    print(f"  Model:     {model_name}")
    print(f"  Benchmark: {benchmark_name}")
    print(f"  Framework: {framework}")
    print(f"  Server:    {api_url}")
    if is_hosted:
        print(f"  Mode:      hosted (no local server)")
    print(f"{'='*60}\n")

    if is_hosted:
        runner(model_cfg, benchmark_cfg, api_url)
    else:
        process, log_path = start_server(model_cfg, port, benchmark_cfg)
        try:
            runner(model_cfg, benchmark_cfg, api_url)
        finally:
            print(f"\n  Shutting down vLLM server (pid={process.pid})...")
            shutdown_server(process, _server_state.get("log_file", open(log_path, "a")))
            print(f"  Done.\n")


if __name__ == "__main__":
    main()
