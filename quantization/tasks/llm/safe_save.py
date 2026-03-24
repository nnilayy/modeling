import json
import os
import re
from contextlib import contextmanager

import torch
import llmcompressor.transformers.compression.compressed_tensors_utils as _ct_utils


@contextmanager
def skip_accelerate_save():
    """Disable to_accelerate/from_accelerate during save to prevent
    incomplete device_map construction for multimodal models.
    See: https://github.com/vllm-project/llm-compressor/issues/1721"""
    orig_to = _ct_utils.to_accelerate
    orig_from = _ct_utils.from_accelerate
    _ct_utils.to_accelerate = lambda m: None
    _ct_utils.from_accelerate = lambda m: None
    try:
        yield
    finally:
        _ct_utils.to_accelerate = orig_to
        _ct_utils.from_accelerate = orig_from


def fix_quantization_config(model, save_directory, ignore_patterns):
    """Expand regex ignore patterns to explicit module names in the saved config.

    llmcompressor may save regex patterns (e.g. 're:vision_tower.*') without
    expanding them, causing vLLM/SGLang to think ignored modules are quantized
    and look for missing weight_scale parameters.
    See: https://github.com/vllm-project/llm-compressor/issues/1306

    Walks the model, matches Linear modules against the ignore patterns using
    the same matching rules as llmcompressor (regex or endswith), and rewrites
    quantization_config.ignore in config.json with the explicit list.
    """
    config_path = os.path.join(save_directory, "config.json")
    if not os.path.exists(config_path):
        return

    expanded = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        for pat in ignore_patterns:
            if pat.startswith("re:"):
                if re.search(pat[3:], name):
                    expanded.append(name)
                    break
            else:
                if name.endswith(pat) or name == pat:
                    expanded.append(name)
                    break

    if not expanded:
        return

    # Newer transformers (4.57+) wraps submodules under self.model, giving
    # names like "model.vision_tower.xxx". vLLM strips this prefix internally,
    # so the ignore list must use names without it to match correctly.
    expanded = [n[len("model."):] if n.startswith("model.") else n for n in expanded]

    with open(config_path, "r") as f:
        config = json.load(f)

    quant_config = config.get("quantization_config")
    if quant_config is None:
        return

    quant_config["ignore"] = sorted(expanded)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Fixed quantization_config.ignore: {len(expanded)} modules explicitly listed")
