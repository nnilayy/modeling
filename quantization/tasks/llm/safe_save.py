from contextlib import contextmanager

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


def cache_num_parameters(model):
    """Pre-compute and cache num_parameters() on the model.

    When a model is partially offloaded to CPU via device_map="auto",
    save_pretrained calls num_parameters() which iterates all parameters,
    triggering compressed_tensors' offload cache to move them back to GPU.
    This OOMs on tight-memory setups. Caching the count avoids the iteration.
    """
    count = sum(p.numel() for p in model.parameters())
    model.num_parameters = lambda *args, **kwargs: count
