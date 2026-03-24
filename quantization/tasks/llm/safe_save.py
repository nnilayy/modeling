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
