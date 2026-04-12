"""
Shim that patches RULER before handing off to lm_eval's CLI.

Patches applied:
  1. (OpenAI only) Replace get_tokenizer with a tiktoken wrapper.
  2. Override num_samples in all RULER task generators when
     --metadata contains "num_samples".

Called by run.py for all RULER evaluations.
"""
import json
import sys

# ---------------------------------------------------------------------------
# 1. Parse num_samples and openai flag from argv metadata
# ---------------------------------------------------------------------------
_num_samples = None
_use_tiktoken = "--apply_chat_template" in sys.argv

for i, arg in enumerate(sys.argv):
    if arg == "--metadata" and i + 1 < len(sys.argv):
        _meta = json.loads(sys.argv[i + 1])
        _num_samples = _meta.pop("num_samples", None)
        sys.argv[i + 1] = json.dumps(_meta)
        break

# ---------------------------------------------------------------------------
# 2. OpenAI tiktoken patch
# ---------------------------------------------------------------------------
if _use_tiktoken:
    import tiktoken

    class _TiktokenWrapper:
        def __init__(self, encoding_name: str = "o200k_base"):
            self._enc = tiktoken.get_encoding(encoding_name)

        def __call__(self, text: str):
            class _Result:
                def __init__(self, input_ids):
                    self.input_ids = input_ids
            return _Result(self._enc.encode(text))

    _wrapper = _TiktokenWrapper("o200k_base")
    _tok_replacement = lambda *args, **kwargs: _wrapper  # noqa: E731

    from lm_eval.tasks.ruler import common_utils
    common_utils.get_tokenizer = _tok_replacement

    try:
        from lm_eval.tasks.ruler import qa_utils as _qa
        _qa.get_tokenizer = _tok_replacement
    except (ImportError, AttributeError):
        pass

# ---------------------------------------------------------------------------
# 3. Patch num_samples in every RULER generator
# ---------------------------------------------------------------------------
if _num_samples is not None:
    from lm_eval.tasks.ruler import niah_utils, cwe_utils, fwe_utils, vt_utils, qa_utils
    import functools

    _NS = int(_num_samples)

    # --- niah_utils: 8 functions that call generate_samples(... num_samples=500) ---
    _orig_niah_gen = niah_utils.generate_samples

    @functools.wraps(_orig_niah_gen)
    def _patched_niah_gen(*args, **kwargs):
        kwargs["num_samples"] = _NS
        return _orig_niah_gen(*args, **kwargs)

    niah_utils.generate_samples = _patched_niah_gen

    # --- cwe_utils: sys_word_pair_random(num_samples=500, ...) ---
    _orig_cwe_gen = cwe_utils.sys_word_pair_random

    @functools.wraps(_orig_cwe_gen)
    def _patched_cwe_gen(*args, **kwargs):
        kwargs["num_samples"] = _NS
        return _orig_cwe_gen(*args, **kwargs)

    cwe_utils.sys_word_pair_random = _patched_cwe_gen

    # --- fwe_utils: sys_kwext(... num_samples=500) ---
    _orig_fwe_gen = fwe_utils.sys_kwext

    @functools.wraps(_orig_fwe_gen)
    def _patched_fwe_gen(*args, **kwargs):
        kwargs["num_samples"] = _NS
        return _orig_fwe_gen(*args, **kwargs)

    fwe_utils.sys_kwext = _patched_fwe_gen

    # --- vt_utils: sys_vartrack_w_noise_random(... num_samples=500) ---
    _orig_vt_gen = vt_utils.sys_vartrack_w_noise_random

    @functools.wraps(_orig_vt_gen)
    def _patched_vt_gen(*args, **kwargs):
        kwargs["num_samples"] = _NS
        return _orig_vt_gen(*args, **kwargs)

    vt_utils.sys_vartrack_w_noise_random = _patched_vt_gen

    # --- qa_utils: get_dataset(... num_samples=500) and generate_samples(... num_samples=500) ---
    _orig_qa_get = qa_utils.get_dataset

    @functools.wraps(_orig_qa_get)
    def _patched_qa_get(*args, **kwargs):
        kwargs["num_samples"] = _NS
        return _orig_qa_get(*args, **kwargs)

    qa_utils.get_dataset = _patched_qa_get

    _orig_qa_gen = qa_utils.generate_samples

    @functools.wraps(_orig_qa_gen)
    def _patched_qa_gen(*args, **kwargs):
        kwargs["num_samples"] = _NS
        return _orig_qa_gen(*args, **kwargs)

    qa_utils.generate_samples = _patched_qa_gen

# ---------------------------------------------------------------------------
# 4. Hand off to lm_eval CLI
# ---------------------------------------------------------------------------
sys.argv = ["lm_eval"] + sys.argv[1:]

from lm_eval.__main__ import cli_evaluate  # noqa: E402

cli_evaluate()
