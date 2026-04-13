"""
Shim that patches RULER for:
  1. OpenAI tiktoken tokenizer (when --apply_chat_template is present)
  2. Caching synthetic dataset generation across model runs

Called by run.py for all RULER evaluations.
"""
import functools
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

CACHE_DIR = Path(os.environ.get("RULER_CACHE_DIR", "cache/ruler_datasets"))

# ---------------------------------------------------------------------------
# Dataset generation cache — wraps RULER generation functions so synthetic
# data is built once and reused across model runs.
# ---------------------------------------------------------------------------
def _cache_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(**kwargs):
        cache_key_data = {
            "func": fn.__module__ + "." + fn.__qualname__,
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
        }
        key = hashlib.sha256(
            json.dumps(cache_key_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        cache_path = CACHE_DIR / f"{fn.__name__}_{key}.pkl"

        if cache_path.exists():
            print(f"  [ruler-cache] Loading {fn.__name__} from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        print(f"  [ruler-cache] Generating {fn.__name__} (will cache to {cache_path})")
        result = fn(**kwargs)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        return result
    return wrapper


def _patch_ruler_caching():
    """Wrap all RULER generation functions with disk caching."""
    modules_and_funcs = []

    from lm_eval.tasks.ruler import niah_utils
    for name in dir(niah_utils):
        if name.startswith("niah_"):
            obj = getattr(niah_utils, name)
            if callable(obj):
                modules_and_funcs.append((niah_utils, name, obj))

    try:
        from lm_eval.tasks.ruler import cwe_utils
        for name in ["get_cw_dataset"]:
            if hasattr(cwe_utils, name):
                modules_and_funcs.append((cwe_utils, name, getattr(cwe_utils, name)))
    except ImportError:
        pass

    try:
        from lm_eval.tasks.ruler import fwe_utils
        for name in ["fwe_download"]:
            if hasattr(fwe_utils, name):
                modules_and_funcs.append((fwe_utils, name, getattr(fwe_utils, name)))
    except ImportError:
        pass

    try:
        from lm_eval.tasks.ruler import vt_utils
        for name in ["get_vt_dataset"]:
            if hasattr(vt_utils, name):
                modules_and_funcs.append((vt_utils, name, getattr(vt_utils, name)))
    except ImportError:
        pass

    try:
        from lm_eval.tasks.ruler import qa_utils
        for name in ["get_squad", "get_hotpotqa"]:
            if hasattr(qa_utils, name):
                modules_and_funcs.append((qa_utils, name, getattr(qa_utils, name)))
    except ImportError:
        pass

    patched = 0
    for mod, name, func in modules_and_funcs:
        wrapped = _cache_wrapper(func)
        setattr(mod, name, wrapped)
        patched += 1

    # Ensure short-name imports (used by !function in YAML) also see our patches.
    # lm_eval resolves "!function niah_utils.func" via a relative import that may
    # create a separate sys.modules entry from "lm_eval.tasks.ruler.niah_utils".
    import sys as _sys
    for short, full in [
        ("niah_utils", "lm_eval.tasks.ruler.niah_utils"),
        ("cwe_utils", "lm_eval.tasks.ruler.cwe_utils"),
        ("fwe_utils", "lm_eval.tasks.ruler.fwe_utils"),
        ("vt_utils", "lm_eval.tasks.ruler.vt_utils"),
        ("qa_utils", "lm_eval.tasks.ruler.qa_utils"),
        ("common_utils", "lm_eval.tasks.ruler.common_utils"),
    ]:
        if full in _sys.modules:
            _sys.modules[short] = _sys.modules[full]

    print(f"  [ruler-cache] Patched {patched} generation functions with disk caching")


_patch_ruler_caching()

# ---------------------------------------------------------------------------
# OpenAI tiktoken patch (only when --apply_chat_template is present)
# ---------------------------------------------------------------------------
if "--apply_chat_template" in sys.argv:
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
        from lm_eval.tasks.ruler import qa_utils
        qa_utils.get_tokenizer = _tok_replacement
    except (ImportError, AttributeError):
        pass

# ---------------------------------------------------------------------------
# Hand off to lm_eval CLI
# ---------------------------------------------------------------------------
sys.argv = ["lm_eval"] + sys.argv[1:]

from lm_eval.__main__ import cli_evaluate  # noqa: E402

cli_evaluate()
