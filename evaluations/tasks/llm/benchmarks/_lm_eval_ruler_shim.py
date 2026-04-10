"""
Shim that patches RULER's get_tokenizer for OpenAI models (which lack a
HuggingFace tokenizer) then hands off to lm_eval's CLI.

Called by run.py when provider=openai and benchmark=ruler.
"""
import sys

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
_replacement = lambda *args, **kwargs: _wrapper  # noqa: E731

from lm_eval.tasks.ruler import common_utils  # noqa: E402

common_utils.get_tokenizer = _replacement

try:
    from lm_eval.tasks.ruler import qa_utils  # noqa: E402
    qa_utils.get_tokenizer = _replacement
except (ImportError, AttributeError):
    pass

# Strip the shim script path so lm_eval sees clean argv.
sys.argv = ["lm_eval"] + sys.argv[1:]

from lm_eval.__main__ import cli_evaluate  # noqa: E402

cli_evaluate()
