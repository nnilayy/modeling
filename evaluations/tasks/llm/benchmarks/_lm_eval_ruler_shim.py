"""
Shim that patches RULER before handing off to lm_eval's CLI.

Patches applied:
  1. (OpenAI only) Replace get_tokenizer with a tiktoken wrapper.
  2. Override num_samples in all RULER task generators when
     --metadata contains "num_samples".

The key insight: lm_eval calls custom_dataset(**metadata), so num_samples
arrives as a kwarg. But every RULER function ignores it and hardcodes 500.
We fix this by replacing each entry function to forward num_samples.
"""
import json
import sys

# ---------------------------------------------------------------------------
# 1. Parse config from argv
# ---------------------------------------------------------------------------
_num_samples = None
_use_tiktoken = "--apply_chat_template" in sys.argv

for i, arg in enumerate(sys.argv):
    if arg == "--metadata" and i + 1 < len(sys.argv):
        _meta = json.loads(sys.argv[i + 1])
        _num_samples = _meta.get("num_samples")
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
# 3. Patch num_samples
#
# Each RULER YAML does: custom_dataset: !function <module>.<func>
# lm_eval calls that func(**metadata). metadata includes num_samples.
# But the functions hardcode num_samples=500 internally.
#
# Fix: replace each entry function so it pops num_samples from kwargs
# and passes it through to the inner generate call.
# ---------------------------------------------------------------------------
if _num_samples is not None:
    _NS = int(_num_samples)
    print(f"[shim] Patching RULER generators: num_samples={_NS}")

    from lm_eval.tasks.ruler import niah_utils, cwe_utils, fwe_utils, vt_utils, qa_utils
    from lm_eval.tasks.ruler.prepare_niah import generate_samples as _niah_generate, get_haystack
    from lm_eval.tasks.ruler.common_utils import DEFAULT_SEQ_LENGTHS

    NIAH_TEMPLATE = niah_utils.TEMPLATE

    # ---- NIAH tasks: redefine each function to use _NS ----
    def _make_niah(type_haystack, type_needle_k, type_needle_v,
                   num_needle_k=1, num_needle_v=1, num_needle_q=1):
        def fn(**kwargs):
            kwargs.pop("num_samples", None)
            seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
            return niah_utils.download_dataset(
                _niah_generate(
                    get_haystack(type_haystack=type_haystack),
                    max_seq_length=seq,
                    template=NIAH_TEMPLATE,
                    type_haystack=type_haystack,
                    type_needle_k=type_needle_k,
                    type_needle_v=type_needle_v,
                    num_needle_k=num_needle_k,
                    num_needle_v=num_needle_v,
                    num_needle_q=num_needle_q,
                    num_samples=_NS,
                    TOKENIZER=niah_utils.get_tokenizer(**kwargs),
                )
                for seq in seq_lengths
            )
        return fn

    niah_utils.niah_single_1 = _make_niah("repeat", "words", "numbers")
    niah_utils.niah_single_2 = _make_niah("essay", "words", "numbers")
    niah_utils.niah_single_3 = _make_niah("essay", "words", "uuids")
    niah_utils.niah_multikey_1 = _make_niah("essay", "words", "numbers", num_needle_k=4)
    niah_utils.niah_multikey_2 = _make_niah("needle", "words", "numbers")
    niah_utils.niah_multikey_3 = _make_niah("needle", "uuids", "uuids")
    niah_utils.niah_multivalue = _make_niah("essay", "words", "numbers", num_needle_v=4)
    niah_utils.niah_multiquery = _make_niah("essay", "words", "numbers", num_needle_q=4)

    # ---- CWE: cwe_utils.get_cw_dataset -> get_dataset -> sys_word_pair_random(num_samples=500) ----
    _orig_cwe_sys = cwe_utils.sys_word_pair_random

    def _new_cwe_get_dataset(pretrained, seq=None, **kwargs):
        tokenizer = cwe_utils.get_tokenizer(pretrained)
        return _orig_cwe_sys(num_samples=_NS, max_seq_length=seq, tokenizer=tokenizer)

    cwe_utils.get_dataset = _new_cwe_get_dataset

    def _new_cwe_download(**kwargs):
        kwargs.pop("num_samples", None)
        pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", {}))
        import itertools, datasets
        df = (
            _new_cwe_get_dataset(pretrained, seq=seq)
            for seq in kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
        )
        return {
            "test": datasets.Dataset.from_list(
                list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
            )
        }

    cwe_utils.get_cw_dataset = _new_cwe_download

    # ---- FWE: fwe_utils.fwe_download -> get_dataset -> sys_kwext(num_samples=500 default) ----
    _orig_fwe_sys = fwe_utils.sys_kwext

    def _new_fwe_get_dataset(pretrained, max_seq_length=None, **kwargs):
        tokenizer = fwe_utils.get_tokenizer(pretrained)
        return _orig_fwe_sys(tokenizer=tokenizer, max_seq_length=max_seq_length, num_samples=_NS)

    fwe_utils.get_dataset = _new_fwe_get_dataset

    def _new_fwe_download(**kwargs):
        kwargs.pop("num_samples", None)
        pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", {}))
        import itertools, datasets
        df = (
            _new_fwe_get_dataset(pretrained, max_seq_length=seq)
            for seq in kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
        )
        return {
            "test": datasets.Dataset.from_list(
                list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
            )
        }

    fwe_utils.fwe_download = _new_fwe_download

    # ---- VT: vt_utils.get_vt_dataset -> get_dataset -> sys_vartrack(num_samples=500) ----
    _orig_vt_sys = vt_utils.sys_vartrack_w_noise_random

    def _new_vt_get_dataset(tokenizer, seq=None, **kwargs):
        icl_example = _orig_vt_sys(
            tokenizer=tokenizer, num_samples=1, max_seq_length=500, incremental=5,
        )[0]
        return _orig_vt_sys(
            tokenizer=tokenizer, num_samples=_NS, max_seq_length=seq, icl_example=icl_example,
        )

    vt_utils.get_dataset = _new_vt_get_dataset

    def _new_vt_download(**kwargs):
        kwargs.pop("num_samples", None)
        pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", ""))
        import itertools, datasets
        df = (
            _new_vt_get_dataset(tokenizer=vt_utils.get_tokenizer(pretrained), seq=seq)
            for seq in kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
        )
        return {
            "test": datasets.Dataset.from_list(
                list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
            )
        }

    vt_utils.get_vt_dataset = _new_vt_download

    # ---- QA: qa_utils.get_squad / get_hotpotqa -> get_qa_dataset -> get_dataset -> generate_samples(num_samples=500) ----
    _orig_qa_gen = qa_utils.generate_samples

    def _new_qa_get_dataset(pretrained, docs, qas, max_seq_length=None, **kwargs):
        tokenizer = qa_utils.get_tokenizer(pretrained)
        return _orig_qa_gen(
            tokenizer=tokenizer, docs=docs, qas=qas,
            num_samples=_NS, tokens_to_generate=32, max_seq_length=max_seq_length,
        )

    qa_utils.get_dataset = _new_qa_get_dataset

    def _new_qa_dataset(ds, **kwargs):
        kwargs.pop("num_samples", None)
        pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", {}))
        if ds == "squad":
            qas, docs = qa_utils.read_squad()
        else:
            qas, docs = qa_utils.read_hotpotqa()
        import itertools, datasets
        df = (
            _new_qa_get_dataset(pretrained=pretrained, docs=docs, qas=qas, max_seq_length=seq)
            for seq in kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
        )
        return {
            "test": datasets.Dataset.from_list(
                list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
            )
        }

    def _new_get_squad(**kwargs):
        return _new_qa_dataset("squad", **kwargs)

    def _new_get_hotpotqa(**kwargs):
        return _new_qa_dataset("hotpotqa", **kwargs)

    qa_utils.get_qa_dataset = _new_qa_dataset
    qa_utils.get_squad = _new_get_squad
    qa_utils.get_hotpotqa = _new_get_hotpotqa

# ---------------------------------------------------------------------------
# 4. Hand off to lm_eval CLI
# ---------------------------------------------------------------------------
sys.argv = ["lm_eval"] + sys.argv[1:]

from lm_eval.__main__ import cli_evaluate  # noqa: E402

cli_evaluate()
