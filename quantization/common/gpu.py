import gc
import torch
from llmcompressor.core.session_functions import reset_session


def print_gpu_stats(label=""):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - allocated
    print(f"\n{'=' * 50}")
    print(f"  GPU Stats: {label}")
    print(f"{'=' * 50}")
    print(f"  Device:    {torch.cuda.get_device_name(0)}")
    print(f"  Total:     {total:.2f} GB")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Free:      {free:.2f} GB")
    print(f"{'=' * 50}\n")


def clear_gpu():
    reset_session()
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
