import importlib
import os
import sys
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.awq import AWQModifier
from huggingface_hub import HfApi, login
from quantization.common.config import load_config
from quantization.common.gpu import print_gpu_stats, clear_gpu
from quantization.tasks.llm.safe_save import skip_accelerate_save


def prepare_calibration(tokenizer, calibration_config):
    dataset_name = calibration_config.get("dataset", "HuggingFaceH4/ultrachat_200k")
    split = calibration_config.get("split", "train_sft")
    num_samples = calibration_config.get("num_samples", 512)
    max_sequence_length = calibration_config.get("max_seq_length", 2048)
    seed = calibration_config.get("seed", 42)

    dataset = load_dataset(dataset_name, split=f"{split}[:{num_samples}]")
    dataset = dataset.shuffle(seed=seed)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )
        }

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )

    dataset = dataset.map(preprocess)
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    return dataset


def build_recipe(recipe_config):
    method = recipe_config["method"]
    ignore_layers = recipe_config.get("ignore", ["lm_head"])
    targets = recipe_config.get("targets", "Linear")
    scheme = recipe_config["scheme"]
    if method == "fp8":
        return QuantizationModifier(targets=targets, scheme=scheme, ignore=ignore_layers)

    if method == "int8":
        smoothquant_config = recipe_config.get("smoothquant", {})
        return [
            SmoothQuantModifier(smoothing_strength=smoothquant_config.get("smoothing_strength", 0.8)),
            GPTQModifier(targets=targets, scheme=scheme, ignore=ignore_layers),
        ]

    if method == "gptq_int4":
        return GPTQModifier(targets=targets, scheme=scheme, ignore=ignore_layers)

    if method == "awq_int4":
        return [AWQModifier(targets=[targets], scheme=scheme, ignore=ignore_layers)]

    raise ValueError(f"Unknown method: {method}")


def run(config_path):
    load_dotenv("quantization/.env")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN not found. Set via 'export HF_TOKEN=...' or add to quantization/.env")

    login(token=hf_token)

    config = load_config(config_path)

    model_config = config["model"]
    recipe_config = config["recipe"]
    calibration_config = config.get("calibration", {})
    offload_config = config.get("offload", {})
    save_config = config.get("save", {})

    base_model = model_config["base"]
    output_repo = model_config["output"]
    method = recipe_config["method"]
    ignore_layers = recipe_config.get("ignore", ["lm_head"])

    print(f"Base:   {base_model}")
    print(f"Output: {output_repo}")
    print(f"Method: {method}")
    print(f"Ignore: {ignore_layers}")

    print_gpu_stats("Before loading")

    dtype = getattr(torch, model_config.get("dtype", "bfloat16"))

    is_multimodal = model_config.get("multimodal", False)

    model_class_name = model_config["model_class"]   
    transformers_module = importlib.import_module("transformers")
    model_class = getattr(transformers_module, model_class_name)
    model = model_class.from_pretrained(base_model, device_map="auto", torch_dtype=dtype)
    print(f"Loaded with {model_class_name} (device_map=auto)")

    if is_multimodal:
        processor = AutoProcessor.from_pretrained(base_model)
        tokenizer = processor.tokenizer
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    print_gpu_stats("After loading")

    calibration_dataset = None
    if calibration_config.get("enabled", False):
        calibration_dataset = prepare_calibration(tokenizer, calibration_config)
        print(f"Calibration: {len(calibration_dataset)} samples, max_seq_length={calibration_config.get('max_seq_length', 2048)}")

    recipe = build_recipe(recipe_config)

    oneshot_kwargs = {"model": model, "recipe": recipe}

    if calibration_dataset is not None:
        oneshot_kwargs["dataset"] = calibration_dataset
        oneshot_kwargs["max_seq_length"] = calibration_config.get("max_seq_length", 2048)
        oneshot_kwargs["num_calibration_samples"] = calibration_config.get("num_samples", 512)

    save_directory = output_repo.split("/")[-1]
    oneshot_kwargs["output_dir"] = save_directory

    sequential_device = offload_config.get("sequential_device")
    if sequential_device:
        oneshot_kwargs["sequential_offload_device"] = sequential_device

    with skip_accelerate_save():
        oneshot(**oneshot_kwargs)

    print_gpu_stats("After quantization")

    if processor is not None:
        processor.save_pretrained(save_directory)
    else:
        tokenizer.save_pretrained(save_directory)

    api = HfApi()
    api.create_repo(output_repo, exist_ok=True)
    api.upload_folder(folder_path=save_directory, repo_id=output_repo)

    print(f"\nDone! Pushed to https://huggingface.co/{output_repo}")

    clear_gpu()
    print_gpu_stats("After cleanup")


if __name__ == "__main__":
    run(sys.argv[1])
