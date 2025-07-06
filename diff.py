# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Callable, Optional, Type
import json
import os
import pickle
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from pathlib import Path

from utils import CustomJSONEncoder
from lm_eval.utils import simple_parse_args_string
from quantize import quantize_model, quant_methods
from pre_process.pre_quant import pre_quant_methods
from calibrate import calibrate
from utils import remove_all_hooks, dtype_str_to_torch

# TODO: This also exists in calibrate.py. move to one file?
default_prompt = """This is a diverse prompt that contains:
                - Fiction: "Once upon a time, a girl named Alice was living alone on an island. One day, she met a wizard ..."
                - News: "The United Nations held its General Assembly meeting this year amid multiple world crises and wars. In his speech, the General Secretary called for ..."
                - Code: `public static void main(String[] args) {\n System.out.println("Hello world!);\n}`
                - Math: (5.2 + 2.7) / 0.6 - 1.9 * 2.2 =
                - Facts: "The capital of Egypt is Cairo. It is the largest city in the region and is home to..."
                """

layer_to_output = {}

def get_output(name):
    def hook(model, input, output):
        layer_to_output[name] = output
    return hook

def register_forward_hook(model: torch.nn.Module, layer_type: Type = torch.nn.Linear, **kwargs):
    for name, module in model.named_modules():
        if isinstance(module, (layer_type)):
            module.register_forward_hook(get_output(name))

    return model

def main(
    model_name: str,
    model_args: Dict,
    device: str,
    quant_args: Dict,
    quant_method: Callable,
    pre_quant_method: Optional[Callable] = None,
    pre_quant_args: Optional[Dict] = {},
    prompt: str = default_prompt,
    calibrate_args: Dict = {},
    log_dir: Path = "./diffs",
):
    global layer_to_output

    log_dir.mkdir(parents=True, exist_ok=True)
    # Log args
    args = locals()
    print(args)
    with Path(log_dir/"args.json").open("w") as f:
        json.dump(args, f, indent=4, cls=CustomJSONEncoder)

    # Log command to a file
    arg_str = ' '.join([arg.replace("'", "'\\''") for arg in sys.argv[1:]])
    with open(log_dir / "command_line.txt", "w") as f:
        f.write(f"python {os.path.basename(__file__)} {arg_str}\n")

    # Pre-process Args
    if model_args:
        if "torch_dtype" in model_args:
            model_args["torch_dtype"] = dtype_str_to_torch[model_args["torch_dtype"]]

    # Setup Model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_args, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Calibrate Original Model
    layer_to_output = {}
    register_forward_hook(model)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    model(inputs)
    layer_to_output_orig = layer_to_output.copy()
    remove_all_hooks(model)

    # Apply our quantization algorithms
    if quant_args:
        if "sample_weight" in quant_args:
            if quant_args["sample_weight"].endswith(".pickle"):
                with open(quant_args["sample_weight"], 'rb') as handle:
                   quant_args["sample_weight"] = pickle.load(handle)
            elif quant_args["sample_weight"].endswith(".pt"):
                quant_args["sample_weight"] = torch.load(quant_args["sample_weight"], map_location=torch.device(model.device))
            elif quant_args["sample_weight"] == "calibrate":
                quant_args["sample_weight"] = calibrate
    if pre_quant_method:
        model = pre_quant_method(model, tokenizer, **pre_quant_args)
    if quant_method:
        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        model = quantize_model(model, layer_from=torch.nn.Linear, layer_to=quant_method, tokenizer=tokenizer, calibrate_args=calibrate_args, **quant_args)

    # Calibrate Quantized Model
    layer_to_output = {}
    register_forward_hook(model)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    model(inputs)
    layer_to_output_quantized = layer_to_output.copy()
    remove_all_hooks(model)

    # Log Differences
    layers_diffs = []
    for layer_name in layer_to_output_orig.keys():
        layer_diffs = {}
        layer_diffs["layer"] = layer_name

        y_orig = layer_to_output_orig[layer_name]
        y_quant = layer_to_output_quantized[layer_name]
        layer_diffs["mean_square_error"] = torch.mean((y_orig - y_quant) ** 2).item()
        layer_diffs["cosine_similarity"] = torch.nn.functional.cosine_similarity(y_orig.flatten(), y_quant.flatten(), dim=0).item()

        layers_diffs.append(layer_diffs)

    # Log stats
    df = pd.DataFrame(layers_diffs)
    df.to_csv(log_dir / "diffs.csv", index=False)


if __name__ == '__main__':
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate any4 quantization on various language tasks using lm-evaluation-harness.")

    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace model name or path.")
    parser.add_argument("--model-args", type=str, help="Comma separated string arguments for HuggingFace model.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use.")
    parser.add_argument("--prompt", type=str, default=default_prompt, help="Prompt to apply.")
    parser.add_argument("--quantize", type=str, choices=quant_methods.keys(), help="Quantization method.")
    parser.add_argument("--quantize-args", type=str, help="Comma separated string args to pass to quantization method.")
    parser.add_argument("--pre-quantize", type=str, choices=pre_quant_methods.keys(), help="Quantization pre-processing method.")
    parser.add_argument("--pre-quantize-args", type=str, help="Comma separated string args to pass to quantization pre-process function.")
    parser.add_argument("--calibrate-args", type=str, help="Comma separated string args to pass to calibration function.")
    parser.add_argument("--log-dir", type=Path, default="./diffs", help="Directory to log to.")

    args = parser.parse_args()

    # Pre-process some args
    quant_method = None if not args.quantize else quant_methods[args.quantize]
    quant_args = {} if not args.quantize_args else simple_parse_args_string(args.quantize_args)
    pre_quant_method = None if not args.pre_quantize else pre_quant_methods[args.pre_quantize]
    pre_quant_args = {} if not args.pre_quantize_args else simple_parse_args_string(args.pre_quantize_args)
    calibrate_args = {} if not args.calibrate_args else simple_parse_args_string(args.calibrate_args)
    model_args = {} if not args.model_args else simple_parse_args_string(args.model_args)

    # Run Evaluation
    main(
        model_name=args.model_name,
        model_args=model_args,
        device=args.device,
        prompt=args.prompt,
        quant_method=quant_method,
        quant_args=quant_args,
        pre_quant_method=pre_quant_method,
        pre_quant_args=pre_quant_args,
        calibrate_args=calibrate_args,
        log_dir=args.log_dir
    )
