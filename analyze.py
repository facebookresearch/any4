from typing import Callable, Dict, List, Optional
from pathlib import Path
import json
import os
import sys
import torch
import transformers
import matplotlib.pyplot as plt
from lm_eval.utils import simple_parse_args_string

from any4 import convert, intq, anyq
from utils import CustomJSONEncoder

def main(
    model_name: str,
    layers: List[int],
    sub_layer: str,
    row: int,
    quant_args: Dict,
    quant_method: Callable,
    model_args: Dict,
    device: str,
    log_dir: Path,
    bnb_args: Optional[Dict] = None,
):
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

    if bnb_args:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_args["quantization_config"] = bnb_config

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_args).to(device)
    modules = []
    names = []
    for name, module in model.named_modules():
        for layer in layers:
            if name == f"model.layers.{layer}.{sub_layer}":
                modules.append(module)
                names.append(name)

    for name, module in zip(names, modules):
        print(f"{name}...", end="", flush=True)
        # Store the weight
        w = module.weight.data.clone()

        # Plot original weight distribution
        counts, bins = torch.histogram(module.weight.data[row].float().cpu(), bins=40)
        plt.figure()
        plt.hist(bins[:-1], bins=bins, weights=counts.float().numpy())
        plt.show()
        plt.savefig(log_dir / "w.png")

        # Apply our quantization algorithms
        if quant_method:
            module = convert(module, layer_from=torch.nn.Linear, layer_to=quant_method, **quant_args)

        # Get mean square error
        wq = module.weight.data
        mse = torch.mean((w - wq)**2)
        print(f"\tMean Square Error: {mse}")
        # TODO: Log to CSV file

        # Overlay quantized values
        for wq_val in module.weight.data[row].unique().float().cpu():
            plt.axvline(x=wq_val, color="b", linestyle="--")

        # Plot reconstructed weight distribution
        counts, bins = torch.histogram(module.weight.data[row].float().cpu(), bins=40)
        plt.figure()
        plt.hist(bins[:-1], bins=bins, weights=counts.float().numpy())
        plt.show()
        plt.savefig(log_dir / "w_reconstructed.png")


if __name__ == '__main__':
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_methods = {
        "intq": intq,
        "anyq": anyq,
    }

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate any4 quantization on various language tasks using lm-evaluation-harness.")

    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace model name or path.")
    parser.add_argument("--model-args", type=str, help="Comma separated string arguments for HuggingFace model.")
    parser.add_argument("--quantize", type=str, choices=quant_methods.keys(), help="Quantization method.")
    parser.add_argument("--quantize-args", type=str, help="Comma separated string args to pass to quantization method.")
    parser.add_argument("--bnb-args", type=str, help="Comma separated string args to pass to BitsAndBytes quantization config.")
    parser.add_argument("--device", type=str, default=default_device, help="Device to use.")
    parser.add_argument("--log-dir", type=Path, default="./logs", help="Directory to log to.")
    parser.add_argument("--layers", type=int, nargs="+", default=0, help="Transformer layer to analyze")
    parser.add_argument("--sub-layer", type=str, choices=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"], default="self_attn.q_proj", help="Linear module within a transformer layer to analyze.")
    parser.add_argument("--row", type=int, default=0, help="Row of weight matrix to analyze.")

    args = parser.parse_args()

    # Pre-process some args
    model_args = {} if not args.model_args else simple_parse_args_string(args.model_args)
    quant_method = None if not args.quantize else quant_methods[args.quantize]
    quant_args = {} if not args.quantize_args else simple_parse_args_string(args.quantize_args)
    bnb_args = None if not args.bnb_args else simple_parse_args_string(args.bnb_args)

    # Run Evaluation
    main(model_name=args.model_name, layers=args.layers, sub_layer=args.sub_layer, row=args.row, model_args=model_args, quant_method=quant_method, quant_args=quant_args, device=args.device, log_dir=args.log_dir, bnb_args=bnb_args)
