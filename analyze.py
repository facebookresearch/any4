from typing import Callable, Dict, List, Optional
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import pickle
import sys
import torch
import transformers
import lm_eval
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from lm_eval.utils import simple_parse_args_string

from any4 import convert, quant_methods
from utils import CustomJSONEncoder

def main(
    model_name: str,
    layers: List[int],
    sub_layers: List[str],
    row: int,
    quant_args: Dict,
    quant_method: Callable,
    model_args: Dict,
    device: str,
    log_dir: Path,
    bnb_args: Optional[Dict] = None,
    bs: int = 10,
    parallelize: bool = True,
    calib_activations_file: Optional[Dict] = None,
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

    # Create PDF for logs
    pdf = PdfPages(log_dir / "plots.pdf")

    if bnb_args:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_args["quantization_config"] = bnb_config

    lm_obj = lm_eval.models.huggingface.HFLM(pretrained=model_name, device=device, batch_size=bs, parallelize=parallelize, **model_args)
    model = lm_obj._model

    # Load calibration activations
    calib_activations = None
    if calib_activations_file:
        if calib_activations_file.endswith(".pickle"):
            with open(calib_activations_file, 'rb') as handle:
                calib_activations = pickle.load(handle)
        elif calib_activations_file.endswith(".pt"):
            calib_activations = torch.load(calib_activations_file, map_location=torch.device(device))

    # Create list of modules
    if layers is None:
        layers = np.arange(0, len(model.model.layers))
    modules = []
    names = []
    layers_stats = []
    for name, module in model.named_modules():
        for layer in layers:
            for sub_layer in sub_layers:
                if name == f"model.layers.{layer}.{sub_layer}":
                    modules.append(module)
                    names.append(name)
                    layers_stats.append({})

    # Analyze each module
    for name, module, layer_stats in zip(names, modules, layers_stats):
        print(f"{name}", end="", flush=True)
        layer_stats["layer"] = name

        # Store the weight
        w = module.weight.data.clone()

        # Apply on random inputs
        x_uni = torch.rand(size=(bs, module.in_features), device=w.device, dtype=w.dtype)
        y_uni = module(x_uni)
        x_norm = torch.randn(size=(bs, module.in_features), device=w.device, dtype=w.dtype)
        y_norm = module(x_norm)
        if calib_activations:
            x_calib = calib_activations[name].to(w.device).to(w.dtype)
            y_calib = module(x_calib)

        # Plot original weight surface
        fig = plot_surface(w, name)
        fig.savefig(pdf, format="pdf")

        # Plot original weight distribution
        counts, bins = torch.histogram(module.weight.data[row].float().cpu(), bins=40)
        plt.figure()
        plt.hist(bins[:-1], bins=bins, weights=counts.float().numpy())
        plt.show()
        plt.title(f"w_{name}_{row}")
        plt.savefig(pdf, format="pdf")
        plt.close()

        # Apply our quantization algorithms
        if quant_method:
            module = convert(module, layer_from=torch.nn.Linear, layer_to=quant_method, **quant_args)

        # Get mean square error
        wdeq = module.weight.data
        y_uni_deq = module(x_uni)
        y_norm_deq = module(x_norm)
        w_mse = torch.mean((w - wdeq)**2)
        y_uni_mse = torch.mean((y_uni - y_uni_deq)**2)
        y_norm_mse = torch.mean((y_norm - y_norm_deq)**2)
        layer_stats["w_mse"] = w_mse.item()
        layer_stats["y_uni_mse"] = y_uni_mse.item()
        layer_stats["y_norm_mse"] = y_norm_mse.item()
        print(f"\tMean Square Error: Weight:{w_mse}, Output: Uniform: {y_uni_mse} Normal: {y_norm_mse}", end=" ", flush=True)
        if calib_activations:
            y_calib_deq = module(x_calib)
            y_calib_mse = torch.mean((y_calib - y_calib_deq)**2)
            layer_stats["y_calib_mse"] = y_calib_mse.item()
            print(f"Calib: {y_calib_mse}")
        else:
            print()

        # Overlay quantized values
        for wdeq_val in module.weight.data[row].float().unique().cpu():
            plt.axvline(x=wdeq_val, color="b", linestyle="--")

        # Plot reconstructed weight distribution
        counts, bins = torch.histogram(module.weight.data[row].float().cpu(), bins=40)
        plt.figure()
        plt.hist(bins[:-1], bins=bins, weights=counts.float().numpy())
        plt.show()
        plt.title(f"wdeq_{name}_{row}")
        plt.savefig(pdf, format="pdf")
        plt.close()

    pdf.close()

    # Log stats
    df = pd.DataFrame(layers_stats)
    df.to_csv(log_dir / "stats.csv", index=False)

def plot_surface(x: torch.Tensor, title: str):
    # Create a figure with a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    assert x.dim() == 2

    # Create meshgrid coordinates for each element in the tensor
    rows = torch.arange(x.shape[0])
    cols = torch.arange(x.shape[1])
    rows, cols = torch.meshgrid(rows, cols)

    # Plot the surface
    surf = ax.plot_surface(rows.numpy(), cols.numpy(), x.float().cpu().numpy(), cmap="viridis")

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Labeling
    ax.set_xlabel("n_rows")
    ax.set_ylabel("n_cols")
    ax.set_zlabel("val")
    ax.set_title(title)

    # Adjust z-axis limits
    ax.set_zlim(x.min().item(), x.max().item())

    plt.close()
    return fig

if __name__ == '__main__':
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    sub_layer_choices = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate any4 quantization on various language tasks using lm-evaluation-harness.")

    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace model name or path.")
    parser.add_argument("--model-args", type=str, help="Comma separated string arguments for HuggingFace model.")
    parser.add_argument("--quantize", type=str, choices=quant_methods.keys(), help="Quantization method.")
    parser.add_argument("--quantize-args", type=str, help="Comma separated string args to pass to quantization method.")
    parser.add_argument("--bnb-args", type=str, help="Comma separated string args to pass to BitsAndBytes quantization config.")
    parser.add_argument("--device", type=str, default=default_device, help="Device to use.")
    parser.add_argument("--log-dir", type=Path, default="./analysis/tmp", help="Directory to log to.")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Transformer layers to analyze")
    parser.add_argument("--sub-layers", type=str, nargs="+", choices=sub_layer_choices, default=sub_layer_choices, help="Linear module within a transformer layer to analyze.")
    parser.add_argument("--row", type=int, default=0, help="Row of weight matrix to analyze.")
    parser.add_argument("--calib-file", type=str, default=None, help="Path to calibration activations")

    args = parser.parse_args()

    # Pre-process some args
    model_args = {} if not args.model_args else simple_parse_args_string(args.model_args)
    quant_method = None if not args.quantize else quant_methods[args.quantize]
    quant_args = {} if not args.quantize_args else simple_parse_args_string(args.quantize_args)
    bnb_args = None if not args.bnb_args else simple_parse_args_string(args.bnb_args)

    # Run Evaluation
    main(model_name=args.model_name, layers=args.layers, sub_layers=args.sub_layers, row=args.row, model_args=model_args, quant_method=quant_method, quant_args=quant_args, device=args.device, log_dir=args.log_dir, bnb_args=bnb_args, calib_activations_file=args.calib_file)
