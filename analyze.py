import json
import os
import pickle
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import lm_eval
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import transformers

from any4 import convert, group_q, quant_methods
from lm_eval.utils import simple_parse_args_string
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from utils import CustomJSONEncoder


def get_entropy(t):
    hist, bins = np.histogram(t.flatten().float().cpu(), bins="auto", density=True)
    probs = hist / np.sum(hist)
    return stats.entropy(probs, base=2)


"""
def weighted_entropy(entropies, num_weight_entries):
    total_weights = sum(num_weight_entries)
    return sum(
        [
            ((num_weight_entries[idx] / total_weights) * entropy)
            for idx, entropy in enumerate(entropies)
        ]
    )
"""


def main(
    model_name: str,
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
    with Path(log_dir / "args.json").open("w") as f:
        json.dump(args, f, indent=4, cls=CustomJSONEncoder)

    # Log command to a file
    arg_str = " ".join([arg.replace("'", "'\\''") for arg in sys.argv[1:]])
    with open(log_dir / "command_line.txt", "w") as f:
        f.write(f"python {os.path.basename(__file__)} {arg_str}\n")

    # Create PDF for logs
    pdf = PdfPages(log_dir / "plots.pdf")

    if bnb_args:
        bnb_config = transformers.BitsAndBytesConfig(**bnb_args)
        model_args["quantization_config"] = bnb_config

    lm_obj = lm_eval.models.huggingface.HFLM(
        pretrained=model_name,
        device=device,
        batch_size=bs,
        parallelize=parallelize,
        **model_args,
    )
    model = lm_obj._model

    # Load calibration activations
    calib_activations = None
    if calib_activations_file:
        if calib_activations_file.endswith(".pickle"):
            with open(calib_activations_file, "rb") as handle:
                calib_activations = pickle.load(handle)
        elif calib_activations_file.endswith(".pt"):
            calib_activations = torch.load(
                calib_activations_file, map_location=torch.device(device)
            )

    # Create list of modules
    modules = []
    names = []
    layers_stats = []
    for name, module in model.named_modules():
        # TODO: make condition of layers (e.g., layer type or layer name) as an argument, or decouple the condition from this script
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)):
            modules.append(module)
            names.append(name)
            layers_stats.append({})

    entropies = []
    num_weight_entries = []

    # Analyze each module
    for name, module, layer_stats in zip(names, modules, layers_stats):
        print(f"{name}")
        layer_stats["layer"] = name

        # Store the weight
        w = module.weight.data.clone()

        # Apply on random inputs
        x_uni = torch.rand(
            size=(bs, module.in_features), device=w.device, dtype=w.dtype
        )
        y_uni = module(x_uni)
        x_norm = torch.randn(
            size=(bs, module.in_features), device=w.device, dtype=w.dtype
        )
        y_norm = module(x_norm)
        if calib_activations:
            x_calib = calib_activations[name].to(w.device).to(w.dtype)
            y_calib = module(x_calib)

        ## Original Weight
        # Log Stats
        (w_mean, w_std) = torch.std_mean(w)
        w_min, w_max = w.min(), w.max()
        w_entropy = get_entropy(w)
        layer_stats["w_mean"] = w_mean.item()
        layer_stats["w_std"] = w_std.item()
        layer_stats["w_min"] = w_min.item()
        layer_stats["w_max"] = w_max.item()

        print(
            f"\tWeight: Mean:{w_mean.item()}, Std:{w_std.item()}, Min:{w_min.item()}, Max:{w_max.item()}"
        )

        # Plot Surface
        fig = plot_surface(w)
        fig.suptitle(f"{name}\nw")
        fig.savefig(pdf, format="pdf")

        # Plot Distribution
        fig = plot_histogram(w.flatten().float().cpu(), bins=40)
        fig.suptitle(f"{name}\nw")
        fig.savefig(pdf, format="pdf")

        # Plot Distribution of Row
        fig = plot_histogram(w[row].float().cpu(), bins=40)
        fig.suptitle(f"{name}\nw, row={row}")
        fig.savefig(pdf, format="pdf")

        ## Calibrated Activations
        if calib_activations:
            # Log Stats
            (x_calib_mean, x_calib_std) = torch.std_mean(x_calib)
            x_calib_min, x_calib_max = x_calib.min(), x_calib.max()
            layer_stats["x_calib_mean"] = x_calib_mean.item()
            layer_stats["x_calib_std"] = x_calib_std.item()
            layer_stats["x_calib_min"] = x_calib_min.item()
            layer_stats["x_calib_max"] = x_calib_max.item()
            print(
                f"\tInput: Mean:{x_calib_mean.item()}, Std:{x_calib_std.item()}, Min:{x_calib_min.item()}, Max:{x_calib_max.item()}"
            )

            (y_calib_mean, y_calib_std) = torch.std_mean(y_calib)
            y_calib_min, y_calib_max = y_calib.min(), y_calib.max()
            layer_stats["y_calib_mean"] = y_calib_mean.item()
            layer_stats["y_calib_std"] = y_calib_std.item()
            layer_stats["y_calib_min"] = y_calib_min.item()
            layer_stats["y_calib_max"] = y_calib_max.item()
            print(
                f"\tOutput: Mean:{y_calib_mean.item()}, Std:{y_calib_std.item()}, Min:{y_calib_min.item()}, Max:{y_calib_max.item()}"
            )

            # Plot Line
            fig = plot_line(x_calib)
            fig.suptitle(f"{name}\nx_calib")
            fig.savefig(pdf, format="pdf")

            # Plot Distribution
            fig = plot_histogram(x_calib.flatten().float().cpu(), bins=40)
            fig.suptitle(f"{name}\nx_calib")
            fig.savefig(pdf, format="pdf")

        # Apply our quantization algorithms
        if quant_method:
            per_layer_quant_args = quant_args.copy()
            # TODO: handle if transpose (default group_size will change)
            # TODO: handle default arguments in a different way
            # TODO: consider reading wc from quantize or convert method
            per_layer_quant_args["n_bit"] = per_layer_quant_args.get("n_bit", 4)
            per_layer_quant_args["group_size"] = per_layer_quant_args.get(
                "group_size", w.shape[-1]
            )
            per_layer_quant_args["scale_only"] = per_layer_quant_args.get(
                "scale_only", False
            )

            ## Centred Weights
            wc, _, scales_and_zeros = group_q(
                w,
                n_bit=per_layer_quant_args["n_bit"],
                q_group_size=per_layer_quant_args["group_size"],
                assymetric=not per_layer_quant_args["scale_only"],
            )

            # Log Stats
            (wc_mean, wc_std) = torch.std_mean(wc)
            wc_min, wc_max = wc.min(), wc.max()
            wc_entropy = get_entropy(wc.round())
            entropies.append(wc_entropy.item())
            num_weight_entries.append(wc.numel())

            layer_stats["wc_mean"] = wc_mean.item()
            layer_stats["wc_std"] = wc_std.item()
            layer_stats["wc_min"] = wc_min.item()
            layer_stats["wc_max"] = wc_max.item()
            layer_stats["wc_entropy"] = wc_entropy.item()
            print(
                f"\tWeight Centred: Mean:{wc_mean.item()}, Std:{wc_std.item()}, Min:{wc_min.item()}, Max:{wc_max.item()}, Wc_entropy:{wc_entropy.item()}"
            )

            # Plot Surface
            fig = plot_surface(wc)
            fig.suptitle(f"{name}\nwc")
            fig.savefig(pdf, format="pdf")

            # Plot Distribution
            fig = plot_histogram(wc.flatten().float().cpu(), bins=40)
            fig.suptitle(f"{name}\nwc")
            fig.savefig(pdf, format="pdf")

            # Plot Distribution of Row
            fig = plot_histogram(wc[row].float().cpu(), bins=40)
            fig.suptitle(f"{name}\nwc, row={row}")
            fig.savefig(pdf, format="pdf")

            ## Analyze Quantization
            module = convert(
                module,
                layer_from=torch.nn.Linear,
                layer_to=quant_method,
                **per_layer_quant_args,
            )
            wdeq = module.weight.data

            # Get mean square error
            y_uni_deq = module(x_uni)
            y_norm_deq = module(x_norm)
            w_mse = torch.mean((w - wdeq) ** 2)
            y_uni_mse = torch.mean((y_uni - y_uni_deq) ** 2)
            y_norm_mse = torch.mean((y_norm - y_norm_deq) ** 2)
            layer_stats["w_mse"] = w_mse.item()
            layer_stats["y_uni_mse"] = y_uni_mse.item()
            layer_stats["y_norm_mse"] = y_norm_mse.item()
            print(
                f"\tMean Square Error: Weight:{w_mse}, Output: Uniform: {y_uni_mse} Normal: {y_norm_mse}",
                end=" ",
            )
            if calib_activations:
                y_calib_deq = module(x_calib)
                y_calib_mse = torch.mean((y_calib - y_calib_deq) ** 2)
                layer_stats["y_calib_mse"] = y_calib_mse.item()
                print(f"Calib: {y_calib_mse}")
            else:
                print()

            # Overlay quantized values
            for wdeq_val in wdeq[row].float().unique().cpu():
                plt.axvline(x=wdeq_val, color="b", linestyle="--")

            ## Reconstructed Weight
            # Log Stats
            (wdeq_mean, wdeq_std) = torch.std_mean(wdeq)
            wdeq_min, wdeq_max = wdeq.min(), wdeq.max()
            layer_stats["wdeq_mean"] = wdeq_mean.item()
            layer_stats["wdeq_std"] = wdeq_std.item()
            layer_stats["wdeq_min"] = wdeq_min.item()
            layer_stats["wdeq_max"] = wdeq_max.item()
            print(
                f"\tWeight Dequantized: Mean:{wdeq_mean.item()}, Std:{wdeq_std.item()}, Min:{wdeq_min.item()}, Max:{wdeq_max.item()}"
            )

            # Plot Surface
            fig = plot_surface(wdeq)
            fig.suptitle(f"{name}\nw")
            fig.savefig(pdf, format="pdf")

            # Plot Distribution
            fig = plot_histogram(wdeq.flatten().float().cpu(), bins=40)
            fig.suptitle(f"{name}\nw_deq")
            fig.savefig(pdf, format="pdf")

            # Plot Distribution of Row
            fig = plot_histogram(wdeq[row].float().cpu(), bins=40)
            fig.suptitle(f"{name}\nw_deq, row={row}")
            fig.savefig(pdf, format="pdf")

        print(flush=True)

    # Plot Entropy Distribution
    fig = plot_entropy(entropies)
    fig.savefig(pdf, format="pdf")

    max_entropy = np.max(entropies)
    min_entropy = np.min(entropies)
    mean_entropy = np.average(entropies)
    weighted_mean_entropy = np.average(entropies, weights=num_weight_entries)
    # weight_entropy = weighted_entropy(entropies, num_weight_entries)

    with open(log_dir / "entropy_stats.txt", "w") as f:
        f.write(f"Max Entropy: {max_entropy}\n")
        f.write(f"Min Entropy: {min_entropy}\n")
        f.write(f"Mean Entropy: {mean_entropy}\n")
        f.write(f"Weighted Mean Entropy: {weighted_mean_entropy}\n")
    #    f.write(f"Weight Entropy: {weight_entropy}\n")

    pdf.close()

    # Log stats
    df = pd.DataFrame(layers_stats)
    csv_path = log_dir / "stats.csv"
    print(f"Logging analysis to {csv_path}")
    df.to_csv(csv_path, index=False)


def plot_line(x: torch.Tensor):
    assert x.dim() == 1
    fig = plt.figure()
    plt.plot(np.arange(x.shape[0]), x.float().cpu().numpy())
    plt.xlabel("Channel")
    plt.ylabel("Value")
    plt.grid(True)
    return fig


def plot_histogram(x: torch.Tensor, bins: int):
    counts, bins = torch.histogram(x.float().cpu(), bins=bins)
    fig = plt.figure()
    plt.hist(bins[:-1], bins=bins, weights=counts.float().numpy())
    plt.show()
    plt.close()
    return fig


def plot_surface(x: torch.Tensor):
    # Create a figure with a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    assert x.dim() == 2

    # Create meshgrid coordinates for each element in the tensor
    rows = torch.arange(x.shape[0])
    cols = torch.arange(x.shape[1])
    rows, cols = torch.meshgrid(rows, cols)

    # Plot the surface
    surf = ax.plot_surface(
        rows.numpy(), cols.numpy(), x.float().cpu().numpy(), cmap="viridis"
    )

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Labeling
    ax.set_xlabel("Row")
    ax.set_ylabel("Channel")
    ax.set_zlabel("Value")

    # Adjust z-axis limits
    ax.set_zlim(x.min().item(), x.max().item())

    plt.close()
    return fig


def plot_entropy(entropy_values):
    fig, ax = plt.subplots()
    ax.bar(range(len(entropy_values)), entropy_values)
    ax.set_title("Entropy per Layer")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Entropy")
    return fig


if __name__ == "__main__":
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate any4 quantization on various language tasks using lm-evaluation-harness."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="HuggingFace model name or path.",
    )
    parser.add_argument(
        "--model-args",
        type=str,
        help="Comma separated string arguments for HuggingFace model.",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=quant_methods.keys(),
        help="Quantization method.",
    )
    parser.add_argument(
        "--quantize-args",
        type=str,
        help="Comma separated string args to pass to quantization method.",
    )
    parser.add_argument(
        "--bnb-args",
        type=str,
        help="Comma separated string args to pass to BitsAndBytes quantization config.",
    )
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use."
    )
    parser.add_argument(
        "--log-dir", type=Path, default="./analysis/tmp", help="Directory to log to."
    )
    parser.add_argument(
        "--row", type=int, default=0, help="Row of weight matrix to analyze."
    )
    parser.add_argument(
        "--calib-file", type=str, default=None, help="Path to calibration activations"
    )

    args = parser.parse_args()

    # Pre-process some args
    model_args = (
        {} if not args.model_args else simple_parse_args_string(args.model_args)
    )
    quant_method = None if not args.quantize else quant_methods[args.quantize]
    quant_args = (
        {} if not args.quantize_args else simple_parse_args_string(args.quantize_args)
    )
    bnb_args = None if not args.bnb_args else simple_parse_args_string(args.bnb_args)

    # Run Evaluation
    main(
        model_name=args.model_name,
        row=args.row,
        model_args=model_args,
        quant_method=quant_method,
        quant_args=quant_args,
        device=args.device,
        log_dir=args.log_dir,
        bnb_args=bnb_args,
        calib_activations_file=args.calib_file,
    )
