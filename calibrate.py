from typing import List, Optional, Tuple, Type, Union
from tqdm import tqdm
import gc
import json
import numpy as np
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pickle
from pathlib import Path

from utils import CustomJSONEncoder, remove_all_hooks

default_prompt = """This is a diverse prompt that contains:
                - Fiction: "Once upon a time, a girl named Alice was living alone on an island. One day, she met a wizard ..."
                - News: "The United Nations held its General Assembly meeting this year amid multiple world crises and wars. In his speech, the General Secretary called for ..."
                - Code: `public static void main(String[] args) {\n System.out.println("Hello world!);\n}`
                - Math: (5.2 + 2.7) / 0.6 - 1.9 * 2.2 =
                - Facts: "The capital of Egypt is Cairo. It is the largest city in the region and is home to..."
                """

layer_to_sum_activations = {}
layer_to_num_activations = {}
layer_to_list_activations = {}
layer_filter = None
do_return_activations = False

def get_mean_activations(name):
    def hook(model, input, output):
        if layer_filter is not None and name not in layer_filter:
            return

        if isinstance(input, (List, Tuple)):
            input = input[0]
        # take the mean along all dimensions except the embedding dimension (which is the last dimension)
        # converting input to double() to minimize chances of overflow when summing activations
        input = input.detach().cpu().double()
        sum_activation = input.sum(dim=torch.arange(input.ndim - 1).tolist())
        num_activations = np.prod(input.shape[:input.ndim - 1])

        if name not in layer_to_sum_activations:
            layer_to_sum_activations[name] = sum_activation
            layer_to_num_activations[name] = num_activations
            if do_return_activations:
                layer_to_list_activations[name] = [input]
        else:
            layer_to_sum_activations[name] += sum_activation
            layer_to_num_activations[name] += num_activations
            if do_return_activations:
                layer_to_list_activations[name].append(input)
    return hook

def register_forward_hook(model: torch.nn.Module, layer_type: Type = torch.nn.Linear, **kwargs):
    for name, module in model.named_modules():
        if isinstance(module, (layer_type)):
            module.register_forward_hook(get_mean_activations(name))

    return model

def calibrate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str = default_prompt,
    dataset: Optional[str] = None,
    config: Optional[Union[str, List[str]]] = None,
    split: str = "train",
    field: str = "text",
    seed: int = 42,
    batch_size: int = 1,
    num_batches: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    padding: bool = True,
    truncate: bool = False,
    layers: List[str] = None,
    return_activations: bool = False,
):
    if max_seq_len is not None:
        truncate = True

    global layer_to_sum_activations
    global layer_to_num_activations
    global layer_to_list_activations
    global layer_filter
    global do_return_activations
    layer_to_sum_activations = {}
    layer_to_num_activations = {}
    layer_filter = layers
    do_return_activations = return_activations
    register_forward_hook(model)

    if not isinstance(config, List):
        config = [config]

    # Apply inputs
    if dataset:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for name in config:
            for idx, batch in enumerate(tqdm(load_dataset(dataset, name=name, split=split, streaming=True).shuffle(seed=seed).iter(batch_size=batch_size))):
                if num_batches is not None and idx >= num_batches:
                    break
                inputs = tokenizer.batch_encode_plus(batch[field], return_tensors="pt", padding=padding, truncation=truncate, max_length=max_seq_len).to(model.device)
                model(**inputs)
                del inputs
    else:
        if os.path.exists(prompt):
            with open(prompt, 'r') as file:
                prompt = file.read()
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        model(inputs)
        del inputs

    # Calculate mean activation per layer
    layer_to_mean_activations = {layer : sum/layer_to_num_activations[layer] for layer,sum in layer_to_sum_activations.items()}

    remove_all_hooks(model)

    for tensor in layer_to_sum_activations.values():
        del tensor
        tensor = None
    gc.collect()

    if do_return_activations:
        return layer_to_mean_activations, layer_to_list_activations
    else:
        return layer_to_mean_activations

def main(
    model_name: str,
    device: str,
    log_dir: Path,
    prompt: str = default_prompt,
    dataset: Optional[str] = None,
    config: Optional[List[str]] = None,
    split: str = "train",
    field: str = "text",
    seed: int = 42,
    batch_size: int = 1,
    num_batches: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    padding: bool = True,
    truncate: bool = False,
    layers: List[str] = None,
    save_type: str = "pt",
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

    # Setup Model
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Calibrate
    layer_to_mean_activations = calibrate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        dataset=dataset,
        config=config,
        split=split,
        field=field,
        seed=seed,
        batch_size=batch_size,
        num_batches=num_batches,
        max_seq_len=max_seq_len,
        padding=padding,
        truncate=truncate,
        layers=layers,
    )
    print(layer_to_num_activations)

    # Log Activations
    log_name = dataset.split("/")[-1] if dataset else "prompt"
    if save_type == "pt":
        pt_file = f"./{log_dir}/{log_name}.pt"
        print(f"Saving calibration activations to {pt_file}")
        torch.save(layer_to_mean_activations, Path(pt_file))
    elif save_type == "pickle":
        pickle_file = f"./{log_dir}/{log_name}.pickle"
        print(f"Saving calibration activations to {pickle_file}")
        with open(Path(pickle_file), 'wb') as handle:
            pickle.dump(layer_to_mean_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unsupported save type {save_type}")

if __name__ == '__main__':
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate any4 quantization on various language tasks using lm-evaluation-harness.")

    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace model name or path.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Maximum sequence length.")
    parser.add_argument("--num-batches", type=int, default=None, help="Limit on number of batches.")
    parser.add_argument("--log-dir", type=Path, default="./profiles", help="Directory to log to.")
    parser.add_argument("--save-type", type=str, default="pt", choices=["pt", "pickle"], help="Type of file to save calibrated activations.")
    parser.add_argument("--prompt", type=str, default=default_prompt, help="Prompt to apply.")
    parser.add_argument("--dataset", type=str, help="Dataset to load samples from.")
    parser.add_argument("--config", nargs="+", type=str, help="Config to load from within the dataset.")
    parser.add_argument("--split", type=str, default="train", help="Split to load from within the dataset.")
    parser.add_argument("--field", type=str, help="Field to load from within the dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling dataset.")
    # TODO: add --task option to load data using lm_eval

    args = parser.parse_args()

    # Run Evaluation
    main(
        model_name=args.model_name,
        device=args.device,
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        field=args.field,
        seed=args.seed,
        prompt=args.prompt,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        max_seq_len=args.max_seq_len,
        log_dir=args.log_dir,
        save_type=args.save_type
    )
