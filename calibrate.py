from typing import List, Tuple, Type

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pickle
from pathlib import Path

default_prompt = """This is a diverse prompt that contains:
                - Fiction: "Once upon a time, a girl named Alice was living alone on an island. One day, she met a wizard ..."
                - News: "The United Nations held its General Assembly meeting this year amid multiple world crises and wars. In his speech, the General Secretary called for ..."
                - Code: `public static void main(String[] args) {\n System.out.println("Hello world!);\n}`
                - Math: (5.2 + 2.7) / 0.6 - 1.9 * 2.2 =
                - Facts: "The capital of Egypt is Cairo. It is the largest city in the region and is home to..."
                """

layer_to_mean_activations = {}

def get_mean_activations(name):
    def hook(model, input, output):
        if isinstance(input, (List, Tuple)):
            input = input[0]
        # take the mean along all dimensions except the embedding dimension (which is the last dimension)
        mean_activation = input.detach().mean(dim=torch.arange(input.ndim - 1).tolist())
        layer_to_mean_activations[name] = mean_activation
    return hook

def register_forward_hook(model: torch.nn.Module, layer_type: Type = torch.nn.Linear, **kwargs):
    for name, module in model.named_modules():
        if isinstance(module, (layer_type)):
            module.register_forward_hook(get_mean_activations(name))

    return model

def main(
    model_name: str,
    device: str,
    log_dir: Path,
    batch_size: int,
    prompt: str = default_prompt,
    save_type: str = "pt",
):

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    register_forward_hook(model)

    if prompt is None:
        dataset = "bigcode/the-stack"
        num_samples = 200
        split = "train"
        field = "content"
        for idx, sample in enumerate(load_dataset(dataset, split=split, streaming=True)):
            if idx >= num_samples:
                break
            inputs = tokenizer.encode(sample[field], return_tensors="pt").to(device)
            model(inputs)
    else:
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        model(inputs)

    print(layer_to_mean_activations)

    if save_type == "pt":
        torch.save(layer_to_mean_activations, Path(f"./{log_dir}/prompt.pt"))
    elif save_type == "pickle":
        with open(Path(f"./{log_dir}/prompt.pickle"), 'wb') as handle:
            pickle.dump(layer_to_mean_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unsupported save type {save_type}")

if __name__ == '__main__':
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate any4 quantization on various language tasks using lm-evaluation-harness.")

    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace model name or path.")
    parser.add_argument("--device", type=str, default=default_device, help="Device to use.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--log-dir", type=Path, default="./profiles", help="Directory to log to.")
    parser.add_argument("--save-type", type=str, default="pt", choices=["pt", "pickle"], help="Type of file to save calibrated activations.")

    args = parser.parse_args()

    # Run Evaluation
    main(model_name=args.model_name, device=args.device, batch_size=args.batch_size, log_dir=args.log_dir, save_type=args.save_type)
