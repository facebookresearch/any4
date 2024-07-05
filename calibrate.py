from typing import List, Tuple, Type

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pickle
from pathlib import Path

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

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

register_forward_hook(model)

prompt =    """This is a diverse prompt that contains:
            - Fiction: "Once upon a time, a girl named Alice was living alone on an island. One day, she met a wizard ..."
            - News: "The United Nations held its General Assembly meeting this year amid multiple world crises and wars. In his speech, the General Secretary called for ..."
            - Code: `public static void main(String[] args) {\n System.out.println("Hello world!);\n}`
            - Math: (5.2 + 2.7) / 0.6 - 1.9 * 2.2 =
            - Facts: "The capital of Egypt is Cairo. It is the largest city in the region and is home to..."
            """

if False:
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

pickle_name = Path(f"./profiles/{model_name}_prompt.pt")
pickle_name.parent.mkdir(parents=True, exist_ok=True)
if pickle_name.endswith(".pt"):
    torch.save(layer_to_mean_activations, pickle_name)
elif pickle_name.endswith(".pickle"):
    with open(pickle_name, 'wb') as handle:
        pickle.dump(layer_to_mean_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    raise ValueError(f"Unsupported file type {pickle_name}")
