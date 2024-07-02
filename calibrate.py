from typing import Iterable, List, Tuple, Type

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

prompt = "Hello, my name is"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
model(inputs)

print(layer_to_mean_activations)
