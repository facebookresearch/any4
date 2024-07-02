from typing import Type

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def register_forward_hook(model: torch.nn.Module, layer_type: Type = torch.nn.Linear, **kwargs):
    for name, module in model.named_modules():
        if isinstance(module, (layer_type)):
            module.register_forward_hook(get_activation(name))

    return model

device = "cuda" if torch.cuda.is_available() else "cpu"

# model_name = "meta-llama/Meta-Llama-3-8B"
model_name = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

register_forward_hook(model)

prompt = "Hello, my name is"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
model(inputs)

print(activation)
