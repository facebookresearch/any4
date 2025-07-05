# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

from quantize import convert, intq, anyq

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "meta-llama/Llama-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer)

model.eval()

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("Warmup...")
_ = model.generate(**inputs, max_new_tokens=4)

print("Baseline:")
outputs = model.generate(**inputs, streamer=streamer, do_sample=True, max_new_tokens=256)
text = tokenizer.batch_decode(outputs)[0]

print("Quantize:")
model = convert(model, layer_from=torch.nn.Linear, layer_to=anyq, skip_modules=["lm_head"], pseudo=False)
outputs = model.generate(**inputs, streamer=streamer, do_sample=True, max_new_tokens=256)
text = tokenizer.batch_decode(outputs)[0]
