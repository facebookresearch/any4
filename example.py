import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from any4 import convert, any4

# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "facebook/opt-125m"
model_name = "facebook/opt-1.3b"
# model_name = "meta-llama/Llama-2-70b-chat-hf"
# model_name = "microsoft/phi-1_5"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer)

messages = [{"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

print("Baseline:")
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=256)
text = tokenizer.batch_decode(outputs)[0]

print("Any4:")
model = convert(model, layer_from=torch.nn.Linear, layer_to=any4)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=256)
text = tokenizer.batch_decode(outputs)[0]
