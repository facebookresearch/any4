import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

from any4 import convert, any4
device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "facebook/opt-125m"
# model_name = "facebook/opt-1.3b"
# model_name = "meta-llama/Llama-2-70b-chat-hf"
# model_name = "microsoft/phi-1_5"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer)

model.eval()

# messages = [{"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}]
messages = [{"role": "user", "content": "Implement bucket sort in Python and provide a test case with expected results."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)

print("Baseline:")
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=256)
text = tokenizer.batch_decode(outputs)[0]

print("Any4:")
model = convert(model, layer_from=torch.nn.Linear, layer_to=any4)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=256)
text = tokenizer.batch_decode(outputs)[0]

# print("NF4:")
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#    bnb_4bit_compute_dtype=torch.bfloat16
# )
# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
# outputs = model.generate(inputs, streamer=streamer, max_new_tokens=256)
# text = tokenizer.batch_decode(outputs)[0]