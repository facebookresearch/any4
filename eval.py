import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import lm_eval

from any4 import convert, any4
# device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "facebook/opt-125m"
# model_name = "facebook/opt-1.3b"
# model_name = "meta-llama/Llama-2-70b-chat-hf"
# model_name = "microsoft/phi-1_5"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

model.eval()
model = convert(model, layer_from=torch.nn.Linear, layer_to=any4)
model = model.to(device)

log_dir = "./logs/llama2-7b-any4-cluster"
import json
import os
os.makedirs(log_dir, exist_ok=True)

# instantiate an LM subclass that takes initialized model and can run
# - `Your_LM.loglikelihood()`
# - `Your_LM.loglikelihood_rolling()`
# - `Your_LM.generate_until()`
lm_obj = lm_eval.models.huggingface.HFLM(pretrained=model, batch_size=16)#, parallelize=True)
# lm_obj._model = convert(lm_obj.model, layer_from=torch.nn.Linear, layer_to=any4)

# indexes all tasks from the `lm_eval/tasks` subdirectory.
# Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
# to include a set of tasks in a separate directory.
task_manager = lm_eval.tasks.TaskManager()

# Setting `task_manager` to the one above is optional and should generally be done
# if you want to include tasks from paths other than ones in `lm_eval/tasks`.
# `simple_evaluate` will instantiate its own task_manager if it is set to None here.
results = lm_eval.simple_evaluate( # call simple_evaluate
    model=lm_obj,
    tasks=["social_iqa", "piqa"],
    num_fewshot=0,
    task_manager=task_manager,
    # model_args={"parallelize": True},
)

print(results["results"])