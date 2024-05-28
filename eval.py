import lm_eval
import lm_eval.models
from transformers import AutoModelForCausalLM

# model_name = "meta-llama/Meta-Llama-3-8B"
model_name = "facebook/opt-125m"
# model = AutoModelForCausalLM.from_pretrained(model_name)

# instantiate an LM subclass that takes initialized model and can run
# - `Your_LM.loglikelihood()`
# - `Your_LM.loglikelihood_rolling()`
# - `Your_LM.generate_until()`
lm_obj = lm_eval.models.huggingface.HFLM(pretrained=model_name, batch_size=16)

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
)

print(results["results"])