from typing import List, Optional
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import lm_eval

from any4 import convert, any4

def main(
    model_name: str,
    tasks: List[str],
    device: str,
    batch_size: int,
    log_dir: Path,
    num_fewshot: Optional[int] = None,
):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    model.eval()
    model = convert(model, layer_from=torch.nn.Linear, layer_to=any4)
    model = model.to(device)

    import json
    log_dir.mkdir(exist_ok=True)

    # instantiate an LM subclass that takes initialized model and can run
    # - `Your_LM.loglikelihood()`
    # - `Your_LM.loglikelihood_rolling()`
    # - `Your_LM.generate_until()`
    lm_obj = lm_eval.models.huggingface.HFLM(pretrained=model, batch_size=batch_size)#, parallelize=True)
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
        tasks=tasks,
        num_fewshot=None,
        task_manager=task_manager,
        # model_args={"parallelize": True},
    )

    print(results["results"])
    with Path(log_dir/"results.json").open("w") as f:
        json.dump(results["results"], f, indent=4)


if __name__ == '__main__':
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate any4 quantization on various language tasks using lm-evaluation-harness.")

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="HuggingFace model name or path.")
    parser.add_argument("--tasks", type=str, nargs="+", default=["arc_easy","arc_challenge","gsm8k","hellaswag","mathqa","mmlu","nq_open","piqa","race","social_iqa","toxigen","triviaqa","truthfulqa","wikitext","winogrande"], help="lm-evaluation-harness tasks to evaluate.")
    parser.add_argument("--device", type=str, default=default_device, help="Device to use.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--log_dir", type=Path, default="./logs", help="Directory to log to.")

    args = parser.parse_args()

    main(model_name=args.model_name, tasks=args.tasks, device=args.device, batch_size=args.batch_size, log_dir=args.log_dir)
