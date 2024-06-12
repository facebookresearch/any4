from typing import Callable, Dict, List, Optional
from pathlib import Path
import json
import os
import sys
import torch
import lm_eval
from lm_eval.utils import simple_parse_args_string

from any4 import convert, any4, intq, anyq
from utils import CustomJSONEncoder

def main(
    model_name: str,
    quant_method: Callable,
    quant_args: Dict,
    tasks: List[str],
    device: str,
    batch_size: int,
    log_dir: Path,
    num_fewshot: Optional[int] = None,
    parallelize: bool = True,
):
    # Log args
    args = locals()
    print(args)
    with Path(log_dir/"args.json").open("w") as f:
        json.dump(args, f, indent=4, cls=CustomJSONEncoder)

    # instantiate an LM subclass that takes initialized model and can run
    # - `Your_LM.loglikelihood()`
    # - `Your_LM.loglikelihood_rolling()`
    # - `Your_LM.generate_until()`
    lm_obj = lm_eval.models.huggingface.HFLM(pretrained=model_name, device=device, batch_size=batch_size, parallelize=parallelize)

    # Apply our quantization algorithms
    if quant_method:
        lm_obj._model = convert(lm_obj.model, layer_from=torch.nn.Linear, layer_to=quant_method, **quant_args)

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
        num_fewshot=num_fewshot,
        task_manager=task_manager,
        model_args={"parallelize": parallelize},
    )

    # Log results
    print(results["results"])
    with Path(log_dir/"results.json").open("w") as f:
        json.dump(results["results"], f, indent=4)


if __name__ == '__main__':
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_methods = {
        "any4": any4,
        "intq": intq,
        "anyq": anyq,
    }

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate any4 quantization on various language tasks using lm-evaluation-harness.")

    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf", help="HuggingFace model name or path.")
    parser.add_argument("--quantize", type=str, choices=quant_methods.keys(), help="Quantization method.")
    parser.add_argument("--quantize-args", type=str, help="Arguments to pass to quantization method.")
    parser.add_argument("--tasks", type=str, nargs="+", default=["arc_easy","arc_challenge","gsm8k","hellaswag","mathqa","mmlu","nq_open","piqa","race","social_iqa","toxigen","triviaqa","truthfulqa","wikitext","winogrande"], help="lm-evaluation-harness tasks to evaluate.")
    parser.add_argument("--device", type=str, default=default_device, help="Device to use.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--parallelize", default=True, action=argparse.BooleanOptionalAction, help="Enable parallel inference on multiple GPUs.")
    parser.add_argument("--log-dir", type=Path, default="./logs", help="Directory to log to.")

    args = parser.parse_args()

    # Log command to a file
    arg_str = ' '.join([arg.replace("'", "'\\''") for arg in sys.argv[1:]])
    with open(args.log_dir / "command_line.txt", "w") as f:
        f.write(f"python {os.path.basename(__file__)} {arg_str}\n")

    # Pre-process some args
    quant_method = None if not args.quantize else quant_methods[args.quantize]
    quant_args = {} if not args.quantize_args else simple_parse_args_string(args.quantize_args)

    # Run Evaluation
    main(model_name=args.model_name, quant_method=quant_method, quant_args=quant_args, tasks=args.tasks, device=args.device, batch_size=args.batch_size, parallelize=args.parallelize, log_dir=args.log_dir)
