from typing import Callable, Dict, List, Optional
from pathlib import Path
import json
import os
import sys
import torch
import transformers
import lm_eval
from lm_eval.utils import simple_parse_args_string

from any4 import convert, intq, anyq
from utils import CustomJSONEncoder

def main(
    model_name: str,
    quant_args: Dict,
    quant_method: Callable,
    model_args: Dict,
    tasks: List[str],
    device: str,
    batch_size: int,
    log_dir: Path,
    num_fewshot: Optional[int] = None,
    parallelize: bool = True,
    bnb_args: Optional[Dict] = None,
):
    log_dir.mkdir(parents=True, exist_ok=True)
    # Log args
    args = locals()
    print(args)
    with Path(log_dir/"args.json").open("w") as f:
        json.dump(args, f, indent=4, cls=CustomJSONEncoder)

    # Log command to a file
    arg_str = ' '.join([arg.replace("'", "'\\''") for arg in sys.argv[1:]])
    with open(log_dir / "command_line.txt", "w") as f:
        f.write(f"python {os.path.basename(__file__)} {arg_str}\n")

    if bnb_args:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_args["quantization_config"] = bnb_config

    # instantiate an LM subclass that takes initialized model and can run
    # - `Your_LM.loglikelihood()`
    # - `Your_LM.loglikelihood_rolling()`
    # - `Your_LM.generate_until()`
    lm_obj = lm_eval.models.huggingface.HFLM(pretrained=model_name, device=device, batch_size=batch_size, parallelize=parallelize, **model_args)

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
        "intq": intq,
        "anyq": anyq,
    }

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate any4 quantization on various language tasks using lm-evaluation-harness.")

    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf", help="HuggingFace model name or path.")
    parser.add_argument("--model-args", type=str, help="Comma separated string arguments for HuggingFace model.")
    parser.add_argument("--quantize", type=str, choices=quant_methods.keys(), help="Quantization method.")
    parser.add_argument("--quantize-args", type=str, help="Comma separated string args to pass to quantization method.")
    parser.add_argument("--bnb-args", type=str, help="Comma separated string args to pass to BitsAndBytes quantization config.")
    parser.add_argument("--tasks", type=str, nargs="+", default=["arc_easy","arc_challenge","gsm8k","hellaswag","mathqa","mmlu","nq_open", "openbookqa", "piqa", "race","social_iqa","toxigen","triviaqa","truthfulqa","wikitext","winogrande"], help="lm-evaluation-harness tasks to evaluate.")
    parser.add_argument("--device", type=str, default=default_device, help="Device to use.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--parallelize", default=True, action=argparse.BooleanOptionalAction, help="Enable parallel inference on multiple GPUs.")
    parser.add_argument("--log-dir", type=Path, default="./logs", help="Directory to log to.")

    args = parser.parse_args()

    # Pre-process some args
    model_args = {} if not args.model_args else simple_parse_args_string(args.model_args)
    quant_method = None if not args.quantize else quant_methods[args.quantize]
    quant_args = {} if not args.quantize_args else simple_parse_args_string(args.quantize_args)
    bnb_args = None if not args.bnb_args else simple_parse_args_string(args.bnb_args)

    # Run Evaluation
    main(model_name=args.model_name, model_args=model_args, quant_method=quant_method, quant_args=quant_args, tasks=args.tasks, device=args.device, batch_size=args.batch_size, parallelize=args.parallelize, log_dir=args.log_dir, bnb_args=bnb_args)
