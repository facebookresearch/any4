from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional
from pathlib import Path
import argparse
import json
import os
import pickle
import sys
import torch
import transformers

import lm_eval
from lm_eval.utils import simple_parse_args_string

import bigcode_eval
import bigcode_eval.tasks
import bigcode_eval.evaluator
import bigcode_eval.arguments
from accelerate import Accelerator

from any4 import convert, quant_methods
from calibrate import calibrate
from utils import CustomJSONEncoder
from data import task_dataset_configs, eval_perplexity

def main(
    model_name: str,
    quant_args: Dict,
    quant_method: Callable,
    model_args: Dict,
    tasks: List[str],
    device: str,
    batch_size: int,
    log_dir: Path,
    generation_args: Dict,
    append_results: Optional[bool] = False,
    load_weights: Optional[Path] = None,
    tokenizer_name: Optional[str] = None,
    save_weights: Optional[bool] = False,
    num_fewshot: Optional[int] = None,
    parallelize: bool = True,
    bnb_args: Optional[Dict] = None,
    calibrate_args: Dict = {},
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

    # Pre-process some args
    if bnb_args:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_args["quantization_config"] = bnb_config

    if quant_args:
        if "sample_weight" in quant_args:
            if quant_args["sample_weight"].endswith(".pickle"):
                with open(quant_args["sample_weight"], 'rb') as handle:
                   quant_args["sample_weight"] = pickle.load(handle)
            elif quant_args["sample_weight"].endswith(".pt"):
                quant_args["sample_weight"] = torch.load(quant_args["sample_weight"], map_location=torch.device(device))
            elif quant_args["sample_weight"] == "calibrate":
                quant_args["sample_weight"] = calibrate

    # instantiate an LM subclass that takes initialized model and can run
    # - `Your_LM.loglikelihood()`
    # - `Your_LM.loglikelihood_rolling()`
    # - `Your_LM.generate_until()`
    lm_obj = lm_eval.models.huggingface.HFLM(
        pretrained=model_name,
        tokenizer=tokenizer_name,
        device=device,
        batch_size=batch_size,
        parallelize=parallelize,
        trust_remote_code=True,
        **model_args
    )

    # Load weights
    if load_weights:
        weights = torch.load(Path(load_weights))
        lm_obj._model.load_state_dict(weights)

    # Apply our quantization algorithms
    if quant_method:
        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        lm_obj._model = convert(lm_obj.model, layer_from=torch.nn.Linear, layer_to=quant_method, tokenizer=lm_obj.tokenizer, calibrate_args=calibrate_args, **quant_args)

    # Save weights
    if save_weights:
        weights = lm_obj._model.state_dict()
        torch.save(weights, Path(log_dir/"weights.pth"))

    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    results = {}
    # LM Eval Harness Evaluation
    harness_tasks = []
    for task in tasks.copy():
        if task in task_manager.all_tasks:
            tasks.remove(task)
            harness_tasks.append(task)
    if harness_tasks:
        # Setting `task_manager` to the one above is optional and should generally be done
        # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
        # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
        harness_results = lm_eval.simple_evaluate( # call simple_evaluate
            model=lm_obj,
            tasks=harness_tasks,
            num_fewshot=num_fewshot,
            task_manager=task_manager,
            model_args={"parallelize": parallelize},
        )
        results.update(harness_results["results"])
        print(f"NLP Eval Results: {harness_results}")

    # BigCode Evaluation
    bigcode_tasks = []
    for task in tasks.copy():
        if task in bigcode_eval.tasks.ALL_TASKS:
            tasks.remove(task)
            bigcode_tasks.append(task)
    if bigcode_tasks:
        accelerator = Accelerator()
        bigcode_args = {
            "modeltype": "causal",
            "batch_size": batch_size,
            "max_length_generation": 512,
            "limit": None,
            "limit_start": 0,
            "instruction_tokens": None,
            "save_every_k_tasks": 1,
            "postprocess": True,
            "allow_code_execution": True,
            "generation_only": False,
            "load_generations_path": None,
            "load_data_path": None,
            "metric_output_path": str(Path(log_dir/"bigcode_evaluation_results.json")),
            "load_generations_intermediate_paths": None,
            "save_generations": False,
            "save_generations_path": str(Path(log_dir/"bigcode_generations.json")),
            "save_references": False,
            "save_generations_path": str(Path(log_dir/"bigcode_references.json")),
            "check_references": False,
            "max_memory_per_gpu": None,
            **generation_args,
        }
        bigcode_evaluator = bigcode_eval.evaluator.Evaluator(
            accelerator=accelerator,
            model=lm_obj.model,
            tokenizer=lm_obj.tokenizer,
            args=argparse.Namespace(**bigcode_args),
        )
        bigcode_results = {}
        for task in bigcode_tasks:
            bigcode_results[task] = bigcode_evaluator.evaluate(task)
        results.update(bigcode_results)
        print(f"Code Eval Results: {bigcode_results}")

    # TODO: args.datasets could be nargs of comma-listed arguments
    data_tasks = []
    for task in tasks.copy():
        if task in task_dataset_configs:
            tasks.remove(task)
            data_tasks.append(task)
    if data_tasks:
        data_results = {}
        for task in data_tasks:
            data_results[task] = eval_perplexity(model=lm_obj.model, tokenizer=lm_obj.tokenizer, batch_size=1, **task_dataset_configs[task])
        results.update(data_results)
        print(f"Perplexity Eval Results: {data_results}")

    if tasks:
        print(f"WARNING: The following tasks are unknown: {tasks}")

    # Log results
    print(f"All Eval Results: {results}")
    if append_results and Path(log_dir/"results.json").exists():
        with Path(log_dir/"results.json").open("r") as f:
            prev_results = json.load(f)
            results.update(prev_results)
    with Path(log_dir/"results.json").open("w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Evaluate any4 quantization on various language tasks using lm-evaluation-harness.")

    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace model name or path.")
    parser.add_argument("--model-args", type=str, help="Comma separated string arguments for HuggingFace model.")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="HuggingFace tokenizer name or path.")
    parser.add_argument("--quantize", type=str, choices=quant_methods.keys(), help="Quantization method.")
    parser.add_argument("--quantize-args", type=str, help="Comma separated string args to pass to quantization method.")
    parser.add_argument("--calibrate-args", type=str, help="Comma separated string args to pass to calibration function.")
    parser.add_argument("--bnb-args", type=str, help="Comma separated string args to pass to BitsAndBytes quantization config.")
    parser.add_argument("--tasks", type=str, nargs="+", default=["arc_easy","arc_challenge","gsm8k","hellaswag","mathqa","mmlu","nq_open", "openbookqa", "piqa", "race","social_iqa","toxigen","triviaqa","truthfulqa","wikitext","winogrande","boolq", "copa"], help="lm-evaluation-harness tasks to evaluate.")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Number of few shots to evaluate tasks.")
    parser.add_argument("--device", type=str, default=default_device, help="Device to use.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--parallelize", default=True, action=argparse.BooleanOptionalAction, help="Enable parallel inference on multiple GPUs.")
    parser.add_argument("--generation-args", type=str, help="Comma separated string args to pass to lm_eval and BigCode generation args.")
    parser.add_argument("--log-dir", type=Path, default="./logs", help="Directory to log to.")
    parser.add_argument("--append-results", default=False, action=argparse.BooleanOptionalAction, help="Append to any existing results file.")
    parser.add_argument("--save-weights", default=False, action=argparse.BooleanOptionalAction, help="Save checkpoint after quantizing to args.log-dir.")
    parser.add_argument("--load-weights", type=Path, help="Path to laod weights")

    args = parser.parse_args()

    # Pre-process some args
    model_args = {} if not args.model_args else simple_parse_args_string(args.model_args)
    quant_method = None if not args.quantize else quant_methods[args.quantize]
    quant_args = {} if not args.quantize_args else simple_parse_args_string(args.quantize_args)
    calibrate_args = {} if not args.calibrate_args else simple_parse_args_string(args.calibrate_args)
    bnb_args = None if not args.bnb_args else simple_parse_args_string(args.bnb_args)
    # TODO: create our own generation args and then adapt them to Eval Harness and BigCode Eval
    generation_args = asdict(bigcode_eval.arguments.EvalArguments())
    if args.generation_args:
        generation_args.update(simple_parse_args_string(args.generation_args))

    # Run Evaluation
    main(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        model_args=model_args,
        quant_method=quant_method,
        quant_args=quant_args,
        calibrate_args=calibrate_args,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        device=args.device,
        batch_size=args.batch_size,
        parallelize=args.parallelize,
        generation_args=generation_args,
        log_dir=args.log_dir,
        append_results=args.append_results,
        save_weights=args.save_weights,
        load_weights=args.load_weights,
        bnb_args=bnb_args,
    )
