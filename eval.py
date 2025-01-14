from dataclasses import dataclass, asdict
from datetime import timedelta
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
from accelerate import Accelerator, InitProcessGroupKwargs

from any4 import convert, quant_methods
from calibrate import calibrate
from utils import CustomJSONEncoder, dtype_str_to_torch
from data import task_dataset_configs, eval_perplexity
from data_gptq import task_dataset_gptq_configs, get_loaders, llama_eval

def log_results(
        log_dir: Path,
        results: Dict,
        append: bool = True,
        prompt: str = "",
        json_filename: str = "results.json",
    ):
    # Log to Screen
    if prompt:
        print(f"{prompt}: {results}")
    else:
        print(f"{results}")

    # Log to JSON
    json_path = log_dir / json_filename
    if append and Path(json_path).exists():
        with Path(json_path).open("r") as f:
            prev_results = json.load(f)
            prev_results.update(results)
            results = prev_results
    print(f"Logging results to {json_path}")
    with Path(json_path).open("w") as f:
        json.dump(results, f, indent=4)

def main(
    model_name: str,
    quant_args: Dict,
    quant_method: Callable,
    model_args: Dict,
    tasks: List[str],
    device: str,
    log_dir: Path,
    generation_args: Dict,
    batch_size: Optional[int] = None,
    append_results: Optional[bool] = False,
    overwrite_results: Optional[bool] = False,
    load_weights: Optional[Path] = None,
    tokenizer_name: Optional[str] = None,
    save_weights: Optional[bool] = False,
    save_model: Optional[bool] = False,
    num_fewshot: Optional[int] = None,
    parallelize: bool = True,
    bnb_args: Optional[Dict] = None,
    calibrate_args: Dict = {},
    nnq_args: Dict = {},
    seed: int = 0,
    max_seq_len: Optional[int] = 2048,
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
        bnb_config = transformers.BitsAndBytesConfig(**bnb_args)
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

    if nnq_args:
        if "dtype" in nnq_args:
            nnq_args["dtype"] = dtype_str_to_torch[nnq_args["dtype"]]

    if not overwrite_results:
        append_results = True
        skip_tasks = []
        json_path = log_dir / "results.json"
        if Path(json_path).exists():
            with Path(json_path).open("r") as f:
                prev_results = json.load(f)
            for prev_task in prev_results.keys():
                if prev_task in tasks:
                    tasks.remove(prev_task)
                    skip_tasks.append(prev_task)
            print(f"Skipping {skip_tasks} as its results exists from previous run.")

    # instantiate an LM subclass that takes initialized model and can run
    # - `Your_LM.loglikelihood()`
    # - `Your_LM.loglikelihood_rolling()`
    # - `Your_LM.generate_until()`
    lm_obj = lm_eval.models.huggingface.HFLM(
        pretrained=model_name,
        tokenizer=tokenizer_name,
        device=device,
        batch_size=batch_size if batch_size is not None else "auto:16",
        parallelize=parallelize,
        trust_remote_code=True,
        **model_args
    )

    # Load weights
    if load_weights:
        weights = torch.load(Path(load_weights))
        lm_obj._model.load_state_dict(weights)

    # Apply our quantization algorithms
    # TODO: move nnq_args into quant_args only if quant_type is anyq
    if quant_method:
        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        lm_obj._model = convert(lm_obj.model, layer_from=torch.nn.Linear, layer_to=quant_method, tokenizer=lm_obj.tokenizer, calibrate_args=calibrate_args, nnq_args=nnq_args, **quant_args)

    # Save weights
    if save_weights:
        try:
            weights = lm_obj._model.state_dict()
            weights_path = weights, Path(log_dir/"weights.pth")
            print(f"Saving modified weights to {weights_path}...")
            torch.save(weights, weights_path)
        except:
            print(f"Failed to save weights")
            pass
    if save_model:
        try:
            model_path = Path(log_dir/"model")
            print(f"Saving modified model, and tokenizer, to {model_path}...")
            lm_obj._model.save_pretrained(model_path, from_pt=True)
            lm_obj.tokenizer.save_pretrained(model_path, from_pt=True)
        except:
            print(f"Failed to save model")
            pass


    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    results = {}

    # Dataset Perplexity (GPTQ Implementation)
    # Many quantization papers use this implementation to measure perplexity so we are using it here
    # TODO: update data.py with extra arguments to match results of data_gptq.py
    data_gptq_tasks = []
    for task in tasks.copy():
        if task in task_dataset_gptq_configs:
            tasks.remove(task)
            data_gptq_tasks.append(task)
    if data_gptq_tasks:
        print("Running Perplexity (GPTQ Implementation) Tasks")
        data_gptq_results = {}
        for task in data_gptq_tasks:
            _, testloader = get_loaders(
                task, tokenizer=lm_obj.tokenizer, seed=seed, seqlen=max_seq_len
            )
            result = llama_eval(lm_obj.model, testloader, device, seqlen=max_seq_len)
            data_gptq_results[task] = result
            log_results(log_dir, {task: result}, append=append_results or len(results) > 0, prompt="Perplexity (GPTQ Implementation) Eval Results", json_filename="results.json")
            results.update({task: result})
            print(f"Perplexity (GPTQ Implementation) Eval Results: {data_gptq_results}")

    # Dataset Perplexity
    # TODO: args.datasets could be nargs of comma-listed arguments
    data_tasks = []
    for task in tasks.copy():
        if task in task_dataset_configs:
            tasks.remove(task)
            data_tasks.append(task)
    if data_tasks:
        print("Running Perplexity Tasks")
        data_results = {}
        for task in data_tasks:
            result = eval_perplexity(model=lm_obj.model, tokenizer=lm_obj.tokenizer, batch_size=batch_size if batch_size is not None else 1, max_seq_len=max_seq_len, **task_dataset_configs[task])
            data_results[task] = result
            log_results(log_dir, {task: result}, append=append_results or len(results) > 0, prompt="Perplexity Eval Results", json_filename="results.json")
            results.update({task: result})
        print(f"Perplexity Eval Results: {data_results}")

    # BigCode Evaluation
    bigcode_tasks = []
    for task in tasks.copy():
        if task in bigcode_eval.tasks.ALL_TASKS:
            tasks.remove(task)
            bigcode_tasks.append(task)
    if bigcode_tasks:
        print("Running Coding Tasks")
        if hasattr(lm_obj, "accelerator") and lm_obj.accelerator is not None:
            accelerator = lm_obj.accelerator
        else:
            accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(weeks=52)))
        bigcode_args = {
            "modeltype": "causal",
            "n_samples": 1, # number of generated candidate solutions
            "batch_size": 1, # for BigCode, batch_size <= n_samples
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
            "max_memory_per_gpu": "dummy", # setting this to "dummy" instead of None to avoid error of loading model to specific device
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
            result = bigcode_evaluator.evaluate(task)
            bigcode_results[task] = result
            log_results(log_dir, {task: result}, append=append_results or len(results) > 0, prompt="Code Eval Results", json_filename="results.json")
            results.update({task: result})
        print(f"Code Eval Results: {bigcode_results}")

    # LM Eval Harness Evaluation
    harness_tasks = []
    for task in tasks.copy():
        if task in task_manager.all_tasks:
            tasks.remove(task)
            harness_tasks.append(task)
    if harness_tasks:
        print("Running Natural Language Tasks")
        harness_results = {}
        # Setting `task_manager` to the one above is optional and should generally be done
        # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
        # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
        for task in harness_tasks:
            harness_output = lm_eval.simple_evaluate( # call simple_evaluate
                model=lm_obj,
                tasks=[task],
                num_fewshot=num_fewshot,
                task_manager=task_manager,
                model_args={"parallelize": parallelize},
            )
            result = harness_output['results']
            harness_results[task] = result
            log_results(log_dir, {task: result}, append=append_results or len(results) > 0, prompt="NLP Eval Results", json_filename="results.json")
            results.update({task: result})
        print(f"NLP Eval Results: {harness_results}")

    if tasks:
        print(f"WARNING: The following tasks are unknown: {tasks}")

    # Log results
    print(f"All Eval Results: {results}")


if __name__ == '__main__':
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Evaluate any4 quantization on various language tasks using lm-evaluation-harness.")

    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace model name or path.")
    parser.add_argument("--model-args", type=str, help="Comma separated string arguments for HuggingFace model.")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="HuggingFace tokenizer name or path.")
    parser.add_argument("--quantize", type=str, choices=quant_methods.keys(), help="Quantization method.")
    parser.add_argument("--quantize-args", type=str, help="Comma separated string args to pass to quantization method.")
    parser.add_argument("--calibrate-args", type=str, help="Comma separated string args to pass to calibration function.")
    parser.add_argument("--nnq-args", type=str, help="Comma separated string args to pass to neural network training for any4.")
    parser.add_argument("--bnb-args", type=str, help="Comma separated string args to pass to BitsAndBytes quantization config.")
    parser.add_argument("--tasks", type=str, nargs="+", default=["piqa","arc_easy","arc_challenge","hellaswag","winogrande", "bbh","gsm8k","lambada","mathqa","mmlu","nq_open", "openbookqa", "race","social_iqa","toxigen","triviaqa","truthfulqa","wikitext","boolq", "copa", "squadv2", "humaneval", "mbpp", "wikitext-2", "wikipedia", "c4", "c4_new", "ptb", "ptb_new", "codeparrot"], help="lm-evaluation-harness tasks to evaluate.")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Number of few shots to evaluate tasks.")
    parser.add_argument("--device", type=str, default=default_device, help="Device to use.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--parallelize", default=True, action=argparse.BooleanOptionalAction, help="Enable parallel inference on multiple GPUs.")
    parser.add_argument("--generation-args", type=str, help="Comma separated string args to pass to lm_eval and BigCode generation args.")
    parser.add_argument("--log-dir", type=Path, default="./logs/tmp", help="Directory to log to.")
    parser.add_argument("--append-results", default=False, action=argparse.BooleanOptionalAction, help="Append to any existing results file.")
    parser.add_argument("--overwrite-results", default=False, action=argparse.BooleanOptionalAction, help="If task already exist in results.json, re-run and overwrite it.")
    parser.add_argument("--save-weights", default=False, action=argparse.BooleanOptionalAction, help="Save checkpoint after quantizing to args.log_dir.")
    parser.add_argument("--load-weights", type=Path, help="Path to laod weights")
    parser.add_argument("--save-model", default=False, action=argparse.BooleanOptionalAction, help="Save model in HF format after quantizing to args.log_dir.")

    args = parser.parse_args()

    # Pre-process some args
    model_args = {} if not args.model_args else simple_parse_args_string(args.model_args)
    quant_method = None if not args.quantize else quant_methods[args.quantize]
    quant_args = {} if not args.quantize_args else simple_parse_args_string(args.quantize_args)
    calibrate_args = {} if not args.calibrate_args else simple_parse_args_string(args.calibrate_args)
    nnq_args = {} if not args.nnq_args else simple_parse_args_string(args.nnq_args)
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
        nnq_args=nnq_args,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        device=args.device,
        batch_size=args.batch_size,
        parallelize=args.parallelize,
        generation_args=generation_args,
        log_dir=args.log_dir,
        append_results=args.append_results,
        overwrite_results=args.overwrite_results,
        save_weights=args.save_weights,
        load_weights=args.load_weights,
        save_model=args.save_model,
        bnb_args=bnb_args,
    )
