# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Optional
import argparse
import gc
import os
import time
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GPTQConfig

from lm_eval.utils import simple_parse_args_string

from utils import benchmark_in_ms, benchmark_cuda_only_in_ms, get_model_size, benchmark_memory, MemoryTracker
from quantize import convert, quant_methods

def fmt(x): return f"{x:.4f}"
def fmt_gb(x): return f"{x / 2**30:.4f} GB"
def fmt_mb(x): return f"{x / 2**20:.4f} MB"

def print_stat(title, total, cuda):
    print(f"\t{title:<24}Total: {fmt(total):<10}CUDA: {fmt(cuda)}")

def print_percentage(title, total_time, model_time, cuda_time, model_cuda_time):
    print(f"\t{title:<24}Total: {fmt(total_time / model_time * 100):<10}%CUDA: {fmt(cuda_time / model_cuda_time * 100)}%")

def print_speedup(title, base_total, base_cuda, quant_total, quant_cuda):
    print(f"\t{title:<24}Total: {fmt(base_total / quant_total)}x\tCUDA: {fmt(base_cuda / quant_cuda)}x")

default_device = "cuda" if torch.cuda.is_available() else "cpu"

class HookBasedProfiler:
    def __init__(self, mode="cpu"):
        assert mode in ["cpu", "cuda"]
        self.mode = mode
        self.timings = defaultdict(list)
        self.hooks = []

    def register_hooks(self, model):
        def make_pre_hook(name):
            def pre_hook(module, input):
                if self.mode == "cpu":
                    module._start_time = time.perf_counter()
                else:
                    module._start_event = torch.cuda.Event(enable_timing=True)
                    module._start_event.record()
            return pre_hook

        def make_post_hook(name):
            def post_hook(module, input, output):
                if self.mode == "cpu":
                    elapsed = time.perf_counter() - module._start_time
                    self.timings[name].append(elapsed * 1000)
                else:
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed = module._start_event.elapsed_time(end_event)
                    self.timings[name].append(elapsed)
            return post_hook

        for i, layer in enumerate(model.model.layers):
            attn_name = f"attention_layer_{i}"
            mlp_name = f"mlp_layer_{i}"
            self.hooks.append(layer.self_attn.register_forward_pre_hook(make_pre_hook(attn_name)))
            self.hooks.append(layer.self_attn.register_forward_hook(make_post_hook(attn_name)))
            self.hooks.append(layer.mlp.register_forward_pre_hook(make_pre_hook(mlp_name)))
            self.hooks.append(layer.mlp.register_forward_hook(make_post_hook(mlp_name)))

    def run_profiling(self, model, input_ids, attention_mask, warmup=5, iters=10):
        model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        self.register_hooks(model)
        with torch.no_grad():
            for _ in range(iters):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        self.clear_hooks()

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def summarize(self):
        total_attn = 0
        total_mlp = 0
        for name, vals in self.timings.items():
            avg = np.mean(vals)
            if "attention" in name:
                total_attn += avg
            elif "mlp" in name:
                total_mlp += avg
        return {
            "attention_time": total_attn,
            "mlp_time": total_mlp,
            "ratio": total_attn / total_mlp if total_mlp > 0 else 0
        }

@torch.no_grad()
def benchmark_model(
    model_name: str,
    model_args: Dict = {},
    bs: int = 1,
    seqlen: int = 1,
    n_warmup: int = 50,
    n_iters: int = 100,
    device: str = default_device,
    quant_args: Optional[Dict] = {},
    quant_method: Optional[Callable] = None,
    bnb_args: Optional[Dict] = None,
    gptq_args: Optional[Dict] = None,
):
    # Pre-process some args
    if bnb_args:
        bnb_config = BitsAndBytesConfig(**bnb_args)
        model_args["quantization_config"] = bnb_config
    if gptq_args:
        # Defaults to consider setting
        # gptq_args["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        # (bits=4, dataset="c4", tokenizer=tokenizer)
        gptq_config = GPTQConfig(**gptq_args)
        model_args["quantization_config"] = gptq_config

    # Setup
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, **model_args)
    input_ids = torch.randint(0, model.config.vocab_size, (bs, seqlen), device=device)
    attention_mask = torch.ones((bs, seqlen), dtype=torch.long, device=device)

    # Benchmark using principled hook-based approach
    ## Total Time (end-to-end)
    model_time = benchmark_in_ms(model, n_warmup, n_iters, input_ids=input_ids, attention_mask=attention_mask)
    model_cuda_time = benchmark_cuda_only_in_ms(model, n_warmup, n_iters, input_ids=input_ids, attention_mask=attention_mask)

    ## Layer-wise profiling with hooks (most principled approach)
    profiler_cpu = HookBasedProfiler(mode="cpu")
    profiler_cpu.run_profiling(model, input_ids, attention_mask, n_warmup, n_iters)
    profile_cpu = profiler_cpu.summarize()

    profiler_cuda = HookBasedProfiler(mode="cuda")
    profiler_cuda.run_profiling(model, input_ids, attention_mask, n_warmup, n_iters)
    profile_cuda = profiler_cuda.summarize()

    ## Memory
    model_size = get_model_size(model)
    model_peak_memory_mb = benchmark_memory(model, MemoryTracker(), n_warmup, input_ids=input_ids, attention_mask=attention_mask)

    if quant_method:
        # Quantize
        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        qmodel = convert(model, layer_from=torch.nn.Linear, layer_to=quant_method, pseudo=False, **quant_args)

        print(qmodel)

        # Benchmark Quantized
        ## Total Time
        qmodel_time = benchmark_in_ms(qmodel, n_warmup, n_iters, input_ids=input_ids, attention_mask=attention_mask)
        qmodel_cuda_time = benchmark_cuda_only_in_ms(qmodel, n_warmup, n_iters, input_ids=input_ids, attention_mask=attention_mask)

        ## Layer-wise profiling for quantized model
        profiler_q_cpu = HookBasedProfiler(mode="cpu")
        profiler_q_cpu.run_profiling(qmodel, input_ids, attention_mask, n_warmup, n_iters)
        qprofile_cpu = profiler_q_cpu.summarize()

        profiler_q_cuda = HookBasedProfiler(mode="cuda")
        profiler_q_cuda.run_profiling(qmodel, input_ids, attention_mask, n_warmup, n_iters)
        qprofile_cuda = profiler_q_cuda.summarize()

        # Clean up Memory
        torch.cuda.empty_cache()
        gc.collect()
        # FIXME: CUDA Peak memory doesn't seem to reduce after we quantize. Perhaps we need to do something else.

        ## Memory
        qmodel_size = get_model_size(qmodel)
        qmodel_peak_memory_mb = benchmark_memory(qmodel, MemoryTracker(), n_warmup, input_ids=input_ids, attention_mask=attention_mask)

    print("Baseline:")
    print(f"\tModel Size:\t\t{fmt_gb(model_size)}\tCUDA Peak:\t{fmt_mb(model_peak_memory_mb)}")
    print_stat("Model:", model_time, model_cuda_time)
    print_stat("Attention:", profile_cpu['attention_time'], profile_cuda['attention_time'])
    print_stat("MLP:", profile_cpu['mlp_time'], profile_cuda['mlp_time'])
    print_stat("Attention/MLP Ratio:", profile_cpu['ratio'], profile_cuda['ratio'])
    print_percentage("Attention % of Model:", profile_cpu['attention_time'], model_time, profile_cuda['attention_time'], model_cuda_time)
    print_percentage("MLP % of Model:", profile_cpu['mlp_time'], model_time, profile_cuda['mlp_time'], model_cuda_time)

    if quant_method:
        print("Quantized:")
        print(f"\tModel Size:\t\t{fmt_gb(qmodel_size)}\tCUDA Peak:\t{fmt_mb(qmodel_peak_memory_mb)}")
        print_stat("Model:", qmodel_time, qmodel_cuda_time)
        print_stat("Attention:", qprofile_cpu['attention_time'], qprofile_cuda['attention_time'])
        print_stat("MLP:", qprofile_cpu['mlp_time'], qprofile_cuda['mlp_time'])
        print_stat("Attention/MLP Ratio:", qprofile_cpu['ratio'], qprofile_cuda['ratio'])
        print_percentage("Attention % of Model:", qprofile_cpu['attention_time'], qmodel_time, qprofile_cuda['attention_time'], qmodel_cuda_time)
        print_percentage("MLP % of Model:", qprofile_cpu['mlp_time'], qmodel_time, qprofile_cuda['mlp_time'], qmodel_cuda_time)

        print("Speedup Analysis:")
        print_speedup("Model Speedup:", model_time, model_cuda_time, qmodel_time, qmodel_cuda_time)
        print_speedup("Attention Speedup:", profile_cpu['attention_time'], profile_cuda['attention_time'], qprofile_cpu['attention_time'], qprofile_cuda['attention_time'])
        print_speedup("MLP Speedup:", profile_cpu['mlp_time'], profile_cuda['mlp_time'], qprofile_cpu['mlp_time'], qprofile_cuda['mlp_time'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark quantization on a language model.")

    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf", help="HuggingFace model name or path.")
    parser.add_argument("--model-args", type=str, default="torch_dtype=bfloat16", help="Comma separated string arguments for HuggingFace model.")
    parser.add_argument("--quantize", type=str, default=None, choices=quant_methods.keys(), help="Quantization method.")
    parser.add_argument("--quantize-args", type=str, help="Comma separated string args to pass to quantization method.")
    parser.add_argument("--bnb-args", type=str, help="Comma separated string args to pass to BitsAndBytes quantization config.")
    parser.add_argument("--gptq-args", type=str, help="Comma separated string args to pass to AutoGPTQ quantization config.")
    parser.add_argument("--device", type=str, default=default_device, help="Device to use.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size of sample input.")
    parser.add_argument("--seqlen", type=int, default=1, help="Sequence length of sample input.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations for benchmarking.")

    args = parser.parse_args()

    # Pre-process some args
    model_args = {} if not args.model_args else simple_parse_args_string(args.model_args)
    quant_method = None if not args.quantize else quant_methods[args.quantize]
    quant_args = {} if not args.quantize_args else simple_parse_args_string(args.quantize_args)
    bnb_args = None if not args.bnb_args else simple_parse_args_string(args.bnb_args)
    gptq_args = None if not args.gptq_args else simple_parse_args_string(args.gptq_args)

    # Run Evaluation
    benchmark_model(
        model_name=args.model_name,
        model_args=model_args,
        quant_method=quant_method,
        quant_args=quant_args,
        device=args.device,
        bs=args.batch_size,
        seqlen=args.seqlen,
        bnb_args=bnb_args,
        gptq_args=gptq_args,
        n_warmup=args.warmup,
        n_iters=args.iters,
    )
