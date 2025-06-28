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
from any4 import convert, quant_methods

default_device = "cuda" if torch.cuda.is_available() else "cpu"

class LayerProfiler:
    """Hook-based profiler for transformer layers"""
    def __init__(self):
        self.timings = defaultdict(list)
        self.hooks = []

    def register_hooks(self, model):
        """Register forward hooks on all attention and MLP layers"""
        def make_hook(name):
            def hook(module, input, output):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if not hasattr(module, '_start_time'):
                    return
                elapsed = time.perf_counter() - module._start_time
                self.timings[name].append(elapsed * 1000)  # Convert to ms
            return hook

        def make_pre_hook(name):
            def pre_hook(module, input):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                module._start_time = time.perf_counter()
            return pre_hook

        # Register hooks for each layer
        for i, layer in enumerate(model.model.layers):
            # Attention hooks
            attn_name = f"attention_layer_{i}"
            layer.self_attn.register_forward_pre_hook(make_pre_hook(attn_name))
            layer.self_attn.register_forward_hook(make_hook(attn_name))

            # MLP hooks
            mlp_name = f"mlp_layer_{i}"
            layer.mlp.register_forward_pre_hook(make_pre_hook(mlp_name))
            layer.mlp.register_forward_hook(make_hook(mlp_name))

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_summary(self):
        """Get aggregated timing statistics"""
        summary = {}
        total_attn_time = 0
        total_mlp_time = 0

        for name, times in self.timings.items():
            if len(times) > 0:
                summary[name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'total': np.sum(times),
                    'count': len(times)
                }

                if 'attention' in name:
                    total_attn_time += summary[name]['mean']
                elif 'mlp' in name:
                    total_mlp_time += summary[name]['mean']

        return {
            'layer_stats': summary,
            'total_attention_time': total_attn_time,
            'total_mlp_time': total_mlp_time,
            'attention_mlp_ratio': total_attn_time / total_mlp_time if total_mlp_time > 0 else 0
        }

def profile_model_with_hooks(model, input_ids, attention_mask, n_warmup=10, n_iters=50):
    """Profile model using hook-based approach for layer-wise breakdown"""
    profiler = LayerProfiler()
    profiler.register_hooks(model)

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        # Actual profiling runs
        for _ in range(n_iters):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    profiler.clear_hooks()
    return profiler.get_summary()

# TODO: support int4, int8
@torch.no_grad()
def benchmark_model(
    model_name: str,
    model_args: Dict ={},
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
    input_ids = torch.randint(0, model.config.vocab_size, (bs, seqlen), device=model.device)
    attention_mask = torch.ones((bs, seqlen), dtype=torch.long, device=model.device)

    # Benchmark using principled hook-based approach
    ## Total Time (end-to-end)
    model_time = benchmark_in_ms(model, n_warmup, n_iters, input_ids=input_ids, attention_mask=attention_mask)
    model_cuda_time = benchmark_cuda_only_in_ms(model, n_warmup, n_iters, input_ids=input_ids, attention_mask=attention_mask)

    ## Layer-wise profiling with hooks (most principled approach)
    baseline_profile = profile_model_with_hooks(model, input_ids, attention_mask, n_warmup, n_iters)

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
        quantized_profile = profile_model_with_hooks(qmodel, input_ids, attention_mask, n_warmup, n_iters)

        # Clean up Memory
        torch.cuda.empty_cache()
        gc.collect()
        ## Memory
        qmodel_size = get_model_size(qmodel)
        qmodel_peak_memory_mb = benchmark_memory(qmodel, MemoryTracker(), n_warmup, input_ids=input_ids, attention_mask=attention_mask)

    # Log Results
    print("Baseline:")
    print(f"\tModel Size: {model_size/1e9:.4f} GB\t\tCUDA Peak: {model_peak_memory_mb/1e3:.4f} GB")
    print(f"\tModel: Total: {model_time:.4f} ms\tCUDA: {model_cuda_time:.4f} ms")
    print(f"\tTotal Attention Time: {baseline_profile['total_attention_time']:.4f} ms")
    print(f"\tTotal MLP Time: {baseline_profile['total_mlp_time']:.4f} ms")
    print(f"\tAttention/MLP Ratio: {baseline_profile['attention_mlp_ratio']:.2f}")
    print(f"\tAttention % of Total: {(baseline_profile['total_attention_time']/model_time)*100:.1f}%")
    print(f"\tMLP % of Total: {(baseline_profile['total_mlp_time']/model_time)*100:.1f}%")

    if quant_method:
        print("Quantized:")
        print(f"\tModel Size: {qmodel_size/1e9:.4f} GB\t\tCUDA Peak: {qmodel_peak_memory_mb/1e3:.4f} GB")
        print(f"\tModel: Total: {qmodel_time:.4f} ms\tCUDA: {qmodel_cuda_time:.4f} ms")
        print(f"\tTotal Attention Time: {quantized_profile['total_attention_time']:.4f} ms")
        print(f"\tTotal MLP Time: {quantized_profile['total_mlp_time']:.4f} ms")
        print(f"\tAttention/MLP Ratio: {quantized_profile['attention_mlp_ratio']:.2f}")
        print(f"\tAttention % of Total: {(quantized_profile['total_attention_time']/qmodel_time)*100:.1f}%")
        print(f"\tMLP % of Total: {(quantized_profile['total_mlp_time']/qmodel_time)*100:.1f}%")

        # Speedup analysis
        print("Speedup Analysis:")
        print(f"\tModel Speedup: {model_time/qmodel_time:.2f}x")
        print(f"\tAttention Speedup: {baseline_profile['total_attention_time']/quantized_profile['total_attention_time']:.2f}x")
        print(f"\tMLP Speedup: {baseline_profile['total_mlp_time']/quantized_profile['total_mlp_time']:.2f}x")


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
