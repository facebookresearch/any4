# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Optional
import argparse
import gc
import os
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GPTQConfig

from lm_eval.utils import simple_parse_args_string

from utils import benchmark_in_ms, benchmark_cuda_only_in_ms, get_model_size, benchmark_memory, MemoryTracker
from any4 import convert, quant_methods

default_device = "cuda" if torch.cuda.is_available() else "cpu"

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
    hidden_states = torch.randn(bs, seqlen, model.config.hidden_size, dtype=model.dtype, device=model.device)
    position_ids = torch.linspace(0, seqlen, steps=seqlen, device=model.device).repeat(bs, 1)

    # Benchmark
    ## Total Time
    model_time = benchmark_in_ms(model, n_warmup, n_iters, input_ids=input_ids, attention_mask=attention_mask)
    attn_time = benchmark_in_ms(model.model.layers[-1].self_attn, n_warmup, n_iters, hidden_states=hidden_states, position_ids=position_ids)
    mlp_time = benchmark_in_ms(model.model.layers[-1].mlp, n_warmup, n_iters, hidden_states)
    ## CUDA Time
    model_cuda_time = benchmark_cuda_only_in_ms(model, n_warmup, n_iters, input_ids=input_ids, attention_mask=attention_mask)
    attn_cuda_time = benchmark_cuda_only_in_ms(model.model.layers[-1].self_attn, n_warmup, n_iters, hidden_states=hidden_states, position_ids=position_ids)
    mlp_cuda_time = benchmark_cuda_only_in_ms(model.model.layers[-1].mlp, n_warmup, n_iters, hidden_states)
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
        qattn_time = benchmark_in_ms(qmodel.model.layers[-1].self_attn, n_warmup, n_iters, hidden_states=hidden_states, position_ids=position_ids)
        qmlp_time = benchmark_in_ms(qmodel.model.layers[-1].mlp, n_warmup, n_iters, hidden_states)
        ## CUDA Time
        qmodel_cuda_time = benchmark_cuda_only_in_ms(qmodel, n_warmup, n_iters, input_ids=input_ids, attention_mask=attention_mask)
        qattn_cuda_time = benchmark_cuda_only_in_ms(qmodel.model.layers[-1].self_attn, n_warmup, n_iters, hidden_states=hidden_states, position_ids=position_ids)
        qmlp_cuda_time = benchmark_cuda_only_in_ms(qmodel.model.layers[-1].mlp, n_warmup, n_iters, hidden_states)
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
    print(f"\tAttn: Total: {attn_time:.4f} ms\t\tCUDA: {attn_cuda_time:.4f} ms")
    print(f"\tMLP: Total: {mlp_time:.4f} ms\t\tCUDA: {mlp_cuda_time:.4f} ms")
    if quant_method:
        print("Quantized:")
        print(f"\tModel Size: {qmodel_size/1e9:.4f} GB\t\tCUDA Peak: {qmodel_peak_memory_mb/1e3:.4f} GB")
        print(f"\tModel: Total: {qmodel_time:.4f} ms\tCUDA: {qmodel_cuda_time:.4f} ms")
        print(f"\tAttn: Total: {qattn_time:.4f} ms\t\tCUDA: {qattn_cuda_time:.4f} ms")
        print(f"\tMLP: Total: {qmlp_time:.4f} ms\t\tCUDA: {qmlp_cuda_time:.4f} ms")


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
