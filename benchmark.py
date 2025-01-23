from typing import Callable, Dict, Optional
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from lm_eval.utils import simple_parse_args_string

from utils import benchmark_in_ms
from any4 import convert, quant_methods

default_device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: support int4, int8
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
    bnb_args: Optional[Dict] = {},
):
    # Pre-process some args
    if bnb_args:
        bnb_config = BitsAndBytesConfig(**bnb_args)
        model_args["quantization_config"] = bnb_config

    # Setup
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, **model_args)
    input_ids = torch.randint(0, model.config.vocab_size, (bs, seqlen), device=model.device)
    attention_mask = torch.ones((bs, seqlen), dtype=torch.long, device=model.device)

    # Benchmark
    model_time = benchmark_in_ms(model, n_warmup, n_iters, input_ids=input_ids, attention_mask=attention_mask)
    layer = model.model.layers[31].mlp
    layer_time = benchmark_in_ms(model.model.layers[31].mlp, n_warmup, n_iters, torch.randn(bs, seqlen, model.config.hidden_size, dtype=model.dtype, device=model.device))

    if quant_method:
        # Quantize
        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        qmodel = convert(model, layer_from=torch.nn.Linear, layer_to=quant_method, pseudo=False, **quant_args)

        print(qmodel)

        # Benchmark Quantized
        qmodel_time = benchmark_in_ms(qmodel, n_warmup, n_iters, input_ids=input_ids, attention_mask=attention_mask)
        qlayer = model.model.layers[31].mlp
        qlayer_time = benchmark_in_ms(qlayer, n_warmup, n_iters, torch.randn(bs, seqlen, model.config.hidden_size, dtype=model.dtype, device=model.device))

    # Log Results
    print(f"Baseline:\t{model_time} ms, {layer_time} ms")
    if quant_method:
        print(f"Quantized:\t{qmodel_time} ms, {qlayer_time} ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark quantization on a language model.")

    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf", help="HuggingFace model name or path.")
    parser.add_argument("--model-args", type=str, default="torch_dtype=bfloat16", help="Comma separated string arguments for HuggingFace model.")
    parser.add_argument("--quantize", type=str, default="intq", choices=quant_methods.keys(), help="Quantization method.")
    parser.add_argument("--quantize-args", type=str, help="Comma separated string args to pass to quantization method.")
    parser.add_argument("--bnb-args", type=str, help="Comma separated string args to pass to BitsAndBytes quantization config.")
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
        n_warmup=args.warmup,
        n_iters=args.iters,
    )

