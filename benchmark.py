import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import benchmark_in_ms
from any4 import convert, intq

# TODO: support int4, int8
def benchmark_model(model_name="meta-llama/Llama-3.2-1B", bs=1, seqlen=1, dtype=torch.bfloat16, quantizer=intq, n_bit=4, group_size=128, skip_modules="lm_head"):
    device = "cuda"
    w_inner_k = 4
    n_warmup=50
    n_iters=100

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device)
    input_ids = torch.randint(0, model.config.vocab_size, (bs, seqlen), device=device)
    attention_mask = torch.ones((bs, seqlen), dtype=torch.long, device=device)

    model_time = benchmark_in_ms(model, n_warmup, n_iters, input_ids=input_ids, attention_mask=attention_mask)

    qmodel = convert(model, layer_from=torch.nn.Linear, layer_to=quantizer, skip_modules=["lm_head"], group_size=group_size, n_bit=n_bit, pseudo=True)

    qmodel_time = benchmark_in_ms(qmodel, n_warmup, n_iters, input_ids=input_ids, attention_mask=attention_mask)
    print(f"Baseline:\t{model_time}")
    print(f"Quantized:\t{qmodel_time}")


# TODO: add argument parsing
benchmark_model()