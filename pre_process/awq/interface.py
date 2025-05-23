from .pre_quant import run_awq, apply_awq

# TODO: use our intq and adapt it to AWQ's integer quantization
# TODO: use our calibration function and adapt it to AwQ's calibration function
def awq(model, tokenizer, n_bit = 4, q_group_size=128, zero_point=True, **kwargs):
    q_config = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
    }
    awq_results = run_awq(
        model,
        tokenizer,
        w_bit=n_bit,
        q_config=q_config,
        **kwargs,
    )
    apply_awq(model, awq_results)
    return model
