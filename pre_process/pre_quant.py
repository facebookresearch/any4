from accelerate import (
    infer_auto_device_map,
    dispatch_model,
)
from .awq.pre_quant import run_awq, apply_awq

# TODO: use our intq and adapt it to AWQ's integer quantization
# TODO: use our calibration function and adapt it to AwQ's calibration function
def awq(model, tokenizer, n_bit = 4, q_group_size=128, zero_point=True, numeric_type="int", **kwargs):
    orig_device_map = infer_auto_device_map(model)

    q_config = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
        "numeric_type": numeric_type,
    }
    awq_results = run_awq(
        model,
        tokenizer,
        w_bit=n_bit,
        q_config=q_config,
        **kwargs,
    )
    apply_awq(model, awq_results)

    dispatch_model(model, orig_device_map)
    return model


pre_quant_methods = {
    "awq": awq,
}