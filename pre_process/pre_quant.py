import os
import torch
from accelerate import (
    infer_auto_device_map,
    dispatch_model,
)

from .awq.pre_quant import run_awq, apply_awq

# TODO: use our intq and adapt it to AWQ's integer quantization
# TODO: use our calibration function and adapt it to AwQ's calibration function
def awq(
    model,
    tokenizer,
    n_bit = 4,
    q_group_size=128,
    zero_point=True,
    numeric_type="int",
    n_samples=128,
    seqlen=512,
    calib_data="pileval",
    load_awq=None,
    dump_awq=None,
    **kwargs
):
    orig_device_map = infer_auto_device_map(model)

    q_config = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
        "numeric_type": numeric_type,
    }

    if load_awq:
            awq_results = torch.load(load_awq, map_location="cpu")
    else:
        awq_results = run_awq(
            model,
            tokenizer,
            w_bit=n_bit,
            q_config=q_config,
            n_samples=n_samples,
            seqlen=seqlen,
            calib_data=calib_data,
            **kwargs,
        )

    if dump_awq:
        dirpath = os.path.dirname(dump_awq)
        os.makedirs(dirpath, exist_ok=True)

        torch.save(awq_results, dump_awq)
        print("AWQ results saved at", dump_awq)
        print("Exiting...")
        exit(0)

    apply_awq(model, awq_results)

    dispatch_model(model, orig_device_map)
    return model


pre_quant_methods = {
    "awq": awq,
}