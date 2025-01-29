import itertools
import gc
import json
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, OrderedDict
import torch
import numpy as np
from torch._inductor.utils import do_bench_using_profiling

dtype_str_to_torch = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

def benchmark_in_ms(f, warmup, iters, *args, **kwargs):
    for _ in range(warmup):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(iters):
        f(*args, **kwargs)

    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / float(iters)

# Source: https://github.com/drisspg/transformer_nuggets/blob/8b0a671b7b30cc7e186edd654f9c1565251b9b97/transformer_nuggets/utils/benchmark.py#L55
def benchmark_cuda_only_in_ms(func, warmup, iters, *args, **kwargs):
    no_args = lambda: func(*args, **kwargs)
    time = do_bench_using_profiling(no_args, warmup, iters)
    return time

# Source: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
def get_model_size(model):
    model_size = 0
    for name, child in model.named_children():
        model_size += sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(child.parameters(), child.buffers())
            ]
        )
    return model_size

# Source: https://github.com/huggingface/optimum/blob/main/tests/benchmark/memory_tracker.py
import os
import subprocess
from contextlib import contextmanager
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

class MemoryTracker:
    def __init__(self):
        self.peak_memory: int = 0
        self.device_index = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])

    @contextmanager
    def track(self, interval: float = 0.1):
        print(f"Tracking memory for device {self.device_index}")
        yield from self._track_peak_memory(interval)

    def _track_peak_memory(self, interval: float):
        child_connection, parent_connection = Pipe()
        # instantiate process
        mem_process: Process = PeakMemoryMeasureProcess(self.device_index, child_connection, interval)
        mem_process.start()
        # wait until we get memory
        parent_connection.recv()
        yield
        # start parent connection
        parent_connection.send(0)
        # receive peak memory
        self.peak_memory = parent_connection.recv()

class PeakMemoryMeasureProcess(Process):
    def __init__(self, device_index: int, child_connection: Connection, interval: float):
        super().__init__()
        self.device_index = device_index
        self.interval = interval
        self.connection = child_connection
        self.mem_usage = 0

    def run(self):
        self.connection.send(0)
        stop = False

        command = f"nvidia-smi --query-gpu=memory.used --format=csv --id={self.device_index}"

        while True:
            # py3nvml is broken since it outputs only the reserved memory, and nvidia-smi has only the MiB precision.
            gpu_mem_mb = subprocess.check_output(command.split()).decode("ascii").split("\n")[1].split()[0]
            gpu_mem_mb = int(gpu_mem_mb) * 1.048576
            self.mem_usage = max(self.mem_usage, gpu_mem_mb)

            if stop:
                break
            stop = self.connection.poll(self.interval)

        # send results to parent pipe
        self.connection.send(self.mem_usage)
        self.connection.close()

# Source: https://github.com/huggingface/optimum/blob/main/tests/benchmark/benchmark_gptq.py
def benchmark_memory(
    func,
    memory_tracker: MemoryTracker,
    warmup,
    *args,
    **kwargs,
):
    torch.cuda.empty_cache()
    gc.collect()

    print("Measuring peak memory...")
    with memory_tracker.track():
        for _ in range(warmup):
            func(*args, **kwargs)

        torch.cuda.synchronize()

    memory_stats = torch.cuda.memory_stats()

    peak_allocated_torch_mb = memory_stats["allocated_bytes.all.peak"] * 1e-6
    peak_reserved_torch_mb = memory_stats["reserved_bytes.all.peak"] * 1e-6

    peak_nvml_mb = memory_tracker.peak_memory

    # I am not sure whether we should substract here `inactive_split_bytes.all.peak` (not sure what it corresponds to, though it can get quite large, in the several GB).
    peak_external_mb = peak_nvml_mb - peak_reserved_torch_mb
    # assert peak_external_mb > 0

    # This formula is to confirm. We measure the actual allocated PyTorch memory, plus the additional non-PyTorch memory (as the CUDA context, CUDA extension device memory). We need to substract the PyTorch peak reserved memory since this one appears in the peak nvidia-smi/nvmlDeviceGetMemoryInfo.

    # NOTE: I verified this is only a ROUGH estimate. It may be better to use PYTORCH_NO_CUDA_MEMORY_CACHING=1 and just nvmlDeviceGetMemoryInfo.
    # We can actually doubt whether it make sense to try to estimate when we would OOM, given that different devices, CUDA version do have
    # a different CUDA context size.
    peak_memory_mb = peak_allocated_torch_mb + peak_external_mb

    print(f"DEBUG: peak allocated torch: {peak_allocated_torch_mb:.2f} MB")
    print(f"DEBUG: peak nvidia-smi/nvml: {peak_nvml_mb:.2f} MB")
    print(f"DEBUG: peak reserved torch: {peak_reserved_torch_mb:.2f} MB")
    print(f"DEBUG: peak external: {peak_external_mb:.2f} MB")
    print(f"DEBUG: global peak: {peak_memory_mb:.2f} MB")

    return peak_memory_mb

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Path, PurePosixPath)):
            return str(obj)
        elif hasattr(obj, '__name__') and hasattr(obj, '__code__'):
            # Object is a function
            return obj.__name__
        return super().default(obj)


def log(x):
    if isinstance(x, torch.Tensor):
        return torch.log(x)
    else:
        return np.log(x)

def get_max_n_numbers(arr, n):
    return np.sort(arr)[-n:]

def get_min_n_numbers(arr, n):
    return np.sort(arr)[:n]

def remove_all_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_backward_hooks"):
                child._backward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_hooks(child)

def trim_inputs(inputs, start_idx=None, end_idx=None):
    """
    Trim the inputs dictionary from a specific start index to an end index.
    Args:
        inputs (dict): The inputs dictionary returned by the tokenizer.
        start_idx (int, optional): The start index. Defaults to None.
        end_idx (int, optional): The end index. Defaults to None.
    Returns:
        dict: The trimmed inputs dictionary.
    """
    if start_idx is None and end_idx is None:
        return inputs  # No trimming needed
    elif start_idx is None:
        return {key: value[:, :end_idx] for key, value in inputs.items()}
    elif end_idx is None:
        return {key: value[:, start_idx:] for key, value in inputs.items()}
    else:
        return {key: value[:, start_idx:end_idx] for key, value in inputs.items()}
