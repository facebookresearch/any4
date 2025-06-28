# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import gc
import json
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, OrderedDict
import torch
import numpy as np
import unittest
from torch._inductor.utils import do_bench_using_profiling

dtype_str_to_torch = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

def import_or_skip(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None

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

def benchmark_cuda_only_in_ms(func, warmup, iters, *args, **kwargs):
    """
    Measure GPU execution time using CUDA events
    This excludes most CPU overhead but may include some kernel launch latency
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    # Create events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Record timing for all iterations
    start_event.record()
    for _ in range(iters):
        func(*args, **kwargs)
    end_event.record()

    torch.cuda.synchronize()

    # Calculate average time per iteration
    total_time = start_event.elapsed_time(end_event)  # Returns milliseconds
    return total_time / iters

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

import torch

def assert_close(
    actual,
    expected,
    *,
    atol=None,
    rtol=None,
    allowed_violations_factor=10.0,
    allowed_violations=0,
):
    """
    Like torch.testing.assert_close, but allows up to `allowed_violations` elements 
    to exceed tight tolerance, as long as they are within 
    allowed_violations_factor * (atol + rtol * |expected|).
    """
    # Default tolerances per dtype from torch.testing.assert_close (2025)
    DTYPE_TOL = {
        torch.float16:   (1e-5, 1e-3),
        torch.bfloat16:  (1e-5, 1.6e-2),
        torch.float32:   (1e-5, 1.3e-6),
        torch.float64:   (1e-7, 1e-7),
        torch.complex32: (1e-5, 1e-3),
        torch.complex64: (1e-5, 1.3e-6),
        torch.complex128:(1e-7, 1e-7),
        torch.quint8:    (1e-5, 1.3e-6),
        torch.quint2x4:  (1e-5, 1.3e-6),
        torch.quint4x2:  (1e-5, 1.3e-6),
        torch.qint8:     (1e-5, 1.3e-6),
        torch.qint32:    (1e-5, 1.3e-6),
    }

    def get_default_tols(dtype):
        return DTYPE_TOL.get(dtype, (0.0, 0.0))

    actual = torch.as_tensor(actual)
    expected = torch.as_tensor(expected)

    # Determine tolerances if not specified
    if atol is None and rtol is None:
        a0, r0 = get_default_tols(actual.dtype)
        a1, r1 = get_default_tols(expected.dtype)
        atol = max(a0, a1)
        rtol = max(r0, r1)

    # Compute element-wise differences
    diff = torch.abs(actual - expected)
    tight = diff <= (atol + rtol * torch.abs(expected))
    relaxed = diff <= (allowed_violations_factor * (atol + rtol * torch.abs(expected)))

    too_relaxed = (~relaxed).sum().item()
    too_tight = (~tight).sum().item()

    if too_relaxed > 0:
        idx = (~relaxed).nonzero(as_tuple=True)
        val_exact = diff[idx][0].item()
        raise AssertionError(
            f"{too_relaxed} values exceed even relaxed tolerance; "
            f"e.g. diff[{idx[0][0]}] = {val_exact:.3e} > "
            f"{allowed_violations_factor:.1f} * (atol + rtol*|expected|)"
        )
    if too_tight > allowed_violations:
        raise AssertionError(
            f"{too_tight} values exceed tight tolerance "
            f"(allowed {allowed_violations}), "
            f"e.g. {allowed_violations + 1} exceed {(atol + rtol * torch.abs(expected)).max().item():.3e}"
        )

    return  # Passed

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
    if (start_idx is None or start_idx == 0) and (end_idx is None or end_idx == -1):
        return inputs  # No trimming needed
    elif start_idx is None or start_idx == 0:
        return {key: value[:, :end_idx] for key, value in inputs.items()}
    elif end_idx is None or end_idx == -1:
        return {key: value[:, start_idx:] for key, value in inputs.items()}
    else:
        return {key: value[:, start_idx:end_idx] for key, value in inputs.items()}
