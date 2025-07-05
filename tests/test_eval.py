# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from parameterized import parameterized

import quantize
import bigcode_eval
from eval import main as eval

class TestEval(unittest.TestCase):
    # TODO: add tests for data.py, data_gptq.py, lm_eval, bigcode, single tasks, multiple tasks
    @parameterized.expand([
        (dtype, device)
        for dtype in ["float32", "float16", "bfloat16"]
        for device in ["cuda"] # TODO: support "cpu"
    ])
    def test_baseline(
        self,
        dtype="float16",
        device="cuda"
    ):
        model_name="facebook/opt-125m"
        tasks = ["piqa"]
        num_samples = 50
        min_expected_results = [0.50]

        results = eval(
            model_name=model_name,
            model_args={"torch_dtype":dtype},
            tasks=tasks,
            device=device,
            num_samples=num_samples,
            overwrite_results=True,
        )

        for task, min_expected in zip(tasks, min_expected_results):
            self.assertTrue(task in results)
            self.assertTrue(results[task]["acc,none"] > min_expected)

    @parameterized.expand([
        (quant_method)
        for quant_method in [quantize.intq, quantize.fp4, quantize.nf4, quantize.anyq]
    ])
    def test_quantize(
        self,
        quant_method=quantize.intq,
        group_size=64,
        device="cuda",
        dtype="bfloat16",
    ):
        model_name="facebook/opt-125m"
        tasks = ["piqa"]
        num_samples = 50
        min_expected_results = [0.50]

        results = eval(
            model_name=model_name,
            model_args={"torch_dtype":dtype},
            quant_method=quant_method,
            quant_args={"group_size":group_size},
            tasks=tasks,
            device=device,
            num_samples=num_samples,
            overwrite_results=True,
        )

        for task, min_expected in zip(tasks, min_expected_results):
            self.assertTrue(task in results)
            self.assertTrue(results[task]["acc,none"] > min_expected)

    @parameterized.expand([
        (n_bit)
        for n_bit  in [2, 4, 8]
    ])
    def test_intq(
        self,
        n_bit=4,
        group_size=64,
        dtype="float16",
        device="cuda"
    ):
        quant_method=quantize.intq

        model_name="facebook/opt-125m"
        tasks = ["piqa"]
        num_samples = 50
        min_expected_results = [0.50]

        results = eval(
            model_name=model_name,
            model_args={"torch_dtype":dtype},
            quant_method=quant_method,
            quant_args={"n_bit": n_bit, "group_size":group_size},
            tasks=tasks,
            device=device,
            num_samples=num_samples,
            overwrite_results=True,
        )

        for task, min_expected in zip(tasks, min_expected_results):
            self.assertTrue(task in results)
            self.assertTrue(results[task]["acc,none"] > min_expected)

    @parameterized.expand([
        (group_size)
        for group_size in [64, 128, 256]
    ])
    def test_nf4(
        self,
        group_size=128,
        dtype="float16",
    ):
        quant_method=quantize.nf4
        device="cuda"

        model_name="facebook/opt-125m"
        tasks = ["piqa"]
        num_samples = 50
        min_expected_results = [0.50]

        results = eval(
            model_name=model_name,
            model_args={"torch_dtype":dtype},
            quant_method=quant_method,
            quant_args={"group_size":group_size},
            tasks=tasks,
            device=device,
            num_samples=num_samples,
            overwrite_results=True,
        )

        for task, min_expected in zip(tasks, min_expected_results):
            self.assertTrue(task in results)
            self.assertTrue(results[task]["acc,none"] > min_expected)

    @parameterized.expand([
        (group_size)
        for group_size in [64, 128, 256]
    ])
    def test_fp4(
        self,
        group_size=128,
        dtype="float16",
    ):
        quant_method=quantize.fp4
        device="cuda"

        model_name="facebook/opt-125m"
        tasks = ["piqa"]
        num_samples = 50
        min_expected_results = [0.50]

        results = eval(
            model_name=model_name,
            model_args={"torch_dtype":dtype},
            quant_method=quant_method,
            quant_args={"group_size":group_size},
            tasks=tasks,
            device=device,
            num_samples=num_samples,
            overwrite_results=True,
        )

        for task, min_expected in zip(tasks, min_expected_results):
            self.assertTrue(task in results)
            self.assertTrue(results[task]["acc,none"] > min_expected)

    @parameterized.expand([
        (n_bit, group_size)
        for n_bit in [2, 3, 4]
        for group_size in [32, 64, 128]
    ])
    def test_anyq(
        self,
        n_bit=4,
        group_size=128,
        dtype="bfloat16",
        device="cuda"
    ):
        quant_method=quantize.anyq

        model_name="facebook/opt-125m"
        tasks = ["piqa"]
        num_samples = 50
        min_expected_results = [0.50]

        results = eval(
            model_name=model_name,
            model_args={"torch_dtype":dtype},
            quant_method=quant_method,
            quant_args={"n_bit":n_bit, "group_size":group_size},
            tasks=tasks,
            device=device,
            num_samples=num_samples,
            overwrite_results=True,
        )

        for task, min_expected in zip(tasks, min_expected_results):
            self.assertTrue(task in results)
            self.assertTrue(results[task]["acc,none"] > min_expected)
