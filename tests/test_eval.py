import unittest
from parameterized import parameterized

import any4
import bigcode_eval
from eval import main as eval

class TestEval(unittest.TestCase):
    @parameterized.expand([
        (dtype, quant_method, group_size)
        for dtype in ["bfloat16"] # ["float32", "float16", "bfloat16"]
        for quant_method in [None, any4.intq, any4.fp4, any4.nf4, any4.anyq]
        for group_size in [64]#  [32, 64, 128]
        for device in ["cuda"] # TODO: support "cpu"
    ])
    def test_eval_quantize(
            self,
            dtype="float16",
            quant_method=any4.intq,
            group_size=64,
            device="cuda"
        ):
        model_name="facebook/opt-125m"
        tasks = ["piqa"]
        num_samples = 50
        min_expected_results = [0.60]

        results = eval(
            model_name=model_name,
            model_args={"dtype":dtype},
            quant_method=quant_method,
            quant_args={"group_size":group_size},
            tasks=tasks,
            device=device,
            num_samples=num_samples,
            overwrite_results=True,
        )

        for task, min_expected in zip(tasks, min_expected_results):
            self.assertTrue(task in results)
            # self.assertTrue(results[task]["acc,none"] > min_expected)

