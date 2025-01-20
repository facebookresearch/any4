import unittest
import torch

import any4
import bigcode_eval
from eval import main as eval

class TestEval(unittest.TestCase):
    def test_eval(
            self,
            model_name="facebook/opt-125m",
            dtype="float16",
            quant_method=any4.intq,
            group_size=64,
            tasks=["piqa"],
            device="cuda"
        ):
        results = eval(
            model_name=model_name,
            model_args={"dtype":dtype},
            quant_method=quant_method,
            quant_args={"group_size":group_size},
            tasks=tasks,
            device=device,
        )

        for task in tasks:
            self.assertTrue(task in results)

