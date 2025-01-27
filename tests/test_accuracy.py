import unittest
from parameterized import parameterized

import any4
from eval import main as eval

class TestEval(unittest.TestCase):
    def test_any4_paper(
        self,
    ):
        # python eval.py --model-name meta-llama/Llama-3.2-1B --model-args dtype=bfloat16 --quantize anyq --quantize-args group_size=128,skip_modules=lm_head,scale_sample_weight=True,sample_weight=./profiles/meta-llama/Llama-3.2-1B/prompt/prompt.pt --log-dir ./logs/meta-llama/Llama-3.2-1B/llama3.2-1b-any4-g128-skip-lm-head-prompt-calib-s/
        model_name="meta-llama/Llama-3.2-1B"
        device="cuda"
        dtype="bfloat16"
        tasks = ["c4"]
        expected_results = [13.952826499938965]

        results = eval(
            model_name=model_name,
            model_args={"dtype":dtype},
            quant_method=any4.anyq,
            quant_args={
                "group_size":128,
                "skip_modules":"lm_head",
                "scale_sample_weight":True,
                "sample_weight":"./profiles/meta-llama/Llama-3.2-1B/prompt/prompt.pt",
            },
            tasks=tasks,
            device=device,
            overwrite_results=True,
        )

        for task, expected in zip(tasks, expected_results):
            self.assertTrue(task in results)
            self.assertEqual(results[task], expected)

    def test_any4_opt_125m(
        self,
    ):
        # python eval.py --model-name facebook/opt-125m --quantize anyq
        model_name="facebook/opt-125m"
        device="cuda"
        tasks = ["c4"]
        expected_results = [27.1861515045166]

        results = eval(
            model_name=model_name,
            quant_method=any4.anyq,
            tasks=tasks,
            device=device,
            overwrite_results=True,
        )

        for task, expected in zip(tasks, expected_results):
            self.assertTrue(task in results)
            self.assertEqual(results[task], expected)