# any4 + tinygemm

This repo contains the tinygemm low-letency / small batch size Nvidia GPU GEMM library which implements bf16/fp16, int4 grouped quantization, any4 grouped quantization and MX4 quantization, and the code containing the technique to learn any4 quantization codes. 

This code release is meant to accompany our paper [*any4: Learned 4-bit Numeric Representation for LLMs*](https://openreview.net/forum?id=tJmhOPkWCj), **ICML 2025**, by [Mostafa Elhoushi](https://github.com/mostafaelhoushi) and [Jeff Johnson](https://github.com/wickedfoo).

The techique and code for learning any4 representations and quantizing a model was authored by Mostafa Elhoushi (previously Meta FAIR SysML research). The Nvidia GPU tinygemm library was authored by Jeff Johnson (currently Meta FAIR SysML reserach).  An extremely early version of the tinygemm kernels without any4/MX4 support [were upstreamed to PyTorch core in Q4 2023](https://github.com/pytorch/pytorch/pull/110914) for use by the Torch compiler.

## What is any4?
<div align="center">
  <img src="https://github.com/user-attachments/assets/bf4a75f0-a271-43df-bdfe-a5296ca62407" width="500">
</div>
There is a wide variety of 4-bit numerical formats implemented on CPU/GPU for ML inference, such as uniform int4 quantization, "fp4", NF4, AF4 and the like, all of which have the dequantization values fixed a priori. any4 substitutes a lookup table (LUT) to translate the 16 possible 4-bit quantization codes to any arbitrary bfloat16 or float16 floating-point value, and this GPU in-register LUT is used at dequantization time. Each row of a weight matrix can use a different 16 x bfloat16/float16 LUT, so the quantization codes can be tailored to each row of a matrix. k-means or neural network based clustering is used to learn the any4 LUTs based off the weight matrix data distribution. Effectively, any4 is 4-bit grouped quantization like typical int4 quantization, just that instead of the code dequantization values prior to scale and offset being integers in the range [-8, +7] or [0, 15], the dequantization values are here arbitrary floating point values from the LUT. any4 is thus a very efficient means of implementing NormalFloat4 (NF4) or AbnormalFloat4 (AF4), whose initial implementations used GPU unfriendly deeply-nested if/else blocks or switch statements.

## What is tinygemm
The tinygemm low-latency GPU GEMM library implements any4 quantization. Learning the any4 quantization codes is not part of tinygemm itself. While tinygemm supports most any arbitrary GEMM size (assuming the reduction/k dimension is a multiple of 16 or 32), it is primarily meant for matrix multiplication problems where one of the `m` or `n` problem dimensions (for a `(m x k) x (n x k)^t` matrix multiplication) is *smaller*  than a GPU tensor core tile size (e.g., 1 <= m <= 16 or 1 <= n <= 8), usually applied to the "activation" vector in neural networks. 

tinygemm has two different modes, one that computes `Y = X W^t` and the other that computes `Y = (W X^t)^t` (both produce the same result, just whether the "weight" matrix is the "A" or "B" matrix for tensor core usage). All needed transpositions are performed on the fly as needed by tinygemm. For the m16n8k16 A100+ bf16/fp16 tensor core tile, the "A" matrix tile size is 16 x 16 and "B" is 8 x 16 (or 16 x 8 as desired). Putting activations (e.g., a `1 x k` matrix) on the right and weight on the left (so that the `1 x k` matrix will occupy the "B" tile) ensures that we will be running the tensor core unit at 1/8th throughput rather than 1/16th throughput. We have found that using the tensor core in this fashion for e.g., GEMV is pretty fast. tinygemm does not use larger tensor core multiplication primitives (again, because a typical use case is something like a `(1 x k) x (n x k)` GEMM. All matrices presented to tinygemm must be row-major with the reduction dimension `k` being innermost.

To further reduce latency, it is best to lay out weight matrices in "tensor core" format, so no shared memory transposition is needed. Because there is also no reuse of the weight matrix in usual circumstances, we avoid shared memory entirely for buffering or transposition and the kernels load data directly from gmem into registers (though with some degree of multi-buffering into registers, but nvcc/ptxas' register usage heuristics are at odds with this; loads from gmem into a register are still asynchronous until the point of first use).

Please defer to the paper for additional details.

## Getting Started

1. Clone Repo
```
git clone git@github.com:fairinternal/any4.git

cd any4
```

2. Setup Environment
```
conda create --name any4 python=3.10
conda activate any4

pip install -r requirements.txt
```

3. Access Models

Some models (e.g., Llama) require permission. Follow these steps to access them:

a. Submit a request to access a Llama checkpoint, e.g., https://huggingface.co/meta-llama/Llama-3.2-1B.

b. Setup Hugging Face token access by following the steps described [here](https://huggingface.co/docs/hub/en/security-tokens).

c. Then you will be able to login to Hugging Face by running the cell below and entering the token you obtain from Step b. above:
```
huggingface-cli login
```

4. Install tinygemm kernels
```
cd tinygemm_lib
python setup.py install
cd ..
```

## Run
Most of the scripts below will run baseline fp16 model by default. To quantize add the following arguments:
- `--model-args`: pass in any args that are passed to Hugging Face's [`from_pretrained()`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained) function, including `load_in_4bit` and `load_in_8bit`.
- `--quantize`: implements different (fake) quantization algorithms implemented in this codebase. It can take: `intq` (integer quantization), `fp4` (4-bit float quantization), `nf4` (4-bit normal float quantization), `anyq` (proposed lookup table quantization).
    - `--quantize-args`: comma-separated arguments to pass to a quantization algorithm, e.g., `--quantize-args n_bit=4,group_size=128` will perform 4-bit quantization with group size 128.
- `--bnb-args`: comma-separated arguments to pass to [`BitsAndBytesConfig`](https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/quantization#transformers.BitsAndBytesConfig), e.g., `load_in_4bit=True,bnb_4bit_compute_dtype=fp32`

### Quick Example
To run a simple text generation (with and without) quantization example script that you can try and edit:
```
python example.py
```

### Evaluation
Evaluate a model (with or without quantization) on downstream tasks.
- Baseline fp16 model:
```
python eval.py --model-name facebook/opt-125m --tasks piqa
```
- Quantized int4 model:
```
python eval.py --model-name facebook/opt-125m --quantize intq --quantize-args n_bit=4,skip_modules=lm_head --tasks piqa
```
- Quantized any4 model:
```
python eval.py --model-name facebook/opt-125m --quantize anyq --quantize-args n_bit=4,sample_weight=calibrate,scale_sample_weight=True,skip_modules=lm_head --tasks piqa
```

Arguments:
- `--tasks`: by default it runs a large number of natural language, coding, and perplexity evaluation tasks:
    - You can specify a space separate list of tasks, e.g., `--tasks piqa mbpp`.
    - You can pass in any task supported by [Eleuther LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness), [BigCode Eval Harness](https://github.com/bigcode-project/bigcode-evaluation-harness), and any Hugging Face dataset to measure its perplexity.

### Analyze
To analyze weights and mean square errors on weights and activations between baseline model and quantized model at each layer:
```
python analyze.py --model-name meta-llama/Llama-3.2-1B --quantize nf4 --quantize-args skip_modules=lm_head
```

### Calibrate
To pass a dataset or pompt over a model and store output activations of each layer:
```
python calibrate.py --model-name meta-llama/Llama-3.2-1B --dataset cerebras/SlimPajama-627B --num-batches 10
```

### Diff
To pass a prompt to both a baseline model and quantized model and measure the mean square error along each layer:
```
python analyze.py --model-name meta-llama/Llama-3.2-1B --quantize anyq
```

## Test
To run all unit test cases:
```
python -m pytest .
```

## Experiments
In this section we provide the results in the paper and the command to reproduce each result.

### Main Results
#### Llama3.2 1B
|                 | WikiText-2↓ | C4↓   | PTB↓  | CodeParrot↓ | HumanEval↑ | MBPP↑ | MMLU↑  | HellaSwag↑ | GSM8K↑ | BBH↑  |
| --------------- | ----------- | ----- | ----- | ----------- | ---------- | ----- | ------ | ---------- | ------ | ----- |
| FP16 [[1]](#f1) | 9.76        | 12.77 | 16.56 | 3.49        | 16.46%     | 21.4% | 36.1%  | 47.7%      | 6.60%  | 31.1% |
| INT4 [[2]](#f2) | 11.89       | 15.74 | 20.32 | 4.08        | 9.76%      | 11.4% | 30.1%  | 44.7%      | 3.18%  | 26.2% |
| FP4 [[3]](#f3)  | 13.01       | 17.11 | 21.89 | 4.28        | 8.54%      | 5.8%  | 29.3%  | 43.6%      | 2.27%  | 23.3% |
| NF4 [[4]](#f4)  | 10.99       | 14.63 | 18.78 | 3.82        | 13.4%      | 13.8% | 33.3%  | 45.8%      | 3.65%  | 26.8% |
| ANY4 [[5]](#f5) | 10.63       | 13.95 | 17.94 | 3.71        | 11.0%      | 18.6% | 32.9%  | 46.7%      | 3.71%  | 29.0% |

<details open>
<summary>Commands to reproduce results:</summary>

1. <span id="f1"></span> `python eval.py --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log--dir ./logs/llama3.2-1b/bf16`
2. <span id="f2"></span> `python eval.py --quantize intq --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama3.2-1b/int4`
3. <span id="f3"></span> `python eval.py --quantize fp4 --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama3.2-1b/fp4`
4. <span id="f4"></span> `python eval.py --quantize nf4 --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama3.2-1b/nf4`
5. <span id="f5"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama3.2-1b/any4`

</details>

#### Llama3 8B
|                   | WikiText-2↓ | C4↓   | PTB↓  | CodeParrot↓ | HumanEval↑ | MBPP↑ | MMLU↑  | HellaSwag↑ | GSM8K↑ | BBH↑  |
| ----------------- | ----------- | ----- | ----- | ----------- | ---------- | ----- | ------ | ---------- | ------ | ----- |
| FP16 [[6]](#f6)   | 6.14        | 8.93  | 10.59 | 2.54        | 29.3%      | 41.4% | 62.0%  | 60.1%      | 50.7%  | 62.8% |
| INT4 [[7]](#f7)   | 6.87        | 9.89  | 11.37 | 2.83        | 23.2%      | 35.4% | 59.6%  | 58.6%      | 40.6%  | 58.5% |
| FP4 [[8]](#f8)    | 7.10        | 10.22 | 11.81 | 2.89        | 22.0%      | 36.8% | 57.1%  | 58.5%      | 35.0%  | 53.2% |
| NF4 [[9]](#f9)    | 6.63        | 9.52  | 11.14 | 2.72        | 23.2%      | 39.2% | 60.7%  | 59.1%      | 41.1%  | 59.0% |
| ANY4 [[10]](#f10) | 6.51        | 9.40  | 11.07 | 2.68        | 21.3%      | 39.2% | 61.0%  | 59.5%      | 41.7%  | 59.2% |

<details open>
<summary>Commands to reproduce results:</summary>

6. <span id="f6"></span> `python eval.py --model-name meta-llama/Meta-Llama-3-8B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log--dir ./logs/llama3-8b/bf16`
7. <span id="f7"></span> `python eval.py --quantize intq --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Meta-Llama-3-8B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama3-8b/int4`
8. <span id="f8"></span> `python eval.py --quantize fp4 --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Meta-Llama-3-8B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama3-8b/fp4`
9. <span id="f9"></span> `python eval.py --quantize nf4 --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Meta-Llama-3-8B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama3-8b/nf4`
10. <span id="f10"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name meta-llama/Meta-Llama-3-8B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama3-8b/any4`

</details>

#### Llama3 70B
|                    | WikiText-2↓ | C4↓   | PTB↓  | CodeParrot↓ | HumanEval↑ | MBPP↑ | MMLU↑  | HellaSwag↑ | GSM8K↑ | BBH↑  |
| ------------------ | ----------- | ----- | ----- | ----------- | ---------- | ----- | ------ | ---------- | ------ | ----- |
| FP16 [[11]](#f11)  | 2.86        | 6.77  | 8.16  | 1.91        | 17.7%      | 60.8% | 75.4%  | 66.3%      | 80.6%  | 82.4% |
| INT4 [[12]](#f12)  | 3.63        | 7.97  | 8.86  | 2.21        | 18.3%      | 45.0% | 73.0%  | 66.2%      | 73.9%  | 78.4% |
| FP4 [[13]](#f13)   | 3.94        | 7.76  | 8.99  | 2.17        | 22.0%      | 50.8% | 71.9%  | 65.6%      | 75.3%  | 77.9% |
| NF4 [[14]](#f14)   | 3.43        | 7.67  | 8.84  | 2.15        | 18.9%      | 39.6% | 73.7%  | 66.1%      | 75.9%  | 79.3% |
| ANY4 [[15]](#f15)  | 3.20        | 7.01  | 8.33  | 1.99        | 17.1%      | 57.4% | 75.1%  | 66.1%      | 78.5%  | 81.8% |

<details open>
<summary>Commands to reproduce results:</summary>

11. <span id="f11"></span> `python eval.py --model-name meta-llama/Meta-Llama-3-70B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama3-70b/bf16`
12. <span id="f12"></span> `python eval.py --quantize intq --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Meta-Llama-3-70B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama3-70b/int4`
13. <span id="f13"></span> `python eval.py --quantize fp4 --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Meta-Llama-3-70B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama3-70b/fp4`
14. <span id="f14"></span> `python eval.py --quantize nf4 --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Meta-Llama-3-70B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama3-70b/nf4`
15. <span id="f15"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name meta-llama/Meta-Llama-3-70B --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama3-70b/any4`

</details>

**Llama2 7B**
|                    | WikiText-2↓ | C4↓   | PTB↓   | CodeParrot↓ | HumanEval↑ | MBPP↑ | MMLU↑  | HellaSwag↑ | GSM8K↑ | BBH↑  |
| ------------------ | ----------- | ----- | ------ | ----------- | ---------- | ----- | ------ | ---------- | ------ | ----- |
| FP16 [[16]](#f16)  | 5.47        | 6.97  | 20.83  | 2.54        | 17.1%      | 20.0% | 41.3%  | 57.2%      | 13.6%  | 39.8% |
| INT4 [[17]](#f17)  | 5.74        | 7.30  | 24.00  | 2.63        | 14.0%      | 18.2% | 38.1%  | 56.4%      | 10.6%  | 36.5% |
| FP4 [[18]](#f18)   | 5.83        | 7.37  | 22.57  | 2.65        | 11.0%      | 16.8% | 36.5%  | 56.6%      | 11.2%  | 35.5% |
| NF4 [[19]](#f19)   | 5.66        | 7.19  | 22.82  | 2.60        | 11.6%      | 19.2% | 37.4%  | 56.8%      | 10.2%  | 36.8% |
| ANY4 [[20]](#f20)  | 5.59        | 7.10  | 21.23  | 2.57        | 14.0%      | 18.4% | 40.3%  | 56.7%      | 12.7%  | 36.9% |

<details open>
<summary>Commands to reproduce results:</summary>

16. <span id="f16"></span> `python eval.py --model-name meta-llama/Llama-2-7b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-7b/fp16`
17. <span id="f17"></span> `python eval.py --quantize intq --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Llama-2-7b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-7b/int4`
18. <span id="f18"></span> `python eval.py --quantize fp4 --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Llama-2-7b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-7b/fp4`
19. <span id="f19"></span> `python eval.py --quantize nf4 --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Llama-2-7b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-7b/nf4`
20. <span id="f20"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name meta-llama/Llama-2-7b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-7b/any4`

</details>

#### Llama2 13B
|                    | WikiText-2↓ | C4↓   | PTB↓   | CodeParrot↓ | HumanEval↑ | MBPP↑ | MMLU↑  | HellaSwag↑ | GSM8K↑ | BBH↑  |
| ------------------ | ----------- | ----- | ------ | ----------- | ---------- | ----- | ------ | ---------- | ------ | ----- |
| FP16 [[21]](#f21)  | 4.88        | 6.47  | 28.93  | 2.40        | 19.5%      | 18.4% | 50.5%  | 60.0%      | 23.2%  | 47.4% |
| INT4 [[22]](#f22)  | 5.05        | 6.65  | 30.79  | 2.45        | 15.2%      | 16.4% | 48.8%  | 59.3%      | 20.8%  | 44.2% |
| FP4 [[23]](#f23)   | 5.07        | 6.67  | 30.96  | 2.46        | 15.6%      | 16.6% | 49.0%  | 59.7%      | 21.2%  | 44.1% |
| NF4 [[24]](#f24)   | 4.99        | 6.58  | 31.17  | 2.43        | 15.9%      | 16.6% | 49.9%  | 59.9%      | 22.1%  | 44.6% |
| ANY4 [[25]](#f25)  | 4.97        | 6.55  | 28.83  | 2.42        | 15.2%      | 18.0% | 49.3%  | 59.5%      | 21.6%  | 44.6% |

<details open>
<summary>Commands to reproduce results:</summary>

21. <span id="f21"></span> `python eval.py --model-name meta-llama/Llama-2-13b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-13b/fp16`
22. <span id="f22"></span> `python eval.py --quantize intq --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Llama-2-13b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-13b/int4`
23. <span id="f23"></span> `python eval.py --quantize fp4 --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Llama-2-13b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-13b/fp4`
24. <span id="f24"></span> `python eval.py --quantize nf4 --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Llama-2-13b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-13b/nf4`
25. <span id="f25"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name meta-llama/Llama-2-13b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-13b/any4`

</details>

#### Llama2 70B
|                    | WikiText-2↓ | C4↓   | PTB↓   | CodeParrot↓ | HumanEval↑ | MBPP↑ | MMLU↑  | HellaSwag↑ | GSM8K↑ | BBH↑  |
| ------------------ | ----------- | ----- | ------ | ----------- | ---------- | ----- | ------ | ---------- | ------ | ----- |
| FP16 [[26]](#f26)  | 3.32        | 5.52  | 14.44  | 2.11        | 31.7%      | 37.4% | 65.2%  | 64.8%      | 53.3%  | 67.1% |
| INT4 [[27]](#f27)  | 3.46        | 5.61  | 14.61  | 2.14        | 26.8%      | 37.8% | 64.4%  | 64.6%      | 51.4%  | 65.9% |
| FP4 [[28]](#f28)   | 3.53        | 5.67  | 14.34  | 2.16        | 29.0%      | 36.8% | 63.6%  | 63.9%      | 51.2%  | 65.5% |
| NF4 [[29]](#f29)   | 3.44        | 5.61  | 14.36  | 2.13        | 29.9%      | 37.2% | 64.4%  | 63.9%      | 51.9%  | 66.5% |
| ANY4 [[30]](#f30)  | 3.40        | 5.58  | 14.64  | 2.13        | 26.8%      | 38.5% | 64.8%  | 63.6%      | 51.6%  | 66.6% |

<details open>
<summary>Commands to reproduce results:</summary>

26. <span id="f26"></span> `python eval.py --model-name meta-llama/Llama-2-70b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-70b/fp16`
27. <span id="f27"></span> `python eval.py --quantize intq --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Llama-2-70b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-70b/int4`
28. <span id="f28"></span> `python eval.py --quantize fp4 --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Llama-2-70b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-70b/fp4`
29. <span id="f29"></span> `python eval.py --quantize nf4 --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Llama-2-70b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-70b/nf4`
30. <span id="f30"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name meta-llama/Llama-2-70b-hf --tasks wikitext-2 c4 ptb codeparrot humaneval mbpp mmlu hellaswag gsm8k bbh --log-dir ./logs/llama2-70b/any4`

</details>

#### Mistral-7B Instruct v0.2
|                   | WikiText-2↓ | C4↓   | PTB↓   | CodeParrot↓ | MMLU↑  | HellaSwag↑ | GSM8K↑ | BigBench↑ |
| ----------------- | ----------- | ----- | ------ | ----------- | ------ | ---------- | ------ | --------- |
| FP16 [[31]](#f31) | 5.95        | 8.82  | 21.77  | 2.63        | 58.7%  | 66.1%      | 41.7%  | 51.7%     |
| INT4 [[32]](#f32) | 6.14        | 9.03  | 22.02  | 2.70        | 57.1%  | 65.1%      | 39.7%  | 50.4%     |
| FP4 [[33]](#f33)  | 6.19        | 9.10  | 21.62  | 2.70        | 56.6%  | 64.7%      | 38.2%  | 47.7%     |
| NF4 [[34]](#f34)  | 6.06        | 8.93  | 24.72  | 2.66        | 58.0%  | 65.5%      | 38.5%  | 51.8%     |
| ANY4 [[35]](#f35) | 6.00        | 8.85  | 23.24  | 2.64        | 58.6%  | 65.4%      | 41.1%  | 51.7%     |

<details open>
<summary>Commands to reproduce results:</summary>

31. <span id="f31"></span> `python eval.py --model-name mistralai/Mistral-7B-Instruct-v0.2 --tasks wikitext-2 c4 ptb codeparrot mmlu hellaswag gsm8k bigbench --log-dir ./logs/mistral-7b-instruct-v0.2/fp16`
32. <span id="f32"></span> `python eval.py --quantize intq --quantize-args n_bit=4,skip_modules=lm_head --model-name mistralai/Mistral-7B-Instruct-v0.2 --tasks wikitext-2 c4 ptb codeparrot mmlu hellaswag gsm8k bigbench --log-dir ./logs/mistral-7b-instruct-v0.2/int4`
33. <span id="f33"></span> `python eval.py --quantize fp4 --quantize-args n_bit=4,skip_modules=lm_head --model-name mistralai/Mistral-7B-Instruct-v0.2 --tasks wikitext-2 c4 ptb codeparrot mmlu hellaswag gsm8k bigbench --log-dir ./logs/mistral-7b-instruct-v0.2/fp4`
34. <span id="f34"></span> `python eval.py --quantize nf4 --quantize-args n_bit=4,skip_modules=lm_head --model-name mistralai/Mistral-7B-Instruct-v0.2 --tasks wikitext-2 c4 ptb codeparrot mmlu hellaswag gsm8k bigbench --log-dir ./logs/mistral-7b-instruct-v0.2/nf4`
35. <span id="f35"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name mistralai/Mistral-7B-Instruct-v0.2 --tasks wikitext-2 c4 ptb codeparrot mmlu hellaswag gsm8k bigbench --log-dir ./logs/mistral-7b-instruct-v0.2/any4`

</details>

#### Mixtral-8x7B Instruct v0.1
|                   | WikiText-2↓ | C4↓   | PTB↓   | CodeParrot↓ | MMLU↑  | HellaSwag↑ | GSM8K↑ | BigBench↑ |
| ----------------- | ----------- | ----- | ------ | ----------- | ------ | ---------- | ------ | --------- |
| FP16 [[36]](#f36) | 4.14        | 7.18  | 16.47  | 2.20        | 68.2%  | 67.6%      | 64.8%  | 68.1%     |
| INT4 [[37]](#f37) | 4.35        | 7.45  | 16.84  | 2.26        | 66.5%  | 66.3%      | 57.8%  | 61.8%     |
| FP4 [[38]](#f38)  | 4.46        | 7.48  | 18.42  | 2.27        | 66.8%  | 66.5%      | 59.4%  | 62.8%     |
| NF4 [[39]](#f39)  | 4.30        | 7.32  | 15.00  | 2.24        | 67.6%  | 67.2%      | 61.0%  | 66.5%     |
| ANY4 [[40]](#f40) | 4.27        | 7.27  | 16.14  | 2.22        | 67.7%  | 67.1%      | 62.8%  | 65.8%     |

<details open>
<summary>Commands to reproduce results:</summary>

36. <span id="f36"></span> `python eval.py --model-name mistralai/Mixtral-8x7B-Instruct-v0.1 --tasks wikitext-2 c4 ptb codeparrot mmlu hellaswag gsm8k bigbench --log-dir ./logs/mixtral-8x7b-instruct-v0.1/fp16`
37. <span id="f37"></span> `python eval.py --quantize intq --quantize-args n_bit=4,skip_modules=lm_head --model-name mistralai/Mixtral-8x7B-Instruct-v0.1 --tasks wikitext-2 c4 ptb codeparrot mmlu hellaswag gsm8k bigbench --log-dir ./logs/mixtral-8x7b-instruct-v0.1/int4`
38. <span id="f38"></span> `python eval.py --quantize fp4 --quantize-args n_bit=4,skip_modules=lm_head --model-name mistralai/Mixtral-8x7B-Instruct-v0.1 --tasks wikitext-2 c4 ptb codeparrot mmlu hellaswag gsm8k bigbench --log-dir ./logs/mixtral-8x7b-instruct-v0.1/fp4`
39. <span id="f39"></span> `python eval.py --quantize nf4 --quantize-args n_bit=4,skip_modules=lm_head --model-name mistralai/Mixtral-8x7B-Instruct-v0.1 --tasks wikitext-2 c4 ptb codeparrot mmlu hellaswag gsm8k bigbench --log-dir ./logs/mixtral-8x7b-instruct-v0.1/nf4`
40. <span id="f40"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name mistralai/Mixtral-8x7B-Instruct-v0.1 --tasks wikitext-2 c4 ptb codeparrot mmlu hellaswag gsm8k bigbench --log-dir ./logs/mixtral-8x7b-instruct-v0.1/any4`

</details>

### Ablation Studies

#### Group Size
Referencing the paper, *Table 4: C4 perplexity after quantizing with different group sizes.*

|      | 64                 | 128                | 256                | 512                | 1024               |
| ---- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| FP4  | 16.19 [[41]](#f41) | 17.11 [[42]](#f42) | 18.12 [[43]](#f43) | 20.43 [[44]](#f44) | 2.3E6 [[45]](#f45) |
| NF4  | 14.27 [[46]](#f46) | 14.63 [[47]](#f47) | 14.98 [[48]](#f48) | 15.38 [[49]](#f49) | 7.8E5 [[50]](#f50) |
| ANY4 | 13.75 [[51]](#f51) | 13.95 [[52]](#f52) | 14.09 [[53]](#f53) | 14.24 [[54]](#f54) | 14.34 [[55]](#f55) |

<details open>
<summary>Commands to reproduce results:</summary>

41. <span id="f41"></span> `python eval.py --quantize fp4 --quantize-args n_bit=4,group_size=64,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks c4`
42. <span id="f42"></span> `python eval.py --quantize fp4 --quantize-args n_bit=4,group_size=128,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks c4`
43. <span id="f43"></span> `python eval.py --quantize fp4 --quantize-args n_bit=4,group_size=256,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks c4`
44. <span id="f44"></span> `python eval.py --quantize fp4 --quantize-args n_bit=4,group_size=512,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks c4`
45. <span id="f45"></span> `python eval.py --quantize fp4 --quantize-args n_bit=4,group_size=1024,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks c4`
46. <span id="f46"></span> `python eval.py --quantize nf4 --quantize-args n_bit=4,group_size=64,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks c4`
47. <span id="f47"></span> `python eval.py --quantize nf4 --quantize-args n_bit=4,group_size=128,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks c4`
48. <span id="f48"></span> `python eval.py --quantize nf4 --quantize-args n_bit=4,group_size=256,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks c4`
49. <span id="f49"></span> `python eval.py --quantize nf4 --quantize-args n_bit=4,group_size=512,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks c4`
50. <span id="f50"></span> `python eval.py --quantize nf4 --quantize-args n_bit=4,group_size=1024,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks c4`
51. <span id="f51"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,group_size=64,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name meta-llama/Llama-3.2-1B --tasks c4`
52. <span id="f52"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,group_size=128,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name meta-llama/Llama-3.2-1B --tasks c4`
53. <span id="f53"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,group_size=256,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name meta-llama/Llama-3.2-1B --tasks c4`
54. <span id="f54"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,group_size=512,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name meta-llama/Llama-3.2-1B --tasks c4`
55. <span id="f55"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,group_size=1024,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name meta-llama/Llama-3.2-1B --tasks c4`

</details>

#### Calibration Data
Referencing the paper, *Table 3: any4 quantization with different calibration data.*

|                   | Calibration Data    | Number of Samples | Sequence Length per Sample | WikiText-2↓ | C4↓   | PTB↓  | CodeParrot↓ |
| ----------------- | ------------------- | ----------------- | -------------------------- | ----------- | ----- | ----- | ----------- |
| ANY4 [[57]](#f57) | WikiText-2          | 128               | 2048                       | 10.70       | 14.08 | 18.02 | 3.74        |
| ANY4 [[58]](#f58) | Pile                | 128               | 2048                       | 10.70       | 13.99 | 18.26 | 3.74        |
| ANY4 [[59]](#f59) | C4                  | 128               | 2048                       | 10.67       | 14.05 | 17.97 | 3.74        |
| ANY4 [[60]](#f60) | C4                  | 128               | 4096                       | 10.74       | 14.14 | 18.10 | 3.75        |
| ANY4 [[61]](#f61) | C4                  | 128               | 512                        | 10.62       | 13.96 | 18.03 | 3.72        |
| ANY4 [[62]](#f62) | Handwritten Prompt  | 1                 | -                          | 10.63       | 13.95 | 17.94 | 3.71        |

<details open>
<summary>Commands to reproduce results:</summary>

57. <span id="f57"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --calibrate-args dataset=wikitext-2,num_samples=128,max_seq_len=2048 --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot`
58. <span id="f58"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --calibrate-args dataset=pile,num_samples=128,max_seq_len=2048 --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot`
59. <span id="f59"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --calibrate-args dataset=c4,num_samples=128,max_seq_len=2048 --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot`
60. <span id="f60"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --calibrate-args dataset=c4,num_samples=128,max_seq_len=4096 --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot`
61. <span id="f61"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --calibrate-args dataset=c4,num_samples=128,max_seq_len=512 --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot`
62. <span id="f62"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot`

</details>

#### Term to Minimize
Referencing the paper, *Table A3: Perplexity after quantizing Llama3.2 1B with LUTs created by minimizing different terms.*

|                                           | Term to Minimize                      | WikiText-2 | C4    | PTB   | CodeParrot |
| ----------------------------------------- | ------------------------------------- | ---------- | ----- | ----- | ---------- |
| Weights Only [[63]](#f63)                 | $(w_{S_{i,j}} - w_{Q_{i,j}})$         | 6.680      | 9.619 | 11.186| 2.751      |
| Weights × Activations [[64]](#f64)        | $(w_{S_{i,j}}x_j - w_{Q_{i,j}}x_j)$   | 6.496      | 9.375 | 11.055| 2.675      |
| Weights × Activations × Group Scales [[65]](#f65) [Ours] | $(\alpha_{i,j}w_{S_{i,j}}x_j - \alpha_{i,j}w_{Q_{i,j}}x_j)$ | 6.487      | 9.366 | 11.034| 2.680      |

<details open>
<summary>Commands to reproduce results:</summary>

63. <span id="f63"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot`
64. <span id="f64"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot`
65. <span id="f65"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb codeparrot`

</details>

#### K-Means Initialization
Referencing the paper, *Table A4: any4 quantization with K-means clustering initialzied with different algorithms and values.*

|                   | K-Means Initialization | WikiText-2 | C4    | PTB   |
| ----------------- | ---------------------- | ---------- | ----- | ----- |
| FP16 [[66]](#f66) | -                      | 9.76       | 12.77 | 16.56 |
| ANY4 [[67]](#f67) | k-means++              | 10.63      | 13.95 | 17.94 |
| ANY4 [[68]](#f68) | random                 | 10.66      | 13.97 | 18.17 |
| ANY4 [[69]](#f69) | int4                   | 10.83      | 14.21 | 18.69 |
| ANY4 [[70]](#f70) | nf4                    | 10.65      | 13.96 | 18.21 |

<details open>
<summary>Commands to reproduce results:</summary>

66. <span id="f66"></span> `python eval.py --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb`
67. <span id="f67"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True,init=k-means++ --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb`
68. <span id="f68"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True,init=random --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb`
69. <span id="f69"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True,init=int --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb`
70. <span id="f70"></span> `python eval.py --quantize anyq --quantize-args n_bit=4,skip_modules=lm_head,sample_weight=calibrate,scale_sample_weight=True,init=nf4 --model-name meta-llama/Llama-3.2-1B --tasks wikitext-2 c4 ptb`

</details>

# Contribution
We encourage contributions from the community. Please feel free to check our [Issues](https://github.com/facebookresearch/any4/issues) for any task to contribute with, especially our [TODOs](https://github.com/facebookresearch/any4/issues/8) issue, as well as our [Contribiuting Guidelines](CONTRIBUTING.md). 

## License
tinygemm and any4 quantization code are CC-BY-NC 4.0 licensed, as found in the LICENSE file.

## Citation
If you use any4 quantization algorithm and/or tinygemm quantization library, please use the following BibTex entry:
```
@inproceedings{any4,
    title={any4: Learned 4-bit Numeric Representation for {LLM}s},
    author={Mostafa Elhoushi and Jeff Johnson},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=tJmhOPkWCj}
}
```
