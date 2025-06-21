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

### Generation
TBD

### Evaluation
Evaluate a model (with or without quantization) on downstream tasks.
- Baseline fp16 model:
```
python eval.py --model-name facebook/opt-125m --tasks piqa
```
- Quantized int4 model:
```
python eval.py --model-name facebook/opt-125m --quantize intq --tasks piqa
```

Arguments:
- `--tasks`: by default it runs a large number of natural language, coding, and perplexity evaluation tasks:
    - You can specify a space separate list of tasks, e.g., `--tasks piqa mbpp`.
    - You can pass in any task supported by [Eleuther LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness), [BigCode Eval Harness](https://github.com/bigcode-project/bigcode-evaluation-harness), and any Hugging Face dataset to measure its perplexity.

### Analyze
To analyze weights and mean square errors on weights and activations between baseline model and quantized model at each layer:
```
python analyze.py --model-name meta-llama/Llama-3.2-1B --quantize nf4
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
```
python -m pytest .
```

## Experiments
In this section we provide the results in the paper and the command to reproduce each result.


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
