# any4

## Getting Started

0. Setup Proxy
If you are running on DevServer or DevGPU, either run these commands in your terminal, or add them to your `~/.bashrc`:
```
alias proxycurl='curl -x fwdproxy:8080'
alias with-proxy='HTTPS_PROXY=http://fwdproxy:8080 HTTP_PROXY=http://fwdproxy:8080 FTP_PROXY=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 ftp_proxy=http://fwdproxy:8080 http_no_proxy='\''\'\'\''*.facebook.com|*.tfbnw.net|*.fb.com'\''\'\'
```

Then either:
- run at the beginning of your terminal session:
```
with-proxy
```
- add `with-proxy` before any command that may access the web: `git`, `conda`, `pip`, or `python` script that attempts to download a model

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
cd tinygemm
python setup.py install
cd ..
```

## Run
Most of the scripts below will run baseline fp16 model by default. To quantize add the following arguments:
- `--model-args`: pass in any args that are passed to Hugging Face's [`from_pretrained()`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained) function, including `load_in_4bit` and `load_in_8bit`.
- `--quantize`: implements different (fake) quantization algorithms implemented in this codebase. It can take: `intq` (integer quantization), `fp4` (4-bit float quantization), `nf4` (4-bit normal float quantization), `anyq` (proposed lookup table quantization).
    - `--quantize-args`: comma-separated arguments to pass to a quantization algorithm, e.g., `--quantize-args n_bit=4,group_size=32` will perform 4-bit quantization with group size 32.
- `--bnb-args`: comma-separated arguments to pass to [`BitsAndBytesConfig`](https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/quantization#transformers.BitsAndBytesConfig), e.g., `load_in_4bit=True,bnb_4bit_compute_dtype=fp32`
- `--torchao-args`: TBD

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

# TODOs:
- Add Notebook
- Integrate with torchao

## License
tinygemm and any4 quantization code are CC-BY-NC 4.0 licensed, as found in the LICENSE file.
