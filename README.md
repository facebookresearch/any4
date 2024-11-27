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

1. Setup Environment
```
conda create --name any4 python=3.10
conda activate any4

pip install -r requirements.txt
```

2. Access Models

Some models (e.g., Llama) require permission. Follow these steps to access them:

a. Submit a request to access a Llama checkpoint, e.g., https://huggingface.co/meta-llama/Llama-3.2-1B.

b. Setup Hugging Face token access by following the steps described [here](https://huggingface.co/docs/hub/en/security-tokens).

c. Then you will be able to login to Hugging Face by running the cell below and entering the token you obtain from Step b. above:
```
huggingface-cli login
```

## Run
- **Example**: a simple text generation with quantization example script that you can try and edit:
```
python example.py
```

# TODOs:
[] Integrate with torchao