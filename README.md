# any4

## Getting Started
```
conda create --name any4 python=3.10
conda activate any4

pip install -r requirements.txt
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

# Run Example
```
python example.py
```

# TODOs:
[] support arguments
[] try within another codebase
[] support other quantization types
[] plot fp16 and quantized weights / per channel / distribution
[] quantize activations
[] integrate with torch-tune / lm-harness