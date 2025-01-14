from setuptools import Extension, setup
from torch.utils import cpp_extension
import os

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;9.0'

setup(
    name="tinygemm",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "tinygemm",
            ["TinyGemm.cpp", "TinyGemm_bf16.cu", "TinyGemm_int4.cu", "TinyGemm_int8.cu", "TinyGemmConvertA.cu", "TinyGemmConvertB.cu", "TinyGemmDequantize.cu"],
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)