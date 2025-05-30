# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils import cpp_extension
import os

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;9.0'

setup(
    name="tinygemm",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="tinygemm",
            sources=[
                "TinyGemm.cpp",
                "TinyGemm_bf16.cu",
                "TinyGemm_int4.cu",
                "TinyGemm_int8.cu",
                "TinyGemmConvertA.cu",
                "TinyGemmConvertB.cu",
                "TinyGemmDequantize.cu",
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']},
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
