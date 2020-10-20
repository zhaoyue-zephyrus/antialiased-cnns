from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='reflection_pad3d_cuda',
    ext_modules=[
        CUDAExtension('reflection_pad3d_cuda', [
            'src/reflection_pad3d_cuda.cpp',
            'src/reflection_pad3d.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
