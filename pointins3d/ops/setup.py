from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='PI_OP',
    ext_modules=[
        CUDAExtension(
            'PI_OP', ['src/pointins3d_api.cpp', 'src/pointins3d_ops.cpp', 'src/cuda.cu'],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']
            })
    ],
    cmdclass={'build_ext': BuildExtension})
