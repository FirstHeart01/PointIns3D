from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    setup(
        name='PointIns3D',
        version='1.0',
        author='Zeng Qi',
        packages=['pointins3d'],
        package_data={'pointins3d.ops': ['*/*.so']},
        ext_modules=[
            CUDAExtension(
                name='pointins3d.ops.ops',
                sources=[
                    'pointins3d/ops/src/pointins3d_api.cpp', 'pointins3d/ops/src/pointins3d_ops.cpp',
                    'pointins3d/ops/src/cuda.cu'
                ],
                extra_compile_args={
                    'cxx': ['-g'],
                    'nvcc': ['-O2']
                })
        ],
        cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)})
