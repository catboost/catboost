from setuptools import setup, find_packages
import pybind11.setup_helpers
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
from pyomo.common.fileutils import find_library


original_pybind11_setup_helpers_macos = pybind11.setup_helpers.MACOS
pybind11.setup_helpers.MACOS = False

try:
    highs_lib = find_library('highs', include_PATH=True)
    if highs_lib is None:
        raise RuntimeError('Could not find HiGHS library; Please make sure it is in the LD_LIBRARY_PATH environment variable')
    highs_lib_dir = os.path.dirname(highs_lib)
    highs_build_dir = os.path.dirname(highs_lib_dir)
    highs_include_dir = os.path.join(highs_build_dir, 'include', 'highs')
    if not os.path.exists(os.path.join(highs_include_dir, 'Highs.h')):
        raise RuntimeError('Could not find HiGHS include directory')
    
    extensions = list()
    extensions.append(Pybind11Extension('highspy.highs_bindings',
                                        sources=['highspy/highs_bindings.cpp'],
                                        language='c++',
                                        include_dirs=[highs_include_dir],
                                        library_dirs=[highs_lib_dir],
                                        libraries=['highs']))
    
    setup(name='highspy',
          version='1.1.2.dev1',
          packages=find_packages(),
          description='Python interface to HiGHS',
          maintainer_email='highsopt@gmail.com',
          license='MIT',
          url='https://github.com/ergo-code/highs',
          install_requires=['pybind11', 'numpy', 'pyomo'],
          include_package_data=True,
          package_data={'highspy': ['highspy/*.so']},
          ext_modules=extensions,
          cmdclass={'build_ext': build_ext},
          python_requires='>=3.6',
          classifiers=["Programming Language :: Python :: 3",
                       "Programming Language :: Python :: 3.6",
                       "Programming Language :: Python :: 3.7",
                       "Programming Language :: Python :: 3.8",
                       "Programming Language :: Python :: 3.9",
                       "License :: OSI Approved :: MIT License"]
          )
finally:
    pybind11.setup_helpers.MACOS = original_pybind11_setup_helpers_macos
