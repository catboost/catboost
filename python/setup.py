# python/setup.py — CMake extension build for nanobind bindings.
#
# Uses mlx.extension.CMakeExtension and CMakeBuild to compile the
# _core nanobind module. Falls back gracefully if MLX or cmake
# are not available (pure-Python package without native bindings).

import os
import sys

from setuptools import setup

# Try to import MLX extension helpers for native build.
# If unavailable (e.g., no MLX installed), fall back to pure-Python.
ext_modules = []
cmdclass = {}

try:
    from mlx.extension import CMakeExtension, CMakeBuild

    ext_modules = [
        CMakeExtension(
            name="catboost_mlx._core",
            sourcedir="catboost_mlx/_core",
        )
    ]
    cmdclass = {"build_ext": CMakeBuild}
except ImportError:
    print(
        "WARNING: mlx.extension not available — building without native bindings.\n"
        "Install mlx>=0.18 and cmake>=3.27 for in-process GPU training.",
        file=sys.stderr,
    )

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
