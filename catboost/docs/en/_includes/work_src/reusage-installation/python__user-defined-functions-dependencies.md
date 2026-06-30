If you want to use [custom metrics or objectives implemented in your own python code](../../../features/custom-loss-functions.md) you should install [`numba`](https://numba.pydata.org/) package to speed up the code execution using JIT compilation.

If you want to use custom metrics or objectives on GPUs with CUDA support you must install [`numba`](https://numba.pydata.org/) package for JIT compilation of CUDA code.
Installation of [`numba-cuda`](https://github.com/NVIDIA/numba-cuda) package is also encouraged.
CUDA itself (not only drivers) must be installed on machines where this code is executed.
See [`numba` CUDA support documentation](https://numba.readthedocs.io/en/stable/cuda/overview.html) for more details.

These packages are not listed in package requirements that are installed automatically because they are not needed for other functionality.
