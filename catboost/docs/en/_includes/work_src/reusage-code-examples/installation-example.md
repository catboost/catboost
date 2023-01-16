
For example, the following command builds the package for Python 3 with training on GPU support:
```no-highlight
../../../ya make -r -DUSE_ARCADIA_PYTHON=no -DOS_SDK=local -DPYTHON_CONFIG=python3-config -DCUDA_ROOT=/usr/local/cuda
```
