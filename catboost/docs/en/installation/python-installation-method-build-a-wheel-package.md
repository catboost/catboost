# Build a wheel package

{% if audience == "internal" %}

{% include [internal-note__use-outside-arcadia](../yandex_specific/_includes/note__use-outside-arcadia.md) %}

{% endif %}

## Build using `setup.py bdist_wheel` or `build` {#standard}

Since [this commit](https://github.com/catboost/catboost/commit/f37d091421089231ed3c74a0431fff1c3544d606) CatBoost uses standard Python procedures.

See [documentation here](python-installation-method-build-from-source.md#build-wheel).

## Build using mk_wheel.py {#mk-wheel}

Recommended only for versions previous to [this commit](https://github.com/catboost/catboost/commit/f37d091421089231ed3c74a0431fff1c3544d606). For newer versions [build using `setup.py bdist_wheel` or `build`](#standard).

To build a self-contained Python Wheel:

1. [Setup build environment](build-environment-setup-for-ya-make.md)
1. Run the `catboost/catboost/python-package/mk_wheel.py` script.

Optional parameters:

Parameter | Description
----- | -----
`-DCUDA_ROOT` | The path to CUDA. This parameter is required to support training on GPU.

For example, to build and install a wheel on Windows for Anaconda with training on GPU support run:

```
python.exe mk_wheel.py -DPYTHON_INCLUDE="/I C:\Anaconda2\include" -DPYTHON_LIBRARIES="C:\Anaconda2\libs\python27.lib" -DCUDA_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0"
C:\Anaconda2\Scripts\pip.exe install catboost-0.1.0.6-cp27-none-win_amd64.whl
```
