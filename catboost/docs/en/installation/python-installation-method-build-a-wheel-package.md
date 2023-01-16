# Build a wheel package

{% if audience == "internal" %}

{% include [internal-note__use-outside-arcadia](../yandex_specific/_includes/note__use-outside-arcadia.md) %}

{% endif %}

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}


To build a self-contained Python Wheel:
1. Depending on the OS:
    - Linux and macOS — Perform steps [1](python-installation-method-build-from-source-linux-macos.md#build-linux-macos-step1)–[5](python-installation-method-build-from-source-linux-macos.md#build-linux-macos-step5) of the [Build from source on Linux and macOS](python-installation-method-build-from-source-linux-macos.md) operation.
    - Windows — Perform steps [1](python-installation-method-build-from-source-windows.md#build-windows-step1)–[5](python-installation-method-build-from-source-windows.md#build-windows-step5) of the [Build from source on Windows](python-installation-method-build-from-source-windows.md) operation.

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
