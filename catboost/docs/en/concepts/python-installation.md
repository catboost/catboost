# {{ python-package }} installation

{% note alert %}

Installation is only supported by the 64-bit version of Python.

{% endnote %}


Dependencies:
- `numpy`
- `six`
- `pandas`

To install the {{ python-package }}:
1. Choose an installation method:
    - [pip install](../installation/python-installation-method-pip-install.md)
    - [conda install](../installation/python-installation-method-conda-install.md)
{% if audience == "internal" %} - [{#T}](../yandex_specific/python-installation-build-from-arcadia-sources.md) {% endif %}
    - [Build from source on Linux and macOS](../installation/python-installation-method-build-from-source-linux-macos.md)
    - [Build from source on Windows](../installation/python-installation-method-build-from-source-windows.md)
    - [Build a wheel package](../installation/python-installation-method-build-a-wheel-package.md)

1. {% include [general-install-data-visualization-support-package](../_includes/work_src/reusage-installation/install-data-visualization-support-package.md) %}

1. (Optionally) [Test {{ product }}](../installation/python-installation-test-catboost.md).

Note that there are additional [system requirements](#gpu-system-requirements) if training on GPU is required.


## GPU system requirements {#gpu-system-requirements}

The versions of {{ product }} available from [pip install](../installation/python-installation-method-pip-install.md) and [conda install](../installation/python-installation-method-conda-install.md) have GPU support out-of-the-box.

{% include [installation-compute-capability-requirements](../_includes/work_src/reusage-code-examples/compute-capability-requirements.md) %}


{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}


The Python version of {{ product }} for CUDA of compute capability 2.0 can be built from source. In this case the following steps are obligatory:
- Step [4](../installation/python-installation-method-build-from-source-linux-macos.md#build-cuda-2) of the [Build from source on Linux and macOS](../installation/python-installation-method-build-from-source-linux-macos.md) operation.
- Step [3](../installation/python-installation-method-build-from-source-windows.md#build-cuda-2) of the [Build from source on Windows](../installation/python-installation-method-build-from-source-windows.md) operation.
