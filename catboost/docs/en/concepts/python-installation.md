# {{ python-package }} installation

{% note alert %}

Installation is only supported by the 64-bit version of Python.

{% endnote %}


Dependencies:
- `graphviz` (if you want to use [`plot_tree` function](python-reference_catboost_plot_tree.md))
- `matplotlib`
- `numpy (>=1.16.0)`
- `pandas (>=0.24)`
- `plotly`
- `scipy`
- `six`

{% note info %}

Note that in most cases dependencies will be installed automatically using mechanisms built into `setuptools`, `pip` or `conda`.

{% endnote %}

To install the {{ python-package }}:
1. Choose an installation method:
    - [pip install](../installation/python-installation-method-pip-install.md)
    - [conda install](../installation/python-installation-method-conda-install.md)
{% if audience == "internal" %} - [{#T}](../yandex_specific/python-installation-build-from-arcadia-sources.md) {% endif %}
    - [Build from source](../installation/python-installation-method-build-from-source.md)
    - [Build a wheel package](../installation/python-installation-method-build-a-wheel-package.md)

1. {% include [general-install-data-visualization-support-package](../_includes/work_src/reusage-installation/install-data-visualization-support-package.md) %}

1. (Optionally) [Test {{ product }}](../installation/python-installation-test-catboost.md).

Note that there are additional [system requirements](#gpu-system-requirements) if training on GPU is required.


## GPU system requirements {#gpu-system-requirements}

The versions of {{ product }} for Linux and Windows available from [pip install](../installation/python-installation-method-pip-install.md) and [conda install](../installation/python-installation-method-conda-install.md) have CUDA-enabled GPU support out-of-the-box.

{% include [installation-compute-capability-requirements](../_includes/work_src/reusage-code-examples/compute-capability-requirements.md) %}

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}
