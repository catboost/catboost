# Build Python package from source on Linux and macOS using Ya Make

{% if audience == "internal" %}

{% include [internal-note__use-outside-arcadia](../yandex_specific/_includes/note__use-outside-arcadia.md) %}

{% endif %}

{% note info %}

{% include [ya-make-to-cmake-switch](../_includes/work_src/reusage-installation/ya-make-to-cmake-switch.md) %}

This page describes building using Ya Make. For building using CMake see [documentation here](python-installation-method-build-from-source.md).

{% endnote %}

## Dependencies and requirements {#dependencies-and-particularities}

1. [Setup environment for ya make build](build-environment-setup-for-ya-make.md)

2. {% include [installation-packages-for-installation](../_includes/work_src/reusage/packages-for-installation.md) %}


- `python3`
- `python3-dev`
- `numpy`
- `pandas`

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}


## Building steps  {#building-steps}

To build the {{ python-package }} from source on Linux and macOS:

1. Clone the repository:

    ```no-highlight
    {{ installation--git-clone }}
    ```

1. Open the `catboost/catboost/python-package/catboost` directory from the local copy of the {{ product }} repository.

1. {% include [installation-compile-the-library](../_includes/work_src/reusage-code-examples/compile-the-library.md) %}

    {% include [installation-installation-example](../_includes/work_src/reusage-code-examples/installation-example.md) %}

    {% note info %}

    The required version of Xcode for building on macOS is specified on the NVIDIA site when downloading the CUDA toolkit.

    {% endnote %}

1. Add the current directory to `PYTHONPATH`:
    ```
    cd ../; export PYTHONPATH=$PYTHONPATH:$(pwd)
    ```

## Troubleshooting {#troubleshooting}

Error type | Message format | Troubleshooting tips
---------- | -------------- | --------------------
`AttributeError` | ```type object '_catboost.<something>' has no attribute 'reduce_cython'```<br/><b>Example:</b><br/>```AttributeError: type object '_catboost._FloatArrayWrapper' has no attribute 'reduce_cython' ```<br/>```AttributeError: type object '_catboost.Py_ITypedSequencePtr' has no attribute 'reduce_cython' ```<br/><br/>| Install or update the following packages:<br/>- `numpy`<br/>- `pandas`
