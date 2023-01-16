# Build from source on Linux and macOS

{% if audience == "internal" %}

{% include [internal-note__use-outside-arcadia](../yandex_specific/_includes/note__use-outside-arcadia.md) %}

{% endif %}
## Dependencies and requirements {#dependencies-and-particularities}

{% include [installation-packages-for-installation](../_includes/work_src/reusage/packages-for-installation.md) %}


- `python3`
- `python3-dev`
- `numpy`
- `pandas`

{% include [installation-cuda-toolkit__compatible-system-compilers](../_includes/work_src/reusage-code-examples/cuda-toolkit__compatible-system-compilers.md) %}


{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}


## Building steps  {#building-steps}

To build the {{ python-package }} from source on Linux and macOS:
1. {% if audience == "internal" %} {% include [arcadia_users_step](../yandex_specific/_includes/arcadia_users_step.md) %} {% endif %}Install the `libc` header files on macOS and Linux.

    Depending on the used OS:

    - macOS: `xcode-select --install`
    - Linux: Install the appropriate package (for example, `libc6-dev` on Ubuntu)

1. Clone the repository:

    ```no-highlight
    {{ installation--git-clone }}
    ```

1. (_Optionally_) Volta GPU users are advised to precisely set the required NVCC compile flags in the [default_nvcc_flags.make.inc](https://github.com/catboost/catboost/blob/master/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc) configuration file. Removing irrelevant flags speeds up the compilation.

    {% note info %}

    {{ product }} may work incorrectly with Independent Thread Scheduling introduced in Volta GPUs when the number of splits for features exceeds 32.

    {% endnote %}

1. (_Optionally_) CUDA with compute capability 2.0 users must remove all lines starting with `-gencode` from the [default_nvcc_flags.make.inc](https://github.com/catboost/catboost/blob/master/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc) configuration file and add the following line instead:
    ```no-highlight
    -gencode arch=compute_20,code=compute_20
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
