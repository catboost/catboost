# Build the binary from a local copy on Linux and macOS

{% include [installation-cuda-toolkit__compatible-system-compilers](../_includes/work_src/reusage-code-examples/cuda-toolkit__compatible-system-compilers.md) %}


{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}


To build the command-line package from a local copy of the {{ product }} repository on Linux and macOS:
1. {% if audience == "internal" %} {% include [arcadia_users_step](../yandex_specific/_includes/arcadia_users_step.md) %} {% endif %} Install the `libc` header files on macOS and Linux.

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

1. Open the `catboost/catboost/app` directory from the local copy of the {{ product }} repository.

1. Run the following command:

    ```
    ../../ya make -r [optional parameters]
    ```

    Parameter | Description
    ----- | -----
    `-DCUDA_ROOT` | The path to CUDA. This parameter is required to support training on GPU.
    `-DHAVE_CUDA=no` | Disable CUDA support. This speeds up compilation.<br/><br/>By default, the package is built with CUDA support if CUDA Toolkit is installed.
    `-o` | The directory to output the compiled package to. By default, the current directory is used.
