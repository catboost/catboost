# Build the binary from a local copy on Windows

{% include [installation-cuda-toolkit__compatible-system-compilers](../_includes/work_src/reusage-code-examples/cuda-toolkit__compatible-system-compilers.md) %}


{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}


To build the command-line package from a local copy of the {{ product }} repository on Windows:


{% include [windows-build-setup](../_includes/work_src/reusage-installation/windows-build-setup.md) %}


1. Open the `catboost/catboost/app` directory from the local copy of the {{ product }} repository.

1. Run the following command:

    ```no-highlight
    ../../ya make -r [optional parameters]
    ```

    Parameter | Description
    :--- | :---
    `-DCUDA_ROOT` | The path to CUDA. This parameter is required to support training on GPU.
