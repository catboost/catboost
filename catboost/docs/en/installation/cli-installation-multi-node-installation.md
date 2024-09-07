# Build the binary with MPI support from a local source repository (GPU only)

{% if audience == "internal" %}

{% include [internal-note__use-outside-arcadia](../yandex_specific/_includes/note__use-outside-arcadia.md) %}

{% endif %}

{{ product }} provides a beta version of multi-node GPU training. Only the <q>feature parallel</q> learning scheme is currently supported. Therefore, only datasets with many features gain the benefit from multi-host multi-GPU support.

## Build

To build the command-line package with MPI support from a local copy of the {{ product }} repository:

{% note warning %}

{% include [ya-make-to-cmake-switch](../_includes/work_src/reusage-installation/ya-make-to-cmake-switch.md) %}

Select the appropriate build method below accordingly.

{% endnote %}

### Requirements

* Common build environment setup for [Ya Make](build-environment-setup-for-ya-make.md) or [CMake](build-environment-setup-for-cmake.md). 
* Some MPI implementation library (for example, `OpenMPI`) installation is required.

### Source code

{% include [get-source-code-from-github](../_includes/work_src/reusage-installation/get-source-code-from-github.md) %}

### Build using CMake  {#cmake}

Build `catboost` target with CUDA support enabled and `-DUSE_MPI=1` flag passed to `cmake` (possibly through `build_native.py`). See [Build native artifacts](build-native-artifacts.md).

Another flag can be also used:

  `-DWITHOUT_CUDA_AWARE_MPI=1` is required if the installed MPI library does not support CUDA-aware MPI with multiple GPUs for each MPI process on a host.

### Build using Ya Make  {#ya-make}

1. Add the following environment variables:
    ```no-highlight
    export CPLUS_INCLUDE_PATH=$PATH_TO_MPI/include:$CPLUS_INCLUDE_PATH
    export LIBRARY_PATH=$PATH_TO_MPI/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$PATH_TO_MPI/lib:$LD_LIBRARY_PATH
    ```

1. Open the `catboost/catboost/app` directory from the local copy of the {{ product }} repository.

1. Build the command-line version of {{ product }} with MPI support:

    ```
    ya make -r -DUSE_MPI -DCUDA_ROOT=<path to CUDA> [-DWITHOUT_CUDA_AWARE_MPI]
    ```

    `-DWITHOUT_CUDA_AWARE_MPI` is required if the installed MPI library does not support CUDA-aware MPI with multiple GPUs for each MPI process on a host.

## Usage

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

Refer to the MPI library documentation for guides on running MPI jobs on multiple hosts.

{% note info %}

Performance notes:
- {{ product }} requires one working thread per GPU, one thread for routing network messages, and several threads for preprocessing data. Use an appropriate CPU binding policy. It is recommended to bind the application to the full host or PCI root complex used by CUDA devices for optimal performance. See the MPI documentation for more details.
- A fast network connection (like InfiniBand) is required in order to gain the benefit from multi-node support. Slow networks with network capacity of 1Gb/s or less can whittle down the advantages of multi-node training.

{% endnote %}
