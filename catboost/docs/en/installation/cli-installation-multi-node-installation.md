# Build the binary with MPI support from a local copy (GPU only)

{% if audience == "internal" %}

{% include [internal-note__use-outside-arcadia](../yandex_specific/_includes/note__use-outside-arcadia.md) %}

{% endif %}

{% note warning %}

This approach will work only for versions prior to [this commit](https://github.com/catboost/catboost/commit/c5c642ca0b8e093336d0229ac4b14c78db3915bb).

{% endnote %}

{{ product }} provides a beta version of multi-node GPU training. Only the <q>feature parallel</q> learning scheme is currently supported. Therefore, only datasets with many features gain the benefit from multi-host multi-GPU support.

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}


To build the command-line package with MPI support from a local copy of the {{ product }} repository:
1. Install an MPI library (for example, Open MPI) on all machines that are supposed to participate in the training.

1. Refer to the MPI library documentation for guides on running MPI jobs on multiple hosts.

1. Add the following environment variables:
    ```no-highlight
    export CPLUS_INCLUDE_PATH=$PATH_TO_MPI/include:$CPLUS_INCLUDE_PATH
    export LIBRARY_PATH=$PATH_TO_MPI/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$PATH_TO_MPI/lib:$LD_LIBRARY_PATH
    ```

1. Clone the repository:

    ```
    {{ installation--git-clone }}
    ```

1. Open the `catboost/catboost/app` directory from the local copy of the {{ product }} repository.

1. Build the command-line version of {{ product }} with MPI support:

    ```
    ya make -r -DUSE_MPI -DCUDA_ROOT=<path to CUDA> [-DWITHOUT_CUDA_AWARE_MPI]
    ```

    `-DWITHOUT_CUDA_AWARE_MPI` is required if the installed MPI library does not support CUDA-aware MPI with multiple GPUs for each MPI process on a host.

{% note info %}

Performance notes:
- {{ product }} requires one working thread per GPU, one thread for routing network messages, and several threads for preprocessing data. Use an appropriate CPU binding policy. It is recommended to bind the application to the full host or PCI root complex used by CUDA devices for optimal performance. See the MPI documentation for more details.
- A fast network connection (like InfiniBand) is required in order to gain the benefit from multi-node support. Slow networks with network capacity of 1Gb/s or less can whittle down the advantages of multi-node training.

{% endnote %}
