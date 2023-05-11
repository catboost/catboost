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
