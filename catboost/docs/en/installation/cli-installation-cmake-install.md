# Build the binary with `CMake`

{% if audience == "internal" %}

{% include [internal-note__use-outside-arcadia](../yandex_specific/_includes/note__use-outside-arcadia.md) %}

{% endif %}

{% note info %}

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

{% endnote %}

## Dependencies and requirements {#dependencies-and-particularities}

Make sure you have the following installed:

  1. [CMake](https://cmake.org/), version 3.15 or higher
  1. [Python](https://www.python.org/) with development headers installed, version 3.6 or higher
  1. For Linux and Windows - [CUDA](https://docs.nvidia.com/cuda/index.html), version 11.0 or higher
  1. [Conan](https://conan.io/), most recent versions are preferable

## Building steps  {#building-steps}  

1. Clone the repository:

    ```
    {{ installation--git-clone }}
    ```

1. Go to the root directory of the {{ product }} repository local copy.

1. Run CMake with the appropriate build type and specifying Python3_ROOT_DIR, e.g.:

   ```
   cmake ./ -DCMAKE_BUILD_TYPE=Release -DPython3_ROOT_DIR=/Library/Frameworks/Python.framework/Versions/3.10/
   ```
   
1. Build the binary from the generated makefiles:

   ```
   cmake --build catboost
   ```


The output directory for the binary is `./catboost/catboost/app`.
