# Build the binary with `make` on Linux (CPU only)

{% note warning %}

This approach will work only for versions prior to [this commit](https://github.com/catboost/catboost/commit/c5c642ca0b8e093336d0229ac4b14c78db3915bb).

For newer versions use [Build with CMake](cli-installation-local-copy-installation.md#cmake)

{% endnote %}

{% if audience == "internal" %}

{% include [internal-note__use-outside-arcadia](../yandex_specific/_includes/note__use-outside-arcadia.md) %}

{% endif %}

{% note info %}

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

{% endnote %}


To build a CPU version of the command-line package using `make`:

1. Clone the repository:

    ```
    {{ installation--git-clone }}
    ```

1. Install the appropriate package (for example, `libc6-dev` on Ubuntu).

1. Install Clang 3.9.
1. Set the CC and CXX variables to point to the installed Clang 3.9, for example:

    ```
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    ```

1. Run the following command:

    ```
    make -f make/CLANG39-LINUX-X86_64.makefile
    ```


The output directory for the binary is `/catboost/catboost/app`.
