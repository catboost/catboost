# Build the binary from a local copy on Windows

{% include [installation-cuda-toolkit__compatible-system-compilers](../_includes/work_src/reusage-code-examples/cuda-toolkit__compatible-system-compilers.md) %}


{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}


To build the command-line package from a local copy of the {{ product }} repository on Windows:
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

1. {% if audience == "internal" %} {% include [arcadia_users_step](../yandex_specific/_includes/arcadia_users_step.md) %} {% endif %} Install [Visual Studio Community 2017](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017).

    - Choose the **Windows Platform development** and **Desktop Development with C++** options in the **workloads** tab.
    - Choose a suitable version of the MSVC compiler. The version of MSVC for CUDA Toolkit 9.0 and 9.1 should not be higher than 15.6 v14.11, while the version for CUDA Toolkit 9.2 should not be higher than 15.6 v14.13.

    {% note info %}

    Visual Studio forcibly installs the latest version of the compiler upon each update. The latest MSVC compiler may not be suitable for compiling {{ product }}, especially with CUDA. It is advised to install VC++ 2017 version 15.6 v14.13 for CUDA Toolkit 9.2 and 15.6 v14.11 for CUDA Toolkit 9.0 and 9.1.

    {% cut "Identify the set version of the compiler" %}


    1. Open the properties window for any `cpp` file of the project.
    1. Ensure the absence of the `/nologo` option in the compiler's command-line (for example, by adding the deprecated `/nologo-` option in the **Command Line/Additional Options** box).
    1. Compile this source file (**Ctrl** + **F7**).

    The set version of the compiler is printed to the **Output window**. {{ product }} can not be compiled with 19.14.* versions.

    {% endcut %}

    {% cut "Change the version of the compiler" %}

    Use one of the following methods to set the recommended version of the compiler:
    - Enable the required version as described in the [Visual C++ Team Blog](https://blogs.msdn.microsoft.com/vcblog/2017/11/15/side-by-side-minor-version-msvc-toolsets-in-visual-studio-2017/).

    - Run the environment setter from the command line with the `vcvars_ver` option (the path to the script depends on the installation settings):

    ```bash
    call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.13
    ```

    Then open the solution:
    ```no-highlight
    start msvs\arcadia.sln
    ```
    {% endcut %}

    {% endnote %}

1. Open the `catboost/catboost/app` directory from the local copy of the {{ product }} repository.

1. Run the following command:

    ```no-highlight
    ../../ya make -r [optional parameters]
    ```

    Parameter | Description
    :--- | :---
    `-DCUDA_ROOT` | The path to CUDA. This parameter is required to support training on GPU.
