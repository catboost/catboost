# Build environment setup for Ya Make

{% note warning %}

{% include [ya-make-to-cmake-switch](../_includes/work_src/reusage-installation/ya-make-to-cmake-switch.md) %}

For building {{ product }} using CMake see [here](../concepts/build-from-source.md#build-cmake)

{% endnote %}

### Install the `libc` header files (only on macOS and Linux)

{% if audience == "internal" %} {% include [arcadia_users_step](../yandex_specific/_includes/arcadia_users_step.md) %} {% endif %}

  Depending on the used OS:

  - macOS: `xcode-select --install`
  - Linux: Install the appropriate package (for example, `libc6-dev` on Ubuntu)

### Microsoft Visual Studio setup (only for Windows)

{% if audience == "internal" %} {% include [arcadia_users_step](../yandex_specific/_includes/arcadia_users_step.md) %} {% endif %} Install [Visual Studio Community 2019](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2019).

- Choose the **Windows Platform development** and **Desktop Development with C++** options in the **workloads** tab.
- Choose a suitable version of the MSVC compiler. It is advised to install VC++ 2019 version 16.11.11 v14.28 and CUDA Toolkit 11.0 or newer.

{% note info %}

Visual Studio forcibly installs the latest version of the compiler upon each update. The latest MSVC compiler may not be suitable for compiling {{ product }}, especially with CUDA.

{% cut "Identify the set version of the compiler" %}


1. Open the properties window for any `cpp` file of the project.
1. Ensure the absence of the `/nologo` option in the compiler's command-line (for example, by adding the deprecated `/nologo-` option in the **Command Line/Additional Options** box).
1. Compile this source file (**Ctrl** + **F7**).

The set version of the compiler is printed to the **Output window**. {{ product }} can not be compiled with 19.14.* versions.

{% endcut %}

{% cut "Change the version of the compiler" %}

Use one of the following methods to set the recommended version of the compiler:
- Enable the required version as described in the [Visual C++ Team Blog](https://devblogs.microsoft.com/cppblog/side-by-side-minor-version-msvc-toolsets-in-visual-studio-2019/).

- Run the environment setter from the command line with the `vcvars_ver` option (the path to the script depends on the installation settings):

```bash
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.28
```

Then open the solution:
```no-highlight
start msvs\arcadia.sln
```
{% endcut %}

{% endnote %}

### CUDA compatible system compilers (only for builds with CUDA)

{% include [installation-cuda-toolkit__compatible-system-compilers](../_includes/work_src/reusage-code-examples/cuda-toolkit__compatible-system-compilers.md) %}

### NVCC compile flags (only for builds with CUDA)

1. (_Optionally_) Volta GPU users are advised to precisely set the required NVCC compile flags in the [default_nvcc_flags.make.inc](https://github.com/catboost/catboost/blob/master/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc) configuration file. Removing irrelevant flags speeds up the compilation.

    {% note info %}

    {{ product }} may work incorrectly with Independent Thread Scheduling introduced in Volta GPUs when the number of splits for features exceeds 32.

    {% endnote %}

1. (_Optionally_) CUDA with compute capability 2.0 users must remove all lines starting with `-gencode` from the [default_nvcc_flags.make.inc](https://github.com/catboost/catboost/blob/master/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc) configuration file and add the following line instead:
    ```no-highlight
    -gencode arch=compute_20,code=compute_20
    ```
