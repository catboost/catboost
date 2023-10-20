# Evaluation library

This is the fastest way to evaluate a model. The library provides a [C API](#c-api) and a simple [C++ wrapper API](#c-plus-plus-wrapper). The C API interface can be accessed from any programming language.

## Download

Prebuilt shared library artifacts are available.

{% include [reusage-cli-releases-page](../_includes/work_src/reusage-cli/releases-page.md) %}

|Operating system|CPU architectures|GPU support using [CUDA](https://developer.nvidia.com/cuda-zone)|
|--------|-----------------|------------|
| Linux (compatible with [manylinux2014 platform tag](https://peps.python.org/pep-0599/) ) | x86_64 and aarch64 |yes|
| macOS (versions currently supported by Apple) | x86_64 and arm64 |no|
| Windows 10 and 11 | x86_64 |yes|

|Operating system|Files|
|--|-----|
|Linux|`libcatboostmodel-linux-{cpu_arch}-{release_version}.so`|
|macOS|`libcatboostmodel-darwin-universal2-{release_version}.dylib`|
|Windows|`catboostmodel-windows-{cpu_arch}-{release_version}.lib` and `catboostmodel-windows-{cpu_arch}-{release_version}.dll`|

Applying on GPU requires NVIDIA Driver of version 418.xx or higher.

{% note info %}

Only CUDA {{ cuda_version__compiled-packages }} is officially supported in compiled binaries for Windows. [Вuild the binary from a local copy](#build) if GPU support is required and the installed version of CUDA differs from {{ cuda_version__compiled-packages }}. {{ product }} should work fine with CUDA {{ cuda_version__compiled-packages }} and later versions.

All necessary CUDA libraries are statically linked to the Linux and macOS binaries of the {{ product }} applier library, therefore, the only installation necessary is the appropriate version of the CUDA driver.

{% endnote %}

{% note info %}

Release binaries for x86_64 CPU architectures are built with SIMD extensions SSE2, SSE3, SSSE3, SSE4 enabled. If you need to run {{ product }} on older CPUs that do not support these instruction sets [build library binary artifacts yourself](#build)

{% endnote %}


## Build from source {#build}

{% note warning %}

{% include [ya-make-to-cmake-switch](../_includes/work_src/reusage-installation/ya-make-to-cmake-switch.md) %}

Select the appropriate build method below accordingly.

{% endnote %}

### Source code

{% include [get-source-code-from-github](../_includes/work_src/reusage-installation/get-source-code-from-github.md) %}

### Build using CMake  {#cmake}

{% list tabs %}

- Shared library

    Build `catboostmodel` target.

    See [Build native artifacts](../installation/build-native-artifacts.md).

    Built artifacts will be in `$CMAKE_BINARY_DIR/catboost/libs/model_interface`:

    {% include [build-model-interface-shared-artifacts](../_includes/work_src/reusage-installation/build-model-interface-shared-artifacts.md) %}

- Static library

    Build `catboostmodel_static` target.

    See [Build native artifacts](../installation/build-native-artifacts.md).

    Built library will consist of two parts:
        - `global` part. This part contains symbols that require forced initialization.
        - non-`global` part. All other symbols.

    Built artifacts will be in `$CMAKE_BINARY_DIR/catboost/libs/model_interface/static`:

    |OS|Files|
    |--|-----|
    |Linux or macOS|`libcatboostmodel_static.a`, `libcatboostmodel_static.global.a`|
    |Windows|`catboostmodel_static.lib`, `catboostmodel_static.global.lib`|

{% endlist %}

### Build using Ya Make  {#ya-make}

1. [Setup build environment](../installation/build-environment-setup-for-ya-make.md)

1. Open the `catboost` directory from the local copy of the {{ product }} repository.

1. Run the following command:

    {% list tabs %}

    - Shared library

        ```
        ./ya make -r [optional parameters] catboost/libs/model_interface
        ```

        The output directory `catboost/libs/model_interface` will contain:

        {% include [build-model-interface-shared-artifacts](../_includes/work_src/reusage-installation/build-model-interface-shared-artifacts.md) %}

    - Static library (only for Linux and macOS)

        ```
        ./ya make -r [optional parameters] catboost/libs/model_interface/static
        ```

        The output directory `catboost/libs/model_interface/static` will contain a pair of artifacts:

        {% include [build-model-interface-static-old-artifacts](../_includes/work_src/reusage-installation/build-model-interface-static-old-artifacts.md) %}

    {% endlist %}

    Useful parameters:

    Parameter | Description
    ----- | -----
    `-DCUDA_ROOT` | The path to CUDA. This parameter is required to support training on GPU.
    `-DHAVE_CUDA=no` | Disable CUDA support. This speeds up compilation.<br/><br/>By default, the package is built with CUDA support if CUDA Toolkit is installed.

### Build using Make (Linux-only)  {#make}

{% note warning %}

This approach will work only for versions prior to [this commit](https://github.com/catboost/catboost/commit/c5c642ca0b8e093336d0229ac4b14c78db3915bb).

For newer versions use [Build with CMake](#cmake)

{% endnote %}

Choose the preferred way to use the evaluation library and compile it accordingly:

{% list tabs %}

- Shared library

    ```bash
    export CXX=/path/to/clang++
    export CC=/path/to/clang

    make -f make/model_interface.CLANG50-LINUX-X86_64.makefile
    ```

    The output directory `catboost/libs/model_interface` will contain `libcatboostmodel.so`.

- Static library

    ```bash
    export CXX=/path/to/clang++
    export CC=/path/to/clang

    make -f make/model_interface_static.CLANG50-LINUX-X86_64.makefile
    ```

    The output directory `catboost/libs/model_interface/static` will contain a pair of artifacts:

    {% include [build-model-interface-static-old-artifacts](../_includes/work_src/reusage-installation/build-model-interface-static-old-artifacts.md) %}

{% endlist %}

## Usage

The {{ product }} model can be loaded from a file or initialized from the buffer memory.

### C API {#c-api}

Perform the following steps to use this API:

1. Use the methods from the `c_api.h` file (refer to the [doxygen-style documentation](https://github.com/catboost/catboost/blob/master/catboost/libs/model_interface/c_api.h) for details).

    Sample C code without include statements:

    ```c
    float floatFeatures[100];
    char* catFeatures[2] = {"1", "2"};
    double result[1];
    ModelCalcerHandle modelHandle;
    modelHandle = ModelCalcerCreate();
    if (!LoadFullModelFromFile(modelHandle, "model.cbm")) {
        printf("LoadFullModelFromFile error message: %s\n", GetErrorString());
    }
    if (!CalcModelPrediction(
            modelHandle,
            1,
            &floatFeatures, 100,
            &catFeatures, 2,
            &result, 1
        )) {
        printf("CalcModelPrediction error message: %s\n", GetErrorString());
    }
    ModelCalcerDelete(modelHandle);
    ```

1. Add the required libraries to the linking command.

    Linker is often invoked through the compiler call, examples below assume that.

    {% list tabs %}

    - Shared library

        - Linux or macOS

            Example:

            ```
            clang++ <your sources and options> -L<path_to_dir_with_libcatboostmodel> -lcatboostmodel
            ```

        - Windows

            Example:

            ```
            cl.exe <your sources and options> /link <path_to_dir_with_libcatboostmodel>\catboostmodel.lib
            ```

        The shared library must be accessible from the dynamic library loader search path. See your operating system documentation for the details.

    - Static library built using CMake

        Add both `global` and non-`global` parts to the linker input. `global` part requires passing special platform-specific flags to force the required initialization of symbols.

        See per-platform examples below:

        - Linux

            On Linux additional libraries `libdl` and `libpthread` have to be added to the linker input as well.

            ```
            clang++ <your sources and options> -nodefaultlibs -lpthread -ldl -Wl,--whole-archive <catboost_lib_dir>/libcatboostmodel_static.global.a -Wl,--no-whole-archive <catboost_lib_dir>/libcatboostmodel_static.a
            ```

        - macOS

            ```
            clang++ <your sources and options> <catboost_lib_dir>/libcatboostmodel_static.a
            -Wl,-force_load,<catboost_lib_dir>/libcatboostmodel_static.global.a
            ```

       - Windows

            When using `c_api.h` with the static library the additional define `CATBOOST_API_STATIC_LIB` is required.

            ```
            cl.exe <your sources and options> /DCATBOOST_API_STATIC_LIB /link /WHOLEARCHIVE:<catboost_lib_dir>\catboostmodel_static.global.lib <catboost_lib_dir>\catboostmodel_static.lib
            ```

    - Static library built using Ya Make or Make (Linux or macOS only)

        Add both `liblibcatboostmodel.o` and  `libcatboostmodel.a` to the linker input.

        On Linux additional libraries `libdl` and `libpthread` have to be added to the linker input as well.

        Example:

        ```
        clang++ <your sources and options> liblibcatboostmodel.o libcatboostmodel.a -ldl -lpthread
        ```

    {% endlist %}

### C++ wrapper API {#c-plus-plus-wrapper}

A C++ wrapper for the C API interface is also available.

Refer to the [wrapped_calcer.h](https://github.com/catboost/catboost/blob/master/catboost/libs/model_interface/wrapped_calcer.h) file and the sample [CMake project](https://github.com/catboost/catboost/blob/master/catboost/libs/model_interface/cmake_example/CMakeLists.txt) in the {{ product }} repository for more details.

Usage example:
```cpp
ModelCalcerWrapper calcer("model.cbm");
std::vector<float> floatFeatures(100);
std::vector<std::string> catFeatures = {"one", "two", "three"};
std::cout << calcer.Calc(floatFeatures, catFeatures) << std::endl;
```

`ModelCalcerWrapper` also has a constructor to read data from the memory buffer.

### Related information

[Source code and a CMake usage example](https://github.com/catboost/catboost/tree/master/catboost/libs/model_interface/cmake_example)
