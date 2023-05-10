# Evaluation library

This is the fastest way to evaluate a model. The library provides a [C API](#c-api) and a simple [C++ wrapper API](#c-plus-plus-wrapper). The C API interface can be accessed from any programming language.

Perform the following steps to build the library:
1. Clone the repository:

    ```
    {{ installation--git-clone }}
    ```

1. Open the `catboost` directory from the local copy of the {{ product }} repository.

1. Choose the preferred way to use the evaluation library and compile it accordingly:

    {% list tabs %}

    - Shared library

        ```bash
        ./ya make -r catboost/libs/model_interface
        ```

        Or (Linux-only):

        ```bash
        export CXX=/path/to/clang++
        export CC=/path/to/clang

        make -f make/model_interface.CLANG50-LINUX-X86_64.makefile
        ```

        The output directory `catboost/libs/model_interface` will contain:

        |OS|Files|
        |--|-----|
        |Linux|`libcatboostmodel.so`|
        |macOS|`libcatboostmodel.dylib`|
        |Windows|`catboostmodel.lib` and `catboostmodel.dll`|

    - Static library (Linux or macOS only)

        ```bash
        ya make -r catboost/libs/model_interface/static
        ```

        Or (Linux-only):

        ```bash
        export CXX=/path/to/clang++
        export CC=/path/to/clang

        make -f make/model_interface_static.CLANG50-LINUX-X86_64.makefile
        ```

        The output directory `catboost/libs/model_interface/static` will contain a pair of artifacts:

        - `liblibcatboostmodel.o`. This part contains symbols that require forced initialization.
        - `libcatboostmodel.a`. This part contains all other symbols.

    {% endlist %}

The {{ product }} model can be loaded from a file or initialized from the buffer memory.


#### Related information

[Source code and a CMake usage example](https://github.com/catboost/catboost/tree/master/catboost/libs/model_interface/cmake_example)

## C API {#c-api}

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

    - Static library (Linux or macOS only)

        Add both `liblibcatboostmodel.o` and  `libcatboostmodel.a` to the linker input.

        On Linux additional libraries `libdl` and `libpthread` have to be added to the linker input as well.

        Example:

        ```
        clang++ <your sources and options> liblibcatboostmodel.o libcatboostmodel.a -ldl -lpthread
        ```

    {% endlist %}

## C++ wrapper API {#c-plus-plus-wrapper}

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
