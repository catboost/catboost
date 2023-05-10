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

        The output directory for the shared library (`libcatboostmodel.<so|dll|dylib>` for Linux/macOS or `libcatboostmodel.dll` for Windows) is `catboost/libs/model_interface`.

    - Static library

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
         `libcatboostmodel.a` and `liblibcatboostmodel.o`.

    {% endlist %}

The {{ product }} model can be loaded from a file or initialized from the buffer memory.


#### Related information

[Source code and a CMake usage example](https://github.com/catboost/catboost/tree/master/catboost/libs/model_interface)

## C API {#c-api}

Perform the following steps to use this API:
1. Link the required library (`libcatboostmodel.<so|dll|dylib>` for Linux/macOS or `libcatboostmodel.dll` for Windows).
1. Use the methods from the `model_calcer_wrapper.h` file (refer to the [doxygen-style documentation](https://github.com/catboost/catboost/blob/master/catboost/libs/model_interface/model_calcer_wrapper.h) for details).

Sample C code without include statements:

```
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
