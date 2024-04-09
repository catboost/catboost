# Build native artifacts using CMake

{% note warning %}

{% include [ya-make-to-cmake-switch](../_includes/work_src/reusage-installation/ya-make-to-cmake-switch.md) %}

For building {{ product }} using Ya Make see [here](../concepts/build-from-source.md#build-ya-make)

{% endnote %}

We define **native artifacts** as build system artifacts that contain native code - executable binaries, shared libraries, static libraries, Python and R binary extension modules.

{% include [cmake-platforms](../_includes/work_src/reusage-installation/cmake-platforms.md) %}

{% note info %}

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

{% endnote %}

## Source code

{% include [get-source-code-from-github](../_includes/work_src/reusage-installation/get-source-code-from-github.md) %}

{% include [catboost-src-root](../_includes/work_src/reusage-installation/catboost-src-root.md) %}

## [Dependencies and requirements](build-environment-setup-for-cmake.md)

## Targets {#targets}

CMakeFiles for {{ product }} CMake projects contain different targets that correspond to native artifacts.

Final targets that are important:

|Target|Component|Output location|Description|
|----|---------|------|-----------|
|`catboost` | `app` | `$CMAKE_BINARY_DIR/catboost/app` | CLI app |
| `_catboost` | `python-package` | `$CMAKE_BINARY_DIR/catboost/python-package/catboost` | python package shared library |
| `catboostmodel` | `libs` | `$CMAKE_BINARY_DIR/catboost/libs/model_interface` | C/C++ applier shared library |
| `catboostmodel_static` | `libs` | `$CMAKE_BINARY_DIR/catboost/libs/model_interface/static` | C/C++ applier static library with CatBoost dependencies linked in |
| `catboost_train_interface` | `libs` | `$CMAKE_BINARY_DIR/catboost/libs/train_interface` | C/C++ train API shared library |
| `catboost4j-prediction` | `jvm-packages` | `$CMAKE_BINARY_DIR/catboost/jvm-packages/catboost4j-prediction/src/native_impl` | JVM applier JNI shared library |
| `catboost4j-spark-impl` | `spark` | `$CMAKE_BINARY_DIR/catboost/spark/catboost4j-spark/core/src/native_impl` | Spark native JNI shared library part |
| `catboostr` | `R-package` | `$CMAKE_BINARY_DIR/catboost/R-package/src` | R package shared library |

Supplementary utilities targets (used for testing):

|Target|Output location|Description|
|------|---------------|-----------|
|`limited_precision_dsv_diff` |  `$CMAKE_BINARY_DIR/catboost/tools/limited_precision_dsv_diff` | Utility to compare dsv files that may contain floating point numbers with limited precision |
|`limited_precision_json_diff` |  `$CMAKE_BINARY_DIR/catboost/tools/limited_precision_json_diff` | Utility to compare JSON files that may contain floating point numbers with limited precision |
|`model_comparator` |  `$CMAKE_BINARY_DIR/catboost/tools/model_comparator` | Utility to compare models saved as files |

## Build using `build_native.py` {#build-build-native}

[`build_native.py`](https://github.com/catboost/catboost/blob/master/build/build_native.py) is a convenient wrapper for building native artifacts with a simpler interface compared to [invoking `cmake`, `conan` and `ninja` directly](#build-cmake-conan-ninja).

You can obtain all the options for `build_native.py` with the description by calling it with the `--help` flag:

```
python $CATBOOST_SRC_ROOT/build/build_native.py --help
```

The required options are:
- `--targets` - List of CMake targets to build (,-separated). See [the list of supported targets](#targets)
- `--build-root-dir` - CMake build dir (forwarded to `cmake`'s `-B`  option)

Importantly, `build_native.py` has `--dry-run` and `--verbose` options so you can examine the commands it is going to run without actually running them.

### Examples

- Build `catboost` CLI app for the current platform without CUDA:

  ```
  python $CATBOOST_SRC_ROOT/build/build_native.py --build-root-dir=./build_no_cuda --targets catboost
  ```

- Build `catboost` CLI app for the current platform with CUDA support (path to CUDA is taken from `CUDA_ROOT` or `CUDA_PATH` environment variable, if they are not defined the build will fail):

  ```
  python $CATBOOST_SRC_ROOT/build/build_native.py --build-root-dir=./build_with_cuda --targets catboost --have-cuda
  ```

- Build `catboost` CLI app for the current platform with CUDA support and CUDA path specified explicitly:

  ```
  python $CATBOOST_SRC_ROOT/build/build_native.py --targets catboost --build-root-dir=./build_with_cuda_11 --have-cuda --cuda-root-dir=/usr/local/cuda-11/
  ```

- Build C/C++ applier shared library as a macOS universal binary with macOS minimal version set to 11.0:

  ```
  python $CATBOOST_SRC_ROOT/build/build_native.py --targets catboostmodel --build-root-dir=./build_applier --macos-universal-binaries --macosx-version-min=11.0
  ```

## Build by calling `cmake`, `conan` and `ninja` directly {#build-cmake-conan-ninja}

{% note info %}

For most common scenarios it is easier to run [`build_native.py` descibed above](#build-build-native).

{% endnote %}

### Host platform is the same as the target platform (no cross-compilation)

1. Choose some directory as a build root. Prefer short paths on Windows to avoid hitting the path length limit of 260 characters for files in this directory. This directory is referred to as `$CMAKE_BINARY_DIR` later.

1. If you build on Linux for `aarch64` architecture set special compilation flags (will be used in `conan` packages builds):
    ```
    export CFLAGS="-mno-outline-atomics"
    export CXXFLAGS="-mno-outline-atomics"
    ```
    See [GitHub issue #2527](https://github.com/catboost/catboost/issues/2527) for details.

1. Call `cmake` with `$CATBOOST_SRC_ROOT` as a source tree root and a build root specification: `-B $CMAKE_BINARY_DIR`. See [CMake CLI documentation](https://cmake.org/cmake/help/latest/manual/cmake.1.html) for details. Other important options and definitions for this call are [described below](#cmake-options-and-definitions).

1. Call the build tool (depending on what generator has been specified in the `cmake` call above) with the build specification files generated in `$CMAKE_BINARY_DIR`.
    For `ninja` this will be:

    ```
    ninja -C $CMAKE_BINARY_DIR <target> [<target> ... ]
    ```
    See [the list of possible targets](#targets) (also depends on specified `CATBOOST_COMPONENTS` during the `cmake` call above).
1. Native artifacts are generated in `$CMAKE_BINARY_DIR` subdirectories, see [the list of possible targets](#targets) for an exact output location for each target.

### Host platform is different from the target platform (cross-compilation)

1. Choose some directory as a build root for host platform tools (build as parts of {{ product }}'s CMake project). This directory is referred to as `$CMAKE_NATIVE_TOOLS_BINARY_DIR` later.

1. Call `cmake` with `$CATBOOST_SRC_ROOT` as a source tree root and a build root specification: `-B $CMAKE_NATIVE_TOOLS_BINARY_DIR`. Also pass `-DCATBOOST_COMPONENTS=none` to disable building components of {{ product }} itself - we need only tools here.

    See [CMake CLI documentation](https://cmake.org/cmake/help/latest/manual/cmake.1.html) for details. Other important options and definitions for this call are [described below](#cmake-options-and-definitions).

1. Build host platform tools:
    For `ninja` this will be:

    ```
    ninja -C $CMAKE_NATIVE_TOOLS_BINARY_DIR archiver cpp_styleguide enum_parser flatc protoc rescompiler triecompiler
    ```

1. Choose some directory as a target platform build root. This directory is referred to as `$CMAKE_TARGET_PLATFORM_BINARY_DIR` later.

1. Call `conan` to install host platform tools to `$CMAKE_TARGET_PLATFORM_BINARY_DIR`.

   ```
   conan install -s build_type=<build-type> -if $CMAKE_TARGET_PLATFORM_BINARY_DIR --build=missing $CATBOOST_SRC_ROOT/conanfile.txt
   ```

   where `build-type` is either `Debug` or `Release`.

1. Call `conan` to install target platform libraries to `$CMAKE_TARGET_PLATFORM_BINARY_DIR`.

   ```
   conan install -s build_type=<build-type> -if $CMAKE_TARGET_PLATFORM_BINARY_DIR --build=missing --no-imports -pr:h=<conan_host_profile> -pr:b=default $CATBOOST_SRC_ROOT/conanfile.txt
   ```

   where
     - `build-type` is either `Debug` or `Release`
     - `conan_host_profile` is a path to [a Conan profile](https://docs.conan.io/1/reference/profiles.html) for the target platform.
       {{ product }} provides such profiles for supported target platforms in [$CATBOOST_SRC_ROOT/cmake/conan-profiles](https://github.com/catboost/catboost/tree/master/cmake/conan-profiles)

1. Call `cmake` with `$CATBOOST_SRC_ROOT` as a source tree root and a build root specification: `-B $CMAKE_TARGET_PLATFORM_BINARY_DIR`.

    Compared to a usual `cmake` call for a non cross-platform build you have to:
      - Specify a special toolchain for cross-platform building (not usual `$CATBOOST_SRC_ROOT/build/toolchains/clang.toolchain`). Examples of such toolchains are available in [$CATBOOST_SRC_ROOT/build/toolchains/](https://github.com/catboost/catboost/tree/master/build/toolchains) (with `cross-build` prefix).
      - pass `-DTOOLS_ROOT=$CMAKE_NATIVE_TOOLS_BINARY_DIR` to specify the path to native platform tools to be used during the build (that have been build during the first `cmake` and `ninja` calls above). Note that the path specified here has to be absolute!

    See [CMake CLI documentation](https://cmake.org/cmake/help/latest/manual/cmake.1.html) for details. Other important options and definitions for this call are [described below](#cmake-options-and-definitions).

    See also [CMake cross-compilation documentation](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Cross%20Compiling%20With%20CMake.html).

1. Native artifacts are generated in `$CMAKE_TARGET_PLATFORM_BINARY_DIR` subdirectories, see [the list of possible targets](#targets) for an exact output location for each target.

### CMake - important options and definitions {#cmake-options-and-definitions}

- [`-B <path-to-build>`](https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-B) - path to directory which CMake will use as the root of build directory.

- [`-G <generator-name>`](https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-G) - generator name. The recommended generator is ["Ninja"](https://ninja-build.org/).

    {% include [cmake-visual-studio-generator](../_includes/work_src/reusage-installation/cmake-visual-studio-generator.md) %}

    {% include [cmake-unix-makefiles-generator](../_includes/work_src/reusage-installation/cmake-unix-makefiles-generator.md) %}

- [`-DCMAKE_BUILD_TYPE=<build-type>`](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html) -
build type. Use one of `Debug`, `Release`, `RelWithDebInfo` and `MinSizeRel`.
  Note that multiconfig generators are not properly supported right now. Even if you select Visual Studio as a CMake generator choosing between configurations in the generated solution won't switch all the necessary options, this option will take precedence.

- [`-DCMAKE_TOOLCHAIN_FILE=<path>`](https://cmake.org/cmake/help/latest/variable/CMAKE_TOOLCHAIN_FILE.html) - pass toolchain to CMake. On Linux CMake's default configuration will most likely select `gcc` as a C and C++ compiler, but {{ product }} needs to be built with either `clang` (on Linux or macOS) or Microsoft's `cl` compiler on Windows.
So it is recommended to pass toolchain that will set `clang` and `clang++` as C and C++ compilers on Linux and macOS and also set `clang` as a compiler for host code for CUDA (applicable only on Linux). The default toolchain that does that is [`$CATBOOST_SRC_ROOT/build/toolchains/clang.toolchain`](https://github.com/catboost/catboost/blob/master/build/toolchains/clang.toolchain).

  As {{ product }} requires Clang 12+ to build if the default Clang version available from the command line is less than that then use the modified toolchain where all occurences of `clang` and `clang++` are replaced with `clang-$CLANG_VERSION` and `clang++-$CLANG_VERSION` respectively where `$CLANG_VERSION` is the version of `clang` you want to use like, for example, `12` or `14` (must be already installed).

- [`-DCMAKE_POSITION_INDEPENDENT_CODE=<On|Off>`](https://cmake.org/cmake/help/latest/variable/CMAKE_POSITION_INDEPENDENT_CODE.html) - Turn on or off [Position-independent code](https://en.wikipedia.org/wiki/Position-independent_code) generation. Required for building shared libraries. Off by default.

- `-DCATBOOST_COMPONENTS=<components-list>` - As {{ product }}'s CMake project contains many [different CMake targets for different components](#targets) that have their specific configuration dependencies it is often useful to restrict build configuration if you need only a subset of them (e.g. if you only need to build the CLI app you don't want to set up JDK that is required for building components with JVM API). The list is `;`-delimited so it might require shell escaping or quoting.

  See [Targets](#targets) for components needed to build a certain target. `none` value is also possible and used during cross-compilation when building host platform tools only.

- [`-DCMAKE_OSX_DEPLOYMENT_TARGET=<min_macos_version>`](https://cmake.org/cmake/help/latest/variable/CMAKE_OSX_DEPLOYMENT_TARGET.html) - Specify the minimum version of macOS on which the target binaries are to be deployed. Relevant only for macOS.

- `-DHAVE_CUDA=<yes|no>` - Turn CUDA support on or off. No (off) by default.

- [`-DCUDAToolkit_ROOT=<path>`](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html#search-behavior) - Specify path to CUDA installation directory. Useful if a specific CUDA version has to be selected from several versions installed or if CUDA has been installed to a non-standard path so CMake is unable to find it automatically.

- [`-DCMAKE_CUDA_RUNTIME_LIBRARY=<None|Shared|Static>`](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_RUNTIME_LIBRARY.html) - Select the CUDA runtime library for use when compiling and linking CUDA.

- `-DJAVA_HOME=<path>` - path of JDK installation to be used during build. Relevant only to build components with JVM API (JVM applier and CatBoost for Apache Spark). `JAVA_HOME` environment variable will be used by default.

- [`-DJAVA_AWT_LIBRARY=<path>`](https://cmake.org/cmake/help/latest/module/FindJNI.html#cache-variables) - The path to the Java AWT Native Interface (JAWT) library. Use if specifying `JAVA_HOME` has not been sufficient (if CMake default search logic has been unable to find this library inside JDK).

- [`-DJAVA_JVM_LIBRARY=<path>`](https://cmake.org/cmake/help/latest/module/FindJNI.html#cache-variables) - The path to the Java Virtual Machine (JVM) library. Use if specifying `JAVA_HOME` has not been sufficient (if CMake default search logic has been unable to find this library inside JDK).

- [`-DPython3_ROOT_DIR=<path>`](https://cmake.org/cmake/help/latest/module/FindPython3.html#hints) - path to the Python installation to build extension for (not necessarily the same as the current Python interpreter used by default in command line). Must contain Python development artifacts (Python headers in an include directory and Python library for building modules)

- [`-DPython3_LIBRARY=<path>`](https://cmake.org/cmake/help/latest/module/FindPython3.html#artifacts-specification) - The path to the Python library for building modules. Use if specifying `Python3_ROOT_DIR` has not been sufficient (if CMake default search logic has been unable to find this library inside `Python3_ROOT_DIR`)

- [`-DPython3_INCLUDE_DIR=<path>`](https://cmake.org/cmake/help/latest/module/FindPython3.html#artifacts-specification) - The path to the directory of the Python headers. Use if specifying `Python3_ROOT_DIR` has not been sufficient (if CMake default search logic has been unable to find this library inside `Python3_ROOT_DIR`)

- [`-DCMAKE_FIND_ROOT_PATH=<path>[;<path>]`](https://cmake.org/cmake/help/latest/variable/CMAKE_FIND_ROOT_PATH.html) - Semicolon-separated list of root paths to search on the filesystem. This variable is most useful when cross-compiling.
