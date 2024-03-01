# Build environment setup for CMake

{% note warning %}

{% include [ya-make-to-cmake-switch](../_includes/work_src/reusage-installation/ya-make-to-cmake-switch.md) %}

For building {{ product }} using Ya Make see [here](../concepts/build-from-source.md#build-ya-make)

{% endnote %}

{% include [catboost-src-root](../_includes/work_src/reusage-installation/catboost-src-root.md) %}

{% include [cmake-platforms](../_includes/work_src/reusage-installation/cmake-platforms.md) %}

## Native artifacts build requirements

### [Python interpreter](https://www.python.org/)

  Python 3.x interpreter. Python is used in some auxiliary scripts and [`conan` package manager](#conan) uses it.

  For revisions before [98df6bf](https://github.com/catboost/catboost/commit/98df6bf8d4e6ab054b75b727f8d758c3399f4867) python had to have [`six` package](https://pypi.org/project/six/) installed.

### [CMake](https://cmake.org/)

  |Condition|Minimum version|
  |---------|---------|
  | Target OS is Windows | 3.21 |
  | Target OS is Android | 3.21 |
  | CUDA support is enabled | 3.18 |
  | None of the above | 3.15 |

### [Android NDK](https://developer.android.com/ndk/downloads) (only for Android target platform)

### Compilers, linkers and related tools

  Depending on host OS:

  {% list tabs %}

  - Linux

      - [`gcc` compiler](https://gcc.gnu.org/), not used to compile {{ product }} code itself but used to build dependencies as Conan packages.
      - [`clang` compilers](https://clang.llvm.org/), version 14+ and version 12 as a CUDA host compiler if you want to build with CUDA support.
      - [`lld` linker](https://lld.llvm.org/), version 7+

      For Linux target the default CMake toolchain assumes that `clang` and `clang++` are available from the command line and will use them to compile {{ product }} components. If the default version of `clang` and `clang++` is not what is intended to be used for building then modify the toolchain file `$CATBOOST_SRC_ROOT/build/toolchains/clang.toolchain` - replace all occurences of `clang` and `clang++` with `clang-$CLANG_VERSION` and `clang++-$CLANG_VERSION` respectively where `$CLANG_VERSION` is the version of `clang` you want to use like, for example, `16` or `17` (must be already installed).

      For compilation with CUDA support the default CMake toolchain assumes that `clang-12` is available from the command line.

      For revisions before [136f14f](https://github.com/catboost/catboost/commit/136f14f5d55119028a7bb1886814775cd1e2c649) the minimal supported `clang` version has been 12.

      Android target uses its own CMake toolchain and compiler tools specified there are all provided by the NDK.

  - macOS

      - XCode command line tools (must contain `clang` with version 14+, so XCode version must be greater than 14.0 as well)

      For revisions before [136f14f](https://github.com/catboost/catboost/commit/136f14f5d55119028a7bb1886814775cd1e2c649) the minimal supported `clang` version has been 12 (means XCode version must have been 12.0+ as well).

  - Windows

      - Windows 10 or Windows 11 SDK (usually installed as a part of the Microsoft Visual Studio setup)

      - for builds without CUDA support:
        - Microsoft Visual Studio 2022 with `clang-cl` compiler with version 14+ installed (can be selected in `Individual components` pane of the Visual Studio Installer for Visual Studio 2022). See details [here](https://learn.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-170)

      - for builds with CUDA support:
        - MSVC v142 - Visual Studio 2019 C++ x64/x86 build tools version v14.28 - 16.8 or v14.28 - 16.9 (can be selected in `Individual components` pane of the Visual Studio Installer for Visual Studio 2019)

      For revisions before [d5ac776](https://github.com/catboost/catboost/commit/d5ac776e0dd4eeb2ffd99d3fabaaee3e86b8dba1) builds without CUDA have also been using MSVC v142 - Visual Studio 2019 C++ x64/x86 build tools version v14.28 - 16.8 or v14.28 - 16.9.

      For revisions before between [d5ac776](https://github.com/catboost/catboost/commit/d5ac776e0dd4eeb2ffd99d3fabaaee3e86b8dba1) and [136f14f](https://github.com/catboost/catboost/commit/136f14f5d55119028a7bb1886814775cd1e2c649) for builds without CUDA support the minimum supported `clang-cl` version has been 12 (so, Visual Studio 2019 that includes it has also been supported).

  {% endlist %}

### CUDA toolkit (only if CUDA support is needed)

  Supported only for Linux and Windows host and target platforms.

  [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) needs to be installed.

  CUDA version 11.8 is supported by default (because it contains the biggest set of supported target CUDA compute architectures).

  Other CUDA versions (11.4+) can also be used but require changing target compute architecture options in affected CMake targets.

  For revisions before [45cc2e1](https://github.com/catboost/catboost/commit/45cc2e12189e8fef6b0ccfd30ac192efab22ae98) the minimal supported CUDA version has been 11.0 .


### [Conan](https://conan.io/) {#conan}

  Version 1.57.0 - 1.62.0. Version 1.62.0 is required if you use python 3.12. Version 2.x support is [in progress](https://github.com/catboost/catboost/issues/2582).

  Used for some dependencies.

  `conan` command should be available from the command line.

  Make sure that the path to Conan cache does not contain spaces, this causes issues with some projects. Default cache location can be overridden [by specifying `CONAN_USER_HOME` environment variable](https://docs.conan.io/1/mastering/custom_cache.html)

### Build system for CMake

  {% list tabs %}

  - Ninja

      [Ninja](https://ninja-build.org/) is the preferred build system for CMake.

      `ninja` command should be available from the command line.

  - Microsoft Visual Studio solutions

      {% include [cmake-visual-studio-generator](../_includes/work_src/reusage-installation/cmake-visual-studio-generator.md) %}

  - Unix Makefiles

    {% include [cmake-unix-makefiles-generator](../_includes/work_src/reusage-installation/cmake-unix-makefiles-generator.md) %}

  {% endlist %}

### JDK (only for components with JVM API)

  You have to install JDK to build components with JVM API (JVM applier and CatBoost for Apache Spark).
  JDK version has to be 8+ for JVM applier or strictly 8 for CatBoost for Apache Spark.

  Set `JAVA_HOME` environment variable to point to the path of JDK installation to be used during build.

### Python development artifacts (only for Python package)

  You have to install Python development artifacts (Python headers in an include directory and Python library for building modules).

  Note that they are specific to [CPython Python implementation](https://en.wikipedia.org/wiki/CPython). {{ product }} does not currently support other Python implementations like PyPy, Jython or IronPython.

  One convenient way to install different Python versions with development artifacts in one step is to use [pyenv](https://github.com/pyenv/pyenv) (and its variant for Windows - [pyenv-win](https://github.com/pyenv-win/pyenv-win))
