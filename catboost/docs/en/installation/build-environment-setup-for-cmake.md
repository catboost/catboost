# Build environment setup for CMake

{% note warning %}

{% include [ya-make-to-cmake-switch](../_includes/work_src/reusage-installation/ya-make-to-cmake-switch.md) %}

For building {{ product }} using Ya Make see [here](../concepts/build-from-source.md#build-ya-make)

{% endnote %}

{% include [catboost-src-root](../_includes/work_src/reusage-installation/catboost-src-root.md) %}

{% include [cmake-platforms](../_includes/work_src/reusage-installation/cmake-platforms.md) %}

## Native artifacts build requirements

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
      - [`clang` compiler](https://clang.llvm.org/), version 12+
      - [`lld` linker](https://lld.llvm.org/), version 7+

      For Linux target the default CMake toolchain assumes that `clang` and `clang++` are available from the command line and will use them to compile {{ product }} components. If the default version of `clang` and `clang++` is not what is intended to be used for building then modify the toolchain file `$CATBOOST_SRC_ROOT/build/toolchains/clang.toolchain` - replace all occurences of `clang` and `clang++` with `clang-$CLANG_VERSION` and `clang++-$CLANG_VERSION` respectively where `$CLANG_VERSION` is the version of `clang` you want to use like, for example, `12` or `14` (must be already installed).

      Android target uses its own CMake toolchain and compiler tools specified there are all provided by the NDK.

  - macOS

      - XCode command line tools (must contain `clang` with version 12+, so XCode version must be greater than 12.0 as well)

  - Windows

      - Windows 10 or Windows 11 SDK (usually installed as a part of the Microsoft Visual Studio setup)
      - MSVC v142 - VS 2019 C++ x64/x86 build tools version v14.28 - 16.8 or v14.28 - 16.9 (can be selected in `Individual components` pane of the Visual Studio Installer for Visual Studio 2019)

  {% endlist %}

### CUDA toolkit (only if CUDA support is needed)

  Supported only for Linux and Windows host and target platforms.

  [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) needs to be installed.

  CUDA version 11.8 is supported by default (because it contains the biggest set of supported target CUDA compute architectures).

  Other CUDA versions (11.0+) can also be used but require changing target compute architecture options in affected CMake targets.

### [Conan](https://conan.io/)

  Version 1.57.0 - 1.59.0. Version 2.x support is in progress.

  Used for some dependencies.

  `conan` command should be available from the command line.

  Make sure that the path to Conan cache does not contain spaces, this causes issues with some projects. Default cache location can be overridden [by specifying `CONAN_USER_HOME` environment variable](https://docs.conan.io/1/mastering/custom_cache.html)

### [Ninja](https://ninja-build.org/)

  Used as a build system for CMake.

  `ninja` command should be available from the command line.

  {% include [cmake-visual-studio-generator](../_includes/work_src/reusage-installation/cmake-visual-studio-generator.md) %}

  {% include [cmake-unix-makefiles-generator](../_includes/work_src/reusage-installation/cmake-unix-makefiles-generator.md) %}

### JDK (only for components with JVM API)

  You have to install JDK to build components with JVM API (JVM applier and CatBoost for Apache Spark).
  JDK version has to be 8+ for JVM applier or strictly 8 for CatBoost for Apache Spark.

  Set `JAVA_HOME` environment variable to point to the path of JDK installation to be used during build.

### Python development artifacts (only for Python package)

  You have to install Python development artifacts (Python headers in an include directory and Python library for building modules).

  Note that they are specific to [CPython Python implementation](https://en.wikipedia.org/wiki/CPython). {{ product }} does not currently support other Python implementations like PyPy, Jython or IronPython.

  One convenient way to install different Python versions with development artifacts in one step is to use [pyenv](https://github.com/pyenv/pyenv) (and its variant for Windows - [pyenv-win](https://github.com/pyenv-win/pyenv-win))

### Node environment (only for Python package visualization widget)

  You have to have `Node.js` installed. `Yarn` package manager is also required.
