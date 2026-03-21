# Build from source using Maven

## Dependencies and requirements

{% note warning %}

{% include [ya-make-to-cmake-switch](../_includes/work_src/reusage-installation/ya-make-to-cmake-switch.md) %}

Select the appropriate build environment setup below accordingly.

{% endnote %}

* Linux, macOS or Windows.

* Set up build environment depending on build type:
  - [For CMake](build-environment-setup-for-cmake.md)
  - [For Ya Make](build-environment-setup-for-ya-make.md)

* Python. 3.6+
* Maven 3.3.9+
* JDK 8+

  Set `JAVA_HOME` environment variable to point to the path of JDK installation

## CUDA support

[CUDA](https://developer.nvidia.com/cuda) support is available for Linux and Windows target platforms.
It is disabled by default and can be enabled by adding `--have-cuda` flag to `buildNative.extraArgs` definition. See below for details.

{% include [build-cuda-architectures](../_includes/work_src/reusage-installation/build-cuda-architectures.md) %}

## Building steps

1. Clone the repository:

    ```
    git clone https://github.com/catboost/catboost.git
    ```

2. Go to the `catboost/catboost/jvm-packages/catboost4j-prediction` directory from the local copy of the CatBoost repository.
3. Use [the usual maven build phases](https://maven.apache.org/guides/introduction/introduction-to-the-lifecycle.html) in this directory.

   Additional flags for building native dynamic library part can specified using `buildNative.extraArgs` definition.
   Supported flags are:
   * `--build-system=<build_system>`. Supported values are `CMAKE` (default), `YA`.
   * `--build-type=<build_type>`. Supported values are `Release` (default), `Debug`.
   * `--have-cuda`. Add to enable CUDA support.
   Example running tests with this definition:
   ```
   mvn test -DbuildNative.extraArgs="--build-type=Debug"
   ```
