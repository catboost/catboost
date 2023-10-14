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

## Building steps

1. Clone the repository:

    ```
    git clone https://github.com/catboost/catboost.git
    ```

2. Go to the `catboost/catboost/jvm-packages/catboost4j-prediction` directory from the local copy of the CatBoost repository.
3. Use [the usual maven build phases](https://maven.apache.org/guides/introduction/introduction-to-the-lifecycle.html) in this directory.
