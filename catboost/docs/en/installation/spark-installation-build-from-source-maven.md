# Build from source using Maven

## Dependencies and requirements

{% note warning %}

{% include [ya-make-to-cmake-switch](../_includes/work_src/reusage-installation/ya-make-to-cmake-switch.md) %}

Select the appropriate build environment setup below accordingly.

{% endnote %}

* Linux or macOS. Windows support in progress.

* Set up build environment depending on build type:
  - [For CMake](build-environment-setup-for-cmake.md)
  - [For Ya Make](build-environment-setup-for-ya-make.md)

* Python. 3.6+
* Maven 3.3.9+
* JDK 8. Newer versions of JDK are not supported yet.

  Set `JAVA_HOME` environment variable to point to the path of JDK installation
* [Only for Ya Make] SWIG 4.0.2+

  Cmake-based build uses SWIG from Conan packages so an explicit installation is not required.

## Building steps

1. Clone the repository:

    ```
    git clone https://github.com/catboost/catboost.git
    ```

2. Go to the `catboost/catboost/spark/catboost4j-spark` directory from the local copy of the CatBoost repository.
3. Run `python ./generate_projects/generate.py` to generate Maven projects for all supported Spark and Scala versions combinations.
4. Go to the sub-directories `./projects/spark_<spark_compat_version>_<scala_compat_version>` for `spark_compat_version` and `scala_compat_version` you are interested in.
5. Use [the usual maven build phases](https://maven.apache.org/guides/introduction/introduction-to-the-lifecycle.html) in these directories.
