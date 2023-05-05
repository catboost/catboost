# Build from source using Maven

## Dependencies and requirements

* Linux or macOS. Windows support in progress.
* Python. 3.6+
* Maven 3.3.9+
* JDK 8. Newer versions of JDK are not supported yet.
* SWIG 4.0.2+

## Building steps

1. Clone the repository:

    ```
    git clone https://github.com/catboost/catboost.git
    ```

2. Go to the `catboost/catboost/spark/catboost4j-spark` directory from the local copy of the CatBoost repository.
3. Run `python ./generate_projects/generate.py` to generate Maven projects for all supported Spark and Scala versions combinations.
4. Go to the sub-directories `./projects/spark_<spark_compat_version>_<scala_compat_version>` for `spark_compat_version` and `scala_compat_version` you are interested in.
5. Use [the usual maven build phases](https://maven.apache.org/guides/introduction/introduction-to-the-lifecycle.html) in these directories.
