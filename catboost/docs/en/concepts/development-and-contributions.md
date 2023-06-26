# Development and contributions

## [Build from source](build-from-source.md) {#build-from-source}

## Run tests {#run-tests}

{% note warning %}

{% include [ya-make-to-cmake-switch](../_includes/work_src/reusage-installation/ya-make-to-cmake-switch.md) %}

{% endnote %}

### CMake-based build tests

- C/C++ libraries.

  C/C++ libraries contain tests for them in `ut` subdirectories in the source tree. For library in `x/y/z` the corresponding test code will be in `x/y/z/ut` and the target name will be `x-y-z-ut`.
  So, in order to run the test [run CMake](build-from-source.md#build-cmake) and then build the corresponding `x-y-z-ut` target. Building this target will produce an executable `${CMAKE_BUILD_DIR}/x/y/z/x-y-z-ut`. Run this executable to execute all the tests.

- {{ r-package }}

    1. Install additional R packages that are required to run tests:
        - `caret`
        - `dplyr`
        - `jsonlite`
        - `testthat`

    1. Open the `R-package` directory from the local copy of the {{ product }} repository.

    1. Run the following command:

        ```
        R CMD check .
        ```


    To run tests using the devtools package:

    1. Install [devtools](https://github.com/hadley/devtools).
    1. Run the following command from the R session:

        ```
        devtools::test()
        ```

- CLI

  TODO

- Python package

  TODO

- CatBoost for Apache Spark

    See [building CatBoost for Apache Spark from source](../installation/spark-installation-build-from-source-maven.md). Use standard `mvn test` command.

### YaMake-based build tests

{% note warning %}

The following documentation describes running tests using Ya Make which is applicable only for versions prior to [this commit](https://github.com/catboost/catboost/commit/c5c642ca0b8e093336d0229ac4b14c78db3915bb).

{% endnote %}

{{ product }} provides tests that check the compliance of the canonical data with the resulting data.

The required steps for running these tests depend on the implementation.

{% list tabs %}

- Command-line version

    1. {% include [test-common-tests](../_includes/work_src/reusage-installation/common-tests.md) %}

        1. Open the `catboost/pytest` directory from the local copy of the {{ product }} repository.

        1. Run the following command:
        ```bash
        ../../ya make -t -A [-Z]
        ```

        {% include [test-replace-cannonical-files](../_includes/work_src/reusage-installation/replace-cannonical-files.md) %}

    1. {% include [test-gpu-specific-tests](../_includes/work_src/reusage-installation/gpu-specific-tests.md) %}

        1. Open the `catboost/pytest/cuda_tests` directory from the local copy of the {{ product }} repository.

        1. Run the following command:

        ```bash
        ../../../ya make -DCUDA_ROOT=<path_to_CUDA_SDK> -t -A [-Z]
        ```

        - {% include [test-path-to-cuda](../_includes/work_src/reusage-installation/path-to-cuda.md) %}

        - {% include [test-replace-cannonical-files](../_includes/work_src/reusage-installation/replace-cannonical-files.md) %}

    {% include [test-use-vcs-to-analyze-diff](../_includes/work_src/reusage-installation/use-vcs-to-analyze-diff.md) %}


- {{ python-package }}

    1. {% include [test-common-tests](../_includes/work_src/reusage-installation/common-tests.md) %}

        1. Open the `catboost/python-package/ut/medium` directory from the local copy of the {{ product }} repository.

        1. Run the following command:
        ```no-highlight
        ../../../../ya make -t -A [-Z]
        ```

        {% include [test-replace-cannonical-files](../_includes/work_src/reusage-installation/replace-cannonical-files.md) %}

    1. {% include [test-gpu-specific-tests](../_includes/work_src/reusage-installation/gpu-specific-tests.md) %}

        1. Open the `catboost/python-package/ut/medium/gpu` directory from the local copy of the {{ product }} repository.

        1. Run the following command:
        ```
        ../../../../../ya make -DCUDA_ROOT=<path_to_CUDA_SDK> -t -A [-Z]
        ```

        - {% include [test-path-to-cuda](../_includes/work_src/reusage-installation/path-to-cuda.md) %}

        - {% include [test-replace-cannonical-files](../_includes/work_src/reusage-installation/replace-cannonical-files.md) %}

    {% include [test-use-vcs-to-analyze-diff](../_includes/work_src/reusage-installation/use-vcs-to-analyze-diff.md) %}

- {{ r-package }}

    1. Install additional R packages that are required to run tests:
        - `caret`
        - `dplyr`
        - `jsonlite`
        - `testthat`

    1. Open the `R-package` directory from the local copy of the {{ product }} repository.

    1. Run the following command:

        ```
        R CMD check .
        ```


    To run tests using the devtools package:

    1. Install [devtools](https://github.com/hadley/devtools).
    1. Run the following command from the R session:

        ```
        devtools::test()
        ```

{% endlist %}


## Microsoft Visual Studio solution {#compiling-in-windows}

{% note warning %}

Ready Microsoft Visual Studio solution had been provided until [this commit](https://github.com/catboost/catboost/commit/cd63b6c7313a28bcb40cd0674d73e356ad633de4).

For versions after this commit it is recommended [to generate Microsoft Visual Studio 2019 solution using the corresponding CMake generator](../installation/build-native-artifacts.md#build-cmake-conan-ninja).

{% endnote %}

A solution for Visual Studio is available in the {{ product }} repository:

```
catboost/msvs/arcadia.sln
```


## Coding conventions {#coding-convention}

The following coding conventions must be followed in order to successfully contribute to the {{ product }} project:
- [C++ style guide](https://github.com/catboost/catboost/blob/master/CPP_STYLE_GUIDE.md)
- [pep8](https://www.python.org/dev/peps/pep-0008/) for Python


## Versioning conventions {#versioning-conventions}

Do not change the package version when submitting pull requests. Yandex uses an internal repository for this purpose.


## Yandex Contributor License Agreement {#yandex-cla}

To contribute to {{ product }} you need to read the Yandex CLA and indicate that you agree to its terms. Details of how to do that and the text of the CLA can be found in [CONTRIBUTING.md](https://github.com/catboost/catboost/blob/master/CONTRIBUTING.md).
