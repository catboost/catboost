# Development and contributions

## Build from source {#build-from-source}

The required steps for building {{ product }} depend on the implementation.

{% note info %}

Windows build currently requires Microsoft Visual studio 2015.3 toolset v140 and Windows 10 SDK (10.0.17134.0).

{% endnote %}

{% list tabs %}

- Command-line version

  1. Clone the repository:

      ```no-highlight
      {{ installation--git-clone }}
      ```

  1. Open the `catboost/catboost/app` directory from the local copy of the {{ product }} repository.

  1. Run the following command:
      ```no-highlight
      ../../ya make -d [-o <output directory>]
      ```

      Use the `-j <number of threads>` option to change the number of threads used when building the project.


- {{ python-package }}

  {% include [installation-packages-for-installation](../_includes/work_src/reusage/packages-for-installation.md) %}


    - `python3`
    - `python3-dev`
    - `numpy`
    - `pandas`

    To install the {{ python-package }}:

    1. Clone the repository:

        ```no-highlight
        {{ installation--git-clone }}
        ```

    1. Open the `catboost/catboost/python-package/catboost` directory from the local copy of the {{ product }} repository.

    1. {% include [installation-compile-the-library](../_includes/work_src/reusage-code-examples/compile-the-library.md) %}

        {% include [installation-installation-example](../_includes/work_src/reusage-code-examples/installation-example.md) %}

    1. Add the current directory to `PYTHONPATH` to use the built module on macOS or Linux:
        ```
        cd ../; export PYTHONPATH=$PYTHONPATH:$(pwd)
        ```

- {{ r-package }}

    You can build an extension module by running the `ya make` command in the `catboost/R-package/src` directory.

{% endlist %}


## Run tests {#run-tests}

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


## Develop in Windows {#compiling-in-windows}

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
