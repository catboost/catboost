# Build the CLI binary app from a local source repository

{% note warning %}

{% include [ya-make-to-cmake-switch](../_includes/work_src/reusage-installation/ya-make-to-cmake-switch.md) %}

Select the appropriate build method below accordingly.

{% endnote %}

## Source code

{% include [get-source-code-from-github](../_includes/work_src/reusage-installation/get-source-code-from-github.md) %}

## Build using CMake  {#cmake}

Build `catboost` target. See [Build native artifacts](build-native-artifacts.md).

## Build using Ya Make  {#ya-make}

1. [Setup build environment](build-environment-setup-for-ya-make.md)

1. Open the `catboost/catboost/app` directory from the local copy of the {{ product }} repository.

1. Run the following command:

    ```
    ../../ya make -r [optional parameters]
    ```

    Parameter | Description
    ----- | -----
    `-DCUDA_ROOT` | The path to CUDA. This parameter is required to support training on GPU.
    `-DHAVE_CUDA=no` | Disable CUDA support. This speeds up compilation.<br/><br/>By default, the package is built with CUDA support if CUDA Toolkit is installed.
    `-o` | The directory to output the compiled package to. By default, the current directory is used.
