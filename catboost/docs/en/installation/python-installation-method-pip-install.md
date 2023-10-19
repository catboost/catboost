# pip install

To install {{ product }} from pip:

1. Run the following command:

    ```no-highlight
    pip install catboost
    ```

    {% note info %}

    [PyPI](https://pypi.org/) contains precompiled wheels for most commonly used platform configurations:

    |Operating system|CPU architectures|GPU support using [CUDA](https://developer.nvidia.com/cuda-zone)|
    |--------|-----------------|------------|
    | macOS (versions currently supported by Apple) | x86_64 and arm64 |no|
    | Linux (compatible with [manylinux2014 platform tag](https://peps.python.org/pep-0599/) ) | x86_64 and aarch64 |yes|
    | Windows 10 and 11 | x86_64 |yes|

    If the platform where the installation is performed is incompatible with platform tags of the available precompiled wheels then `pip` will try to build {{ product }} python package from source. This approach requires certain [build dependencies and requirements](python-installation-method-build-from-source.md#dependencies-and-requirements) to be set up before the installation.

    {% endnote %}

    {% note info %}

    Release native binaries for x86_64 CPU architectures are built with SIMD extensions SSE2, SSE3, SSSE3, SSE4 enabled. If you need to run {{ product }} on older CPUs that do not support these instruction sets [build {{ product }} package from source](../installation/python-installation-method-build-from-source.md)

    {% endnote %}

1. Install visualization tools:
    1. {% include [visualization-tools-install-apywidgets-short](../_includes/work_src/reusage-installation/install-apywidgets-short.md) %}

    ```no-highlight
    pip install ipywidgets
    ```

    1. {% include [visualization-tools-turn-on-the-widgets-extension-intro](../_includes/work_src/reusage-installation/turn-on-the-widgets-extension-intro.md) %}

    ```no-highlight
    jupyter nbextension enable --py widgetsnbextension
    ```

    Refer to the following sections for details:
    - [Data visualization](../features/visualization.md)
    - [Additional packages for data visualization support](../installation/python-installation-additional-data-visualization-packages.md)
