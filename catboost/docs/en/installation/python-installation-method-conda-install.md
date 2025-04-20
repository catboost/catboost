# conda install

{% include [supported-versions](../_includes/work_src/reusage-installation/python__supported-versions.md) %}

{% note info %}

All Conda packages for Windows have CUDA support.
Conda packages for Linux contain variants with or without CUDA support.

All necessary CUDA libraries are statically linked in the released Linux and Windows binaries, the only installation necessary is the appropriate version of the CUDA driver.

{% endnote %}

{% include [installation-install-from-conda-forge-channel](../_includes/work_src/reusage-common-phrases/install-from-conda-forge-channel.md) %}


1. Add conda-forge to your channels:
    ```no-highlight
    conda config --add channels conda-forge
    ```

1. Install {{ product }}:
    ```no-highlight
    conda install catboost
    ```

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

1. User-defined functions:

    {% include [python__user-defined-function-dependencies](../_includes/work_src/reusage-installation/python__user-defined-functions-dependencies.md) %}
