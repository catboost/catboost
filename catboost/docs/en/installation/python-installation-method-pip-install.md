# pip install

{% include [reusage-installation-gpu-support-from-the-box__p](../_includes/work_src/reusage-installation/gpu-support-from-the-box__p.md) %}


To install {{ product }} from pip:

1. Run the following command:

    ```no-highlight
    pip install catboost
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

