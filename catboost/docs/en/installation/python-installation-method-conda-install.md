# conda install

{% include [reusage-installation-gpu-support-from-the-box__p](../_includes/work_src/reusage-installation/gpu-support-from-the-box__p.md) %}


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
    

