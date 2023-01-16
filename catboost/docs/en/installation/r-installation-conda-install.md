# conda install

{% note info %}

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

{% endnote %}


{% include [reusage-installation-gpu-support-from-the-box__p](../_includes/work_src/reusage-installation/gpu-support-from-the-box__p.md) %}


{% include [installation-install-from-conda-forge-channel](../_includes/work_src/reusage-common-phrases/install-from-conda-forge-channel.md) %}


1. Add conda-forge to your channels:
    ```
    conda config --add channels conda-forge
    ```

1. Install {{ product }}:
    ```
    conda install r-catboost
    ```


