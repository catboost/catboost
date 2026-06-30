# Install R package from a local source repository copy

{% include [r-package-r__recommended-installation-method__note](../_includes/work_src/reusage-installation/r__recommended-installation-method__note.md) %}


{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}


To install the {{ r-package }} from a local copy of the {{ product }} repository:
1. Set up build environment

    {% include [setup-build-environment-alternatives](../_includes/work_src/reusage-installation/setup-build-environment-alternatives.md) %}

1. Clone the repository:
    ```
    {{ installation--git-clone }}
    ```

1. Open the `catboost/catboost/R-package` directory from the local copy of the {{ product }} repository.
1. {% include [reusage-installation-using-the-following-code](../_includes/work_src/reusage-installation/using-the-following-code.md) %}

    ```
    install.packages('devtools')
    devtools::build()
    devtools::install()
    ```

{% include [r__troubleshooting](../_includes/work_src/reusage-installation/r__troubleshooting.md) %}
