# Build R package from source directly from GitHub

{% note info %}

- {% include [r-package-r__recommended-installation-method](../_includes/work_src/reusage-installation/r__recommended-installation-method.md) %}

  {% if audience == "internal" %} {% include [arc_users](../yandex_specific/_includes/arcadia_users_step.md) %} {% endif %}

- {% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

{% endnote %}

To install the {{ r-package }} directly from the {{ product }} repository:

1. Set up build environment

{% include [setup-build-environment-alternatives](../_includes/work_src/reusage-installation/setup-build-environment-alternatives.md) %}

1. Run the following commands:

    ```
    install.packages('devtools')
    devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')
    ```

{% include [r__troubleshooting](../_includes/work_src/reusage-installation/r__troubleshooting.md) %}
