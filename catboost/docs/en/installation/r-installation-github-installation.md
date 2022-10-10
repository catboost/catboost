# Build from source

{% note info %}

- {% include [r-package-r__recommended-installation-method](../_includes/work_src/reusage-installation/r__recommended-installation-method.md) %}

- {% include [installation-windows-visual-cplusplus-required](../_includes/work_src/reusage-code-examples/windows-visual-cplusplus-required.md) %}

  {% if audience == "internal" %} {% include [arc_users](../yandex_specific/_includes/arcadia_users_step.md) %} {% endif %}

- {% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

{% endnote %}


To install the {{ r-package }} directly from the {{ product }} repository:

1. {% if audience == "internal" %} {% include [arcadia_users_step](../yandex_specific/_includes/arcadia_users_step.md) %} {% endif %} Install the `libc` header files on macOS and Linux.

    Depending on the used OS:

    - macOS: `xcode-select --install`
    - Linux: Install the appropriate package (for example, `libc6-dev` on Ubuntu)

1. Run the following commands:

    ```
    install.packages('devtools')
    devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')
    ```

{% include [r__troubleshooting](../_includes/work_src/reusage-installation/r__troubleshooting.md) %}

