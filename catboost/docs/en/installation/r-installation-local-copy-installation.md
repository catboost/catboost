# Install from a local copy on Linux and macOS

{% include [r-package-r__recommended-installation-method__note](../_includes/work_src/reusage-installation/r__recommended-installation-method__note.md) %}


{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}


To install the {{ r-package }} from a local copy of the {{ product }} repository:
1. {% if audience == "internal" %} {% include [arcadia_users_step](../yandex_specific/_includes/arcadia_users_step.md) %} {% endif %}Install the `libc` header files on macOS and Linux.

    Depending on the used OS:

    - macOS: `xcode-select --install`
    - Linux: Install the appropriate package (for example, `libc6-dev` on Ubuntu)

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

