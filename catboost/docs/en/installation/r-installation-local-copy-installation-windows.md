# Install from a local copy on Windows

{% note info %}

- {% include [r-package-r__recommended-installation-method](../_includes/work_src/reusage-installation/r__recommended-installation-method.md) %}

- {% include [installation-windows-visual-cplusplus-required](../_includes/work_src/reusage-code-examples/windows-visual-cplusplus-required.md) %}

  {% if audience == "internal" %} {% include [arc_users](../yandex_specific/_includes/arcadia_users_step.md) %} {% endif %}

- {% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

{% endnote %}


To install the {{ r-package }} from a local copy of the {{ product }} repository on Windows:

1. [Download](https://git-scm.com/download/) and install Git locally.

1. Open Git Bash and create a directory for the local copy of the {{ product }} repository:
    ```no-highlight
    mkdir CatBoostRepository
    ```

1. Open the created directory.

1. (Optionally) Configure the proxy if required:
    ```no-highlight
    git config --global http.proxy http://login:password@ip_address:port_number
    ```

    - `login` and `password` are the proxy credentials
    - `ip_address` is the IP address of the proxy server
    - `port_number` is the configured proxy port number

1. Install Python and add it to PATH.


{% include [windows-build-setup](../_includes/work_src/reusage-installation/windows-build-setup.md) %}


1. Perform the following steps in RStudio:
    1. Check for package updates: .
    1. Open the `catboost/catboost/R-package` directory from the local copy of the {{ product }} repository.

    For example, if the cloned repository local path is `C:\CatBoostRepository` the following command should be used:
    ```
    setwd("C:/CatBoostRepository/catboost/catboost/R-package")
    ```

    1. Run the following commands:

    ```
    install.packages('devtools')
    devtools::build()
    devtools::install()
    ```

{% include [r__troubleshooting](../_includes/work_src/reusage-installation/r__troubleshooting.md) %}

