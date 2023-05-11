# Build from source

{% include [get-source-code-from-github](../_includes/work_src/reusage-installation/get-source-code-from-github.md) %}

{% note warning %}

{% include [ya-make-to-cmake-switch](../_includes/work_src/reusage-installation/ya-make-to-cmake-switch.md) %}

Select the appropriate build method below accordingly.

{% endnote %}

## Build using CMake {#build-cmake}

- [Build environment setup](../installation/build-environment-setup-for-cmake.md)
- [Build native artifacts](../installation/build-native-artifacts.md)
- By component:
    - [Build Python package](../installation/python-installation-method-build-from-source.md)
    - [Build CatBoost for Apache Spark](../installation/spark-installation-build-from-source-maven.md)
    - Build R package:
        - [Install from GitHub](../installation/r-installation-github-installation.md)
{% if audience == "external" %}
        - [Install from a local Git repository](../installation/r-installation-local-copy-installation.md)
{% endif %}

{% if audience == "external" %}
    - [Build Command-line binary](../installation/cli-installation-local-copy-installation.md)
{% endif %}

## Build using Ya Make (for previous versions) {#build-ya-make}

- [Build environment setup](../installation/build-environment-setup-for-ya-make.md)
- By component:
    - Build Python package
        - [In-place build on Linux and macOS](../installation/python-installation-method-build-from-source-linux-macos-using-ya-make.md)
        - [In-place build on Windows](../installation/python-installation-method-build-from-source-windows-using-ya-make.md)
        - [Build a wheel using mk_wheel.py](../installation/python-installation-method-build-a-wheel-package.md#mk-wheel)
    - [Build CatBoost for Apache Spark](../installation/spark-installation-build-from-source-maven.md)
    - Build R package:
        - [Install from GitHub](../installation/r-installation-github-installation.md)
{% if audience == "external" %}
        - [Install from a local Git repository](../installation/r-installation-local-copy-installation.md)
{% endif %}

{% if audience == "external" %}
    - [Build Command-line binary](../installation/cli-installation-local-copy-installation.md)
{% endif %}
