# Build from source on Windows

{% if audience == "internal" %}

{% include [internal-note__use-outside-arcadia](../yandex_specific/_includes/note__use-outside-arcadia.md) %}

{% endif %}

{% include [installation-cuda-toolkit__compatible-system-compilers](../_includes/work_src/reusage-code-examples/cuda-toolkit__compatible-system-compilers.md) %}

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}


To build the {{ python-package }} from source on Windows:


{% include [windows-build-setup](../_includes/work_src/reusage-installation/windows-build-setup.md) %}


1. Open the `catboost/catboost/python-package/catboost` directory from the local copy of the {{ product }} repository.

1. {% include [installation-compile-the-library](../_includes/work_src/reusage-code-examples/compile-the-library.md) %}

    {% include [installation-installation-example](../_includes/work_src/reusage-code-examples/installation-example.md) %}

    {% note info %}

    Explicitly specify the `PYTHON_INCLUDE` and `PYTHON_LIBRARIES` variables:
    ```
    ../../../ya make -r -DUSE_ARCADIA_PYTHON=no -DOS_SDK=local -DPYTHON_INCLUDE="/I C:/Python27/include/" -DPYTHON_LIBRARIES="C:/Python27/libs/python27.lib"
    ```

    {% endnote %}
