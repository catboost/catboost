# Command-line version binary

{% if audience == "internal" %}

{% note alert %}

{% include [arc_users](../yandex_specific/_includes/arcadia_users_step.md) %}

{% endnote %}

{% endif %}


{% include [installation-installation-methods](../_includes/work_src/reusage/installation-methods.md) %}


- [Download](../installation/cli-installation-binaries.md)
{% if audience == "internal" %}
- [{#T}](../yandex_specific/cli-build-from-arcadia-sources.md)
{% endif %}

{% if audience == "external" %}
- Build from source from a local source repository:
    - [For versions after [this commit](https://github.com/catboost/catboost/commit/c5c642ca0b8e093336d0229ac4b14c78db3915bb)] [Build using CMake](../installation/cli-installation-local-copy-installation.md#cmake)
    - [For versions prior to [this commit](https://github.com/catboost/catboost/commit/c5c642ca0b8e093336d0229ac4b14c78db3915bb)] [Build using Ya Make](../installation/cli-installation-local-copy-installation.md#ya-make)
    - [For versions prior to [this commit](https://github.com/catboost/catboost/commit/c5c642ca0b8e093336d0229ac4b14c78db3915bb)] [Build with `make` on Linux (CPU only)](../installation/cli-installation-make-install.md)
    - [For versions prior to [this commit](https://github.com/catboost/catboost/commit/c5c642ca0b8e093336d0229ac4b14c78db3915bb)] [Build with MPI support from a local copy (GPU only)](../installation/cli-installation-multi-node-installation.md)
{% endif %}

Note that there are additional [system requirements](#gpu-system-requirements) and [specifics](#gpu-peculiarities) if training on GPU is required.


## GPU system requirements {#gpu-system-requirements}

{% include [reusage-installation-cli__gpu-support-from-the-box__p](../_includes/work_src/reusage-installation/cli__gpu-support-from-the-box__p.md) %}


{% include [installation-compute-capability-requirements](../_includes/work_src/reusage-code-examples/compute-capability-requirements.md) %}


{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

## GPU specifics {#gpu-peculiarities}

- Some training parameters are missing but will be added in future releases
- Multiple train runs with the same random seed may result in different formulas because of the float summation order
