# Command-line version binary

{% note alert %}

{% include [installation-windows-visual-cplusplus-required](../_includes/work_src/reusage-code-examples/windows-visual-cplusplus-required.md) %}

{% if audience == "internal" %} {% include [arc_users](../yandex_specific/_includes/arcadia_users_step.md) %} {% endif %}

{% endnote %}


{% include [installation-installation-methods](../_includes/work_src/reusage/installation-methods.md) %}


- [Download](../installation/cli-installation-binaries.md)
{% if audience == "internal" %} - [{#T}](../yandex_specific/cli-build-from-arcadia-sources.md){% endif %}
{% if audience == "external" %} - [Build the binary from a local copy on Linux and macOS](../installation/cli-installation-local-copy-installation.md){% endif %}
{% if audience == "external" %} - [Build the binary from a local copy on Windows](../installation/cli-installation-local-copy-installation-windows.md){% endif %}
- [Build the binary with `make` on Linux (CPU only)](../installation/cli-installation-make-install.md)
- [Build the binary with MPI support from a local copy (GPU only)](../installation/cli-installation-multi-node-installation.md)

Note that there are additional [system requirements](#gpu-system-requirements) and [specifics](#gpu-peculiarities) if training on GPU is required.


## GPU system requirements {#gpu-system-requirements}

{% include [reusage-installation-cli__gpu-support-from-the-box__p](../_includes/work_src/reusage-installation/cli__gpu-support-from-the-box__p.md) %}


{% include [installation-compute-capability-requirements](../_includes/work_src/reusage-code-examples/compute-capability-requirements.md) %}


{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

{% if audience == "external" %}
The command-line version of {{ product }} for CUDA with compute capability 2.0 can be built from source. In this case the following steps are required:
- Step [2](../installation/cli-installation-local-copy-installation.md#build-cuda-2) of the [Build the binary from a local copy on Linux and macOS](../installation/cli-installation-local-copy-installation.md) operation.
- Step [2](../installation/cli-installation-local-copy-installation-windows.md#build-cuda-2) of the [Build the binary from a local copy on Windows](../installation/cli-installation-local-copy-installation-windows.md) operation.
{% endif %}

## GPU specifics {#gpu-peculiarities}

- Some training parameters are missing but will be added in future releases
- Multiple train runs with the same random seed may result in different formulas because of the float summation order
