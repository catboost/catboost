
{% note info %}

Only CUDA {{ cuda_version__compiled-packages }} is officially supported in compiled command-line packages for Windows. [Вuild the binary from a local copy](../../../concepts/cli-installation.md) if GPU support is required and the installed version of CUDA differs from {{ cuda_version__compiled-packages }}. {{ product }} should work fine with CUDA {{ cuda_version__compiled-packages }} and later versions.

All necessary CUDA libraries are statically linked to the Linux and macOS binaries of the {{ product }} command-line version, therefore, the only installation necessary is the appropriate version of the CUDA driver.

{% endnote %}
