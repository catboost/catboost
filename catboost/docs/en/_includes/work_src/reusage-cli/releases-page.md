
An up-to-date list of available {{ product }} releases and the corresponding binaries for different operating systems is available in the **Download** section of the [releases]({{ releases-page }}) page on GitHub.

|Operating system|CPU architectures|GPU support using [CUDA](https://developer.nvidia.com/cuda-zone)|
|--------|-----------------|------------|
| macOS (versions currently supported by Apple) | x86_64 and arm64 |no|
| Linux (compatible with [manylinux2014 platform tag](https://peps.python.org/pep-0599/) ) | x86_64 and aarch64 |yes|
| Windows 10 and 11 | x86_64 |yes|

{% note info %}

Release binaries for x86_64 CPU architectures are built with SIMD extensions SSE2, SSE3, SSSE3, SSE4 enabled. If you need to run {{ product }} on older CPUs that do not support these instruction sets [build {{ product }} artifacts yourself](../../../concepts/build-from-source.md)

{% endnote %}
