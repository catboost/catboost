# CUDA resources

`cuda_<version>.json` describes nvcc resources by build-host platform.
`DECLARE_EXTERNAL_HOST_RESOURCES_BUNDLE_BY_JSON` keeps the complete platform map
in the graph; the runner selects the native host archive after UID calculation.
The `_aarch64` variants for CUDA 11.4 and 13.0 preserve the distinct host archives
used when targeting AArch64.

`cuda_target_<version>.json` describes SDK resources by target platform.
`DECLARE_EXTERNAL_RESOURCE_BY_JSON` selects the SDK independently of the host,
including native x86-64 builds. The x86 SDK is the same archive as the x86 host
toolkit; AArch64 uses its separate target archive. CUDA 12.6 has a distinct
`_jetson` SDK. CUDA 12.6.3 on x86 aliases 12.6; legacy JetPack targets CUDA 10.2
and 11.4.19 use CUDA 11.8 host tools.

The JSON resource declarations validate toolkit availability. Unsupported
host/target and legacy JetPack combinations report `CUDA_NOT_FOUND`.
CUDA 11.4 on Windows and its MSVC host compiler also use JSON resources.
Resource descriptions retain toolkit patch versions and links to patches.

`mtime0` and `custom_pid` use the shared prebuilt-tool mechanism, with source
fallback when prebuilt tools are disabled or unavailable. See
[prebuilt CUDA preload tools](../../../../prebuilt/tools/cuda_preload/README.md)
for resource update instructions.
