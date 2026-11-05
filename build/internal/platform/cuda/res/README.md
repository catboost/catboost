# CUDA resources

`cuda_<version>.json` describes nvcc resources by build-host platform.
`DECLARE_EXTERNAL_HOST_RESOURCES_BUNDLE_BY_JSON` keeps the complete platform map
in the graph; the runner selects the native host archive after UID calculation.
Each CUDA version uses one host bundle for all target architectures. CUDA 11.4
uses Linux 11.4.4 host tools; CUDA 13.0 uses 13.0.1 tools, including the x86
compatibility libraries in the existing full toolkit archive.

`cuda_target_<version>.json` describes SDK resources by target platform.
`DECLARE_EXTERNAL_RESOURCE_BY_JSON` selects the SDK independently of the host,
including native x86-64 builds. The x86 SDK is the same archive as the x86 host
toolkit; AArch64 uses its separate target archive. CUDA 12.6 has a distinct
`_jetson` SDK. CUDA 12.6.3 aliases 12.6 for both targets; legacy JetPack targets CUDA 10.2
and 11.4.19 use CUDA 11.8 host tools.

`CudaToolkit` in `build/ymake_conf.py` is the support matrix for both automatic
CUDA detection and resource selection. It emits reactive resource filenames so
module-level CUDA settings keep working. Linux x86-64 and AArch64 hosts use the
same target matrix, including default CUDA 13.0 without an explicit version.
`HAVE_CUDA=no`, MUSL and sanitizer defaults still disable automatic CUDA use.
SDK entries have named fields for the host-tools JSON, target-SDK JSON,
required legacy host version and optional Jetson target-SDK JSON. Version
aliases are declared once and apply to every target that has the canonical SDK.
Unsupported host/target and legacy JetPack combinations report `CUDA_NOT_FOUND`
when bundled CUDA resources are requested explicitly.
CUDA 11.4 on Windows and its MSVC host compiler also use JSON resources.
Resource descriptions retain toolkit patch versions and links to patches.

Host compiler resource selection is emitted by `CudaToolkit` as well. The
resource module declares one host bundle for either Clang or MSVC; the JSON
platform map selects the executable architecture. To rebuild the MSVC archive,
install MSVS on Windows and run `devtools/tools_build/pack_sdk.py msvc out.tar`.
This patches `VC/Auxiliary/Build/vcvarsall.bat` to avoid nvcc's unsupported-host-
compiler check.

ARM target SDKs cover the same CUDA versions as Linux x86-64. Legacy JetPack
10.2 and 11.4.19 remain additional ARM-only pairings with `CUDA_HOST_VERSION=11.8`.
New SBSA SDKs have infinite retention. They contain the NVIDIA
release manifest, per-component licenses, `assembly.json` and the exact
`arcadia-compat.patch` applied to the headers. Both shared-library and static-library ELF architectures are checked.

CUDA 12.1 defaults to C++17 so the build selects its existing `libcxxcuda11`
dependency and matching include map. Its compiler cannot parse the current
libc++ builtins; the same failure reproduces with the archived x86 toolkit.
CUDA 12.2 and later retain C++20. Changing only a module's dialect does not
change the separately configured standard-library dependency.

The mixed-version entries are explicitly grouped as legacy JetPack pairings:
the existing `vhub` and `vhub4` profiles select 11.8 build tools with 10.2 and
11.4.19 target headers/libraries respectively. These curated combinations do
not imply support for arbitrary mixed CUDA versions. Host-tool selection does not depend on the target CPU. Target SDKs retain their
platform-specific releases: the existing ARM CUDA 11.4 SDK is 11.4.2, while its
Linux compiler tools and the x86 target SDK are 11.4.4. Windows 11.4 stays at
11.4.2. These patch versions are documented in the resource JSON descriptions.

For custom host tools, set `CUDA_HOST_TOOLS_JSON` to a host bundle JSON path
(for example `${ARCADIA_ROOT}/path/to/cuda_tools.json`) using the same
`by_platform` schema as the bundled toolchains. Target SDK selection is
unchanged. The old single-architecture `CUDA_AARCH64_HOST_RESOURCE` override
now reports a migration error; put its URI under `linux-aarch64` in the JSON
and include the corresponding `linux-x86_64` entry when sharing the bundle
across hosts. All hosts then use the same resource declaration and selection
mechanism.

cuDNN retains its single default version; set `CUDNN_VERSION` explicitly when
another version is needed. Compatibility checks remain in the cuDNN modules.

`mtime0` and `custom_pid` use the shared prebuilt-tool mechanism, with source
fallback when prebuilt tools are disabled or unavailable. See
[prebuilt CUDA preload tools](../../../../prebuilt/tools/cuda_preload/README.md)
for resource update instructions.
