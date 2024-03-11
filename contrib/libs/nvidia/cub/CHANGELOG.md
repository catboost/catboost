# CUB 2.1.0

## Breaking Changes

- NVIDIA/cub#553: Deprecate the `CUB_USE_COOPERATIVE_GROUPS` macro, as all supported CTK
  distributions provide CG. This macro will be removed in a future version of CUB.

## New Features

- NVIDIA/cub#359: Add new `DeviceBatchMemcpy` algorithm.
- NVIDIA/cub#565: Add `DeviceMergeSort::StableSortKeysCopy` API. Thanks to David Wendt (@davidwendt)
  for this contribution.
- NVIDIA/cub#585: Add SM90 tuning policy for `DeviceRadixSort`. Thanks to Andy Adinets (@canonizer)
  for this contribution.
- NVIDIA/cub#586: Introduce a new mechanism to opt-out of compiling CDP support in CUB algorithms by
  defining `CUB_DISABLE_CDP`.
- NVIDIA/cub#589: Support 64-bit indexing in `DeviceReduce`.
- NVIDIA/cub#607: Support 128-bit integers in radix sort.

## Bug Fixes

- NVIDIA/cub#547: Resolve several long-running issues resulting from using multiple versions of CUB
  within the same process. Adds an inline namespace that encodes CUB version and targeted PTX
  architectures.
- NVIDIA/cub#562: Fix bug in `BlockShuffle` resulting from an invalid thread offset. Thanks to
  @sjfeng1999 for this contribution.
- NVIDIA/cub#564: Fix bug in `BlockRadixRank` when used with blocks that are not a multiple of 32
  threads.
- NVIDIA/cub#579: Ensure that all threads in the logical warp participate in the index-shuffle
  for `BlockRadixRank`. Thanks to Andy Adinets (@canonizer) for this contribution.
- NVIDIA/cub#582: Fix reordering in CUB member initializer lists.
- NVIDIA/cub#589: Fix `DeviceSegmentedSort` when used with `bool` keys.
- NVIDIA/cub#590: Fix CUB's CMake install rules. Thanks to Robert Maynard (@robertmaynard) for this
  contribution.
- NVIDIA/cub#592: Fix overflow in `DeviceReduce`.
- NVIDIA/cub#598: Fix `DeviceRunLengthEncode` when the first item is a `NaN`.
- NVIDIA/cub#611: Fix `WarpScanExclusive` for vector types.

## Other Enhancements

- NVIDIA/cub#537: Add detailed and expanded version of
  a [CUB developer overview](https://github.com/NVIDIA/cub/blob/main/docs/developer_overview.md).
- NVIDIA/cub#549: Fix `BlockReduceRaking` docs for non-commutative operations. Thanks to Tobias
  Ribizel (@upsj) for this contribution.
- NVIDIA/cub#606: Optimize CUB's decoupled-lookback implementation.

# CUB 2.0.1

## Other Enhancements

- Skip device-side synchronization on SM90+. These syncs are a debugging-only feature and not
  required for correctness, and a warning will be emitted if this happens.

# CUB 2.0.0

## Summary

The CUB 2.0.0 major release adds a dependency on libcu++ and contains several
breaking changes. These include new diagnostics when inspecting device-only
lambdas from the host, an updated method of determining accumulator types for
algorithms like Reduce and Scan, and a compile-time replacement for the
runtime `debug_synchronous` debugging flags.

This release also includes several new features. `DeviceHistogram` now
supports `__half` and better handles various edge cases. `WarpReduce` now
performs correctly when restricted to a single-thread “warp”, and will use
the `__reduce_add_sync` accelerated intrinsic (introduced with Ampere) when
appropriate. `DeviceRadixSort` learned to handle the case
where `begin_bit == end_bit`.

Several algorithms also have updated documentation, with a particular focus on
clarifying which operations can and cannot be performed in-place.

## Breaking Changes

- NVIDIA/cub#448 Add libcu++ dependency (v1.8.0+).
- NVIDIA/cub#448: The following macros are no longer defined by default. They
  can be re-enabled by defining `CUB_PROVIDE_LEGACY_ARCH_MACROS`. These will be
  completely removed in a future release.
  - `CUB_IS_HOST_CODE`: Replace with `NV_IF_TARGET`.
  - `CUB_IS_DEVICE_CODE`: Replace with `NV_IF_TARGET`.
  - `CUB_INCLUDE_HOST_CODE`: Replace with `NV_IF_TARGET`.
  - `CUB_INCLUDE_DEVICE_CODE`: Replace with `NV_IF_TARGET`.
- NVIDIA/cub#486: CUB's CUDA Runtime support macros have been updated to
  support `NV_IF_TARGET`. They are now defined consistently across all
  host/device compilation passes. This should not affect most usages of these
  macros, but may require changes for some edge cases.
  - `CUB_RUNTIME_FUNCTION`: Execution space annotations for functions that
    invoke CUDA Runtime APIs.
    - Old behavior:
      - RDC enabled: Defined to `__host__ __device__`
      - RDC not enabled:
        - NVCC host pass: Defined to `__host__ __device__`
        - NVCC device pass: Defined to `__host__`
    - New behavior:
      - RDC enabled: Defined to `__host__ __device__`
      - RDC not enabled: Defined to `__host__`
  - `CUB_RUNTIME_ENABLED`: No change in behavior, but no longer used in CUB.
    Provided for legacy support only. Legacy behavior:
    - RDC enabled: Macro is defined.
    - RDC not enabled:
      - NVCC host pass: Macro is defined.
      - NVCC device pass: Macro is not defined.
  - `CUB_RDC_ENABLED`: New macro, may be combined with `NV_IF_TARGET` to replace
    most usages of `CUB_RUNTIME_ENABLED`. Behavior:
    - RDC enabled: Macro is defined.
    - RDC not enabled: Macro is not defined.
- NVIDIA/cub#509: A compile-time error is now emitted when a `__device__`-only
  lambda's return type is queried from host code (requires libcu++ ≥ 1.9.0).
  - Due to limitations in the CUDA programming model, the result of this query
    is unreliable, and will silently return an incorrect result. This leads to
    difficult to debug errors.
  - When using libcu++ 1.9.0, an error will be emitted with information about
    work-arounds:
    - Use a named function object with a `__device__`-only implementation
      of `operator()`.
    - Use a `__host__ __device__` lambda.
    - Use `cuda::proclaim_return_type` (Added in libcu++ 1.9.0)
- NVIDIA/cub#509: Use the result type of the binary reduction operator for
  accumulating intermediate results in the `DeviceReduce` algorithm, following
  guidance from http://wg21.link/P2322R6.
  - This change requires host-side introspection of the binary operator's
    signature, and device-only extended lambda functions can no longer be used.
  - In addition to the behavioral changes, the interfaces for
    the `Dispatch*Reduce` layer have changed:
    - `DispatchReduce`:
      - Now accepts accumulator type as last parameter.
      - Now accepts initializer type instead of output iterator value type.
      - Constructor now accepts `init` as initial type instead of output
        iterator value type.
    - `DispatchSegmentedReduce`:
      - Accepts accumulator type as last parameter.
      - Accepts initializer type instead of output iterator value type.
  - Thread operators now accept parameters using different types: `Equality`
    , `Inequality`, `InequalityWrapper`, `Sum`, `Difference`, `Division`, `Max`
    , `ArgMax`, `Min`, `ArgMin`.
  - `ThreadReduce` now accepts accumulator type and uses a different type
    for `prefix`.
- NVIDIA/cub#511: Use the result type of the binary operator for accumulating
  intermediate results in the `DeviceScan`, `DeviceScanByKey`,
  and `DeviceReduceByKey` algorithms, following guidance
  from http://wg21.link/P2322R6.
  - This change requires host-side introspection of the binary operator's
    signature, and device-only extended lambda functions can no longer be used.
  - In addition to the behavioral changes, the interfaces for the `Dispatch`
    layer have changed:
    - `DispatchScan`now accepts accumulator type as a template parameter.
    - `DispatchScanByKey`now accepts accumulator type as a template parameter.
    - `DispatchReduceByKey`now accepts accumulator type as the last template
      parameter.
- NVIDIA/cub#527: Deprecate the `debug_synchronous` flags on device algorithms.
  - This flag no longer has any effect. Define `CUB_DEBUG_SYNC` during
    compilation to enable these checks.
  - Moving this option from run-time to compile-time avoids the compilation
    overhead of unused debugging paths in production code.

## New Features

- NVIDIA/cub#514: Support `__half` in `DeviceHistogram`.
- NVIDIA/cub#516: Add support for single-threaded invocations of `WarpReduce`.
- NVIDIA/cub#516: Use `__reduce_add_sync` hardware acceleration for `WarpReduce`
  on supported architectures.

## Bug Fixes

- NVIDIA/cub#481: Fix the device-wide radix sort implementations to simply copy
  the input to the output when `begin_bit == end_bit`.
- NVIDIA/cub#487: Fix `DeviceHistogram::Even` for a variety of edge cases:
  - Bin ids are now correctly computed when mixing different types for `SampleT`
    and `LevelT`.
  - Bin ids are now correctly computed when `LevelT` is an integral type and the
    number of levels does not evenly divide the level range.
- NVIDIA/cub#508: Ensure that `temp_storage_bytes` is properly set in
  the `AdjacentDifferenceCopy` device algorithms.
- NVIDIA/cub#508: Remove excessive calls to the binary operator given to
  the `AdjacentDifferenceCopy` device algorithms.
- NVIDIA/cub#533: Fix debugging utilities when RDC is disabled.

## Other Enhancements

- NVIDIA/cub#448: Removed special case code for unsupported CUDA architectures.
- NVIDIA/cub#448: Replace several usages of `__CUDA_ARCH__` with `<nv/target>`
  to handle host/device code divergence.
- NVIDIA/cub#448: Mark unused PTX arch parameters as legacy.
- NVIDIA/cub#476: Enabled additional debug logging for the onesweep radix sort
  implementation. Thanks to @canonizer for this contribution.
- NVIDIA/cub#480: Add `CUB_DISABLE_BF16_SUPPORT` to avoid including
  the `cuda_bf16.h` header or using the `__nv_bfloat16` type.
- NVIDIA/cub#486: Add debug log messages for post-kernel debug synchronizations.
- NVIDIA/cub#490: Clarify documentation for in-place usage of `DeviceScan`
  algorithms.
- NVIDIA/cub#494: Clarify documentation for in-place usage of `DeviceHistogram`
  algorithms.
- NVIDIA/cub#495: Clarify documentation for in-place usage of `DevicePartition`
  algorithms.
- NVIDIA/cub#499: Clarify documentation for in-place usage of `Device*Sort`
  algorithms.
- NVIDIA/cub#500: Clarify documentation for in-place usage of `DeviceReduce`
  algorithms.
- NVIDIA/cub#501: Clarify documentation for in-place usage
  of `DeviceRunLengthEncode` algorithms.
- NVIDIA/cub#503: Clarify documentation for in-place usage of `DeviceSelect`
  algorithms.
- NVIDIA/cub#518: Fix typo in `WarpMergeSort` documentation.
- NVIDIA/cub#519: Clarify segmented sort documentation regarding the handling of
  elements that are not included in any segment.

# CUB 1.17.2

## Summary

CUB 1.17.2 is a minor bugfix release.

- NVIDIA/cub#547: Introduce an annotated inline namespace to prevent issues with
  collisions and mismatched kernel configurations across libraries. The new
  namespace encodes the CUB version and target SM architectures.

# CUB 1.17.1

## Summary

CUB 1.17.1 is a minor bugfix release.

- NVIDIA/cub#508: Ensure that `temp_storage_bytes` is properly set in
  the `AdjacentDifferenceCopy` device algorithms.
- NVIDIA/cub#508: Remove excessive calls to the binary operator given to
  the `AdjacentDifferenceCopy` device algorithms.
- Fix device-side debug synchronous behavior in `DeviceSegmentedSort`.

# CUB 1.17.0

## Summary

CUB 1.17.0 is the final minor release of the 1.X series. It provides a variety
of bug fixes and miscellaneous enhancements, detailed below.

## Known Issues

### "Run-to-run" Determinism Broken

Several CUB device algorithms are documented to provide deterministic results
(per device) for non-associative reduction operators (e.g. floating-point
addition). Unfortunately, the implementations of these algorithms contain
performance optimizations that violate this guarantee.
The `DeviceReduce::ReduceByKey` and `DeviceScan` algorithms are known to be
affected. We're currently evaluating the scope and impact of correcting this in
a future CUB release. See NVIDIA/cub#471 for details.

## Bug Fixes

- NVIDIA/cub#444: Fixed `DeviceSelect` to work with discard iterators and mixed
  input/output types.
- NVIDIA/cub#452: Fixed install issue when `CMAKE_INSTALL_LIBDIR` contained
  nested directories. Thanks to @robertmaynard for this contribution.
- NVIDIA/cub#462: Fixed bug that produced incorrect results
  from `DeviceSegmentedSort` on sm_61 and sm_70.
- NVIDIA/cub#464: Fixed `DeviceSelect::Flagged` so that flags are normalized to
  0 or 1.
- NVIDIA/cub#468: Fixed overflow issues in `DeviceRadixSort` given `num_items`
  close to 2^32. Thanks to @canonizer for this contribution.
- NVIDIA/cub#498: Fixed compiler regression in `BlockAdjacentDifference`.
  Thanks to @MKKnorr for this contribution.

## Other Enhancements

- NVIDIA/cub#445: Remove device-sync in `DeviceSegmentedSort` when launched via
  CDP.
- NVIDIA/cub#449: Fixed invalid link in documentation. Thanks to @kshitij12345
  for this contribution.
- NVIDIA/cub#450: `BlockDiscontinuity`: Replaced recursive-template loop
  unrolling with `#pragma unroll`. Thanks to @kshitij12345 for this
  contribution.
- NVIDIA/cub#451: Replaced the deprecated `TexRefInputIterator` implementation
  with an alias to `TexObjInputIterator`. This fully removes all usages of the
  deprecated CUDA texture reference APIs from CUB.
- NVIDIA/cub#456: `BlockAdjacentDifference`: Replaced recursive-template loop
  unrolling with `#pragma unroll`. Thanks to @kshitij12345 for this
  contribution.
- NVIDIA/cub#466: `cub::DeviceAdjacentDifference` API has been updated to use
  the new `OffsetT` deduction approach described in NVIDIA/cub#212.
- NVIDIA/cub#470: Fix several doxygen-related warnings. Thanks to @karthikeyann
  for this contribution.

# CUB 1.16.0

## Summary

CUB 1.16.0 is a major release providing several improvements to the device scope
algorithms. `DeviceRadixSort` now supports large (64-bit indexed) input data. A
new `UniqueByKey` algorithm has been added to `DeviceSelect`.
`DeviceAdjacentDifference` provides new `SubtractLeft` and `SubtractRight`
functionality.

This release also deprecates several obsolete APIs, including type traits
and `BlockAdjacentDifference` algorithms. Many bugfixes and documentation
updates are also included.

### 64-bit Offsets in `DeviceRadixSort` Public APIs

Users frequently want to process large datasets using CUB's device-scope
algorithms, but the current public APIs limit input data sizes to those that can
be indexed by a 32-bit integer. Beginning with this release, CUB is updating
these APIs to support 64-bit offsets, as discussed in NVIDIA/cub#212.

The device-scope algorithms will be updated with 64-bit offset support
incrementally, starting with the `cub::DeviceRadixSort` family of algorithms.
Thanks to @canonizer for contributing this functionality.

### New `DeviceSelect::UniqueByKey` Algorithm

`cub::DeviceSelect` now provides a `UniqueByKey` algorithm, which has been
ported from Thrust. Thanks to @zasdfgbnm for this contribution.

### New `DeviceAdjacentDifference` Algorithms

The new `cub::DeviceAdjacentDifference` interface, also ported from Thrust,
provides `SubtractLeft` and `SubtractRight` algorithms as CUB kernels.

## Deprecation Notices

### Synchronous CUDA Dynamic Parallelism Support

**A future version of CUB will change the `debug_synchronous` behavior of
device-scope algorithms when invoked via CUDA Dynamic Parallelism (CDP).**

This will only affect calls to CUB device-scope algorithms launched from
device-side code with `debug_synchronous = true`. Such invocations will continue
to print extra debugging information, but they will no longer synchronize after
kernel launches.

### Deprecated Traits

CUB provided a variety of metaprogramming type traits in order to support C++03.
Since C++14 is now required, these traits have been deprecated in favor of their
STL equivalents, as shown below:

| Deprecated CUB Trait  | Replacement STL Trait |
|-----------------------|-----------------------|
| cub::If               | std::conditional      |
| cub::Equals           | std::is_same          |
| cub::IsPointer        | std::is_pointer       |
| cub::IsVolatile       | std::is_volatile      |
| cub::RemoveQualifiers | std::remove_cv        |
| cub::EnableIf         | std::enable_if        |

CUB now uses the STL traits internally, resulting in a ~6% improvement in
compile time.

### Misnamed `cub::BlockAdjacentDifference` APIs

The algorithms in `cub::BlockAdjacentDifference` have been deprecated, as their
names did not clearly describe their intent. The `FlagHeads` method is
now `SubtractLeft`, and `FlagTails` has been replaced by `SubtractRight`.

## Breaking Changes

- NVIDIA/cub#331: Deprecate the misnamed `BlockAdjacentDifference::FlagHeads`
  and `FlagTails` methods. Use the new `SubtractLeft` and `SubtractRight`
  methods instead.
- NVIDIA/cub#364: Deprecate some obsolete type traits. These should be replaced
  by the equivalent traits in `<type_traits>` as described above.

## New Features

- NVIDIA/cub#331: Port the `thrust::adjacent_difference` kernel and expose it
  as `cub::DeviceAdjacentDifference`.
- NVIDIA/cub#405: Port the `thrust::unique_by_key` kernel and expose it
  as `cub::DeviceSelect::UniqueByKey`. Thanks to @zasdfgbnm for this
  contribution.

## Enhancements

- NVIDIA/cub#340: Allow 64-bit offsets in `DeviceRadixSort` public APIs. Thanks
  to @canonizer for this contribution.
- NVIDIA/cub#400: Implement a significant reduction in `DeviceMergeSort`
  compilation time.
- NVIDIA/cub#415: Support user-defined `CMAKE_INSTALL_INCLUDEDIR` values in
  Thrust's CMake install rules. Thanks for @robertmaynard for this contribution.

## Bug Fixes

- NVIDIA/cub#381: Fix shared memory alignment in `dyn_smem` example.
- NVIDIA/cub#393: Fix some collisions with the `min`/`max`  macros defined
  in `windows.h`.
- NVIDIA/cub#404: Fix bad cast in `util_device`.
- NVIDIA/cub#410: Fix CDP issues in `DeviceSegmentedSort`.
- NVIDIA/cub#411: Ensure that the `nv_exec_check_disable` pragma is only used on
  nvcc.
- NVIDIA/cub#418: Fix `-Wsizeof-array-div` warning on gcc 11. Thanks to
  @robertmaynard for this contribution.
- NVIDIA/cub#420: Fix new uninitialized variable warning in `DiscardIterator` on
  gcc 10.
- NVIDIA/cub#423: Fix some collisions with the `small` macro defined
  in `windows.h`.
- NVIDIA/cub#426: Fix some issues with version handling in CUB's CMake packages.
- NVIDIA/cub#430: Remove documentation for `DeviceSpmv` parameters that are
  absent from public APIs.
- NVIDIA/cub#432: Remove incorrect documentation for `DeviceScan` algorithms
  that guaranteed run-to-run deterministic results for floating-point addition.

# CUB 1.15.0 (NVIDIA HPC SDK 22.1, CUDA Toolkit 11.6)

## Summary

CUB 1.15.0 includes a new `cub::DeviceSegmentedSort` algorithm, which
demonstrates up to 5000x speedup compared to `cub::DeviceSegmentedRadixSort`
when sorting a large number of small segments. A new `cub::FutureValue<T>`
helper allows the `cub::DeviceScan` algorithms to lazily load the
`initial_value` from a pointer. `cub::DeviceScan` also added `ScanByKey`
functionality.

The new `DeviceSegmentedSort` algorithm partitions segments into size groups.
Each group is processed with specialized kernels using a variety of sorting
algorithms. This approach varies the number of threads allocated for sorting
each segment and utilizes the GPU more efficiently.

`cub::FutureValue<T>` provides the ability to use the result of a previous
kernel as a scalar input to a CUB device-scope algorithm without unnecessary
synchronization:

```cpp
int *d_intermediate_result = ...;
intermediate_kernel<<<blocks, threads>>>(d_intermediate_result,  // output
                                         arg1,                   // input
                                         arg2);                  // input

// Wrap the intermediate pointer in a FutureValue -- no need to explicitly
// sync when both kernels are stream-ordered. The pointer is read after
// the ExclusiveScan kernel starts executing.
cub::FutureValue<int> init_value(d_intermediate_result);

cub::DeviceScan::ExclusiveScan(d_temp_storage,
                               temp_storage_bytes,
                               d_in,
                               d_out,
                               cub::Sum(),
                               init_value,
                               num_items);
```

Previously, an explicit synchronization would have been necessary to obtain the
intermediate result, which was passed by value into ExclusiveScan. This new
feature enables better performance in workflows that use cub::DeviceScan.

## Deprecation Notices

**A future version of CUB will change the `debug_synchronous` behavior of
device-scope algorithms when invoked via CUDA Dynamic Parallelism (CDP).**

This will only affect calls to CUB device-scope algorithms launched from
device-side code with `debug_synchronous = true`. These algorithms will continue
to print extra debugging information, but they will no longer synchronize after
kernel launches.

## Breaking Changes

- NVIDIA/cub#305: The template parameters of `cub::DispatchScan` have changed to
  support the new `cub::FutureValue` helper. More details under "New Features".
- NVIDIA/cub#377: Remove broken `operator->()` from
  `cub::TransformInputIterator`, since this cannot be implemented without
  returning a temporary object's address. Thanks to Xiang Gao (@zasdfgbnm) for
  this contribution.

## New Features

- NVIDIA/cub#305: Add overloads to `cub::DeviceScan` algorithms that allow the
  output of a previous kernel to be used as `initial_value` without explicit
  synchronization. See the new `cub::FutureValue` helper for details. Thanks to
  Xiang Gao (@zasdfgbnm) for this contribution.
- NVIDIA/cub#354: Add `cub::BlockRunLengthDecode` algorithm. Thanks to Elias
  Stehle (@elstehle) for this contribution.
- NVIDIA/cub#357: Add `cub::DeviceSegmentedSort`, an optimized version
  of `cub::DeviceSegmentedSort` with improved load balancing and small array
  performance.
- NVIDIA/cub#376: Add "by key" overloads to `cub::DeviceScan`. Thanks to Xiang
  Gao (@zasdfgbnm) for this contribution.

## Bug Fixes

- NVIDIA/cub#349: Doxygen and unused variable fixes.
- NVIDIA/cub#363: Maintenance updates for the new `cub::DeviceMergeSort`
  algorithms.
- NVIDIA/cub#382: Fix several `-Wconversion` warnings. Thanks to Matt Stack
  (@matt-stack) for this contribution.
- NVIDIA/cub#388: Fix debug assertion on MSVC when using
  `cub::CachingDeviceAllocator`.
- NVIDIA/cub#395: Support building with `__CUDA_NO_HALF_CONVERSIONS__`. Thanks
  to Xiang Gao (@zasdfgbnm) for this contribution.

# CUB 1.14.0 (NVIDIA HPC SDK 21.9)

## Summary

CUB 1.14.0 is a major release accompanying the NVIDIA HPC SDK 21.9.

This release provides the often-requested merge sort algorithm, ported from the
`thrust::sort` implementation. Merge sort provides more flexibility than the
existing radix sort by supporting arbitrary data types and comparators, though
radix sorting is still faster for supported inputs. This functionality is
provided through the new `cub::DeviceMergeSort` and `cub::BlockMergeSort`
algorithms.

The namespace wrapping mechanism has been overhauled for 1.14. The existing
macros (`CUB_NS_PREFIX`/`CUB_NS_POSTFIX`) can now be replaced by a single macro,
`CUB_WRAPPED_NAMESPACE`, which is set to the name of the desired wrapped
namespace. Defining a similar `THRUST_CUB_WRAPPED_NAMESPACE` macro will embed
both `thrust::` and `cub::` symbols in the same external namespace. The
prefix/postfix macros are still supported, but now require a new
`CUB_NS_QUALIFIER` macro to be defined, which provides the fully qualified CUB
namespace (e.g. `::foo::cub`). See `cub/util_namespace.cuh` for details.

## Breaking Changes

- NVIDIA/cub#350: When the `CUB_NS_[PRE|POST]FIX` macros are set,
  `CUB_NS_QUALIFIER` must also be defined to the fully qualified CUB namespace
  (e.g. `#define CUB_NS_QUALIFIER ::foo::cub`). Note that this is handled
  automatically when using the new `[THRUST_]CUB_WRAPPED_NAMESPACE` mechanism.

## New Features

- NVIDIA/cub#322: Ported the merge sort algorithm from Thrust:
  `cub::BlockMergeSort` and `cub::DeviceMergeSort` are now available.
- NVIDIA/cub#326: Simplify the namespace wrapper macros, and detect when
  Thrust's symbols are in a wrapped namespace.

## Bug Fixes

- NVIDIA/cub#160, NVIDIA/cub#163, NVIDIA/cub#352: Fixed several bugs in
  `cub::DeviceSpmv` and added basic tests for this algorithm. Thanks to James
  Wyles and Seunghwa Kang for their contributions.
- NVIDIA/cub#328: Fixed error handling bug and incorrect debugging output in
  `cub::CachingDeviceAllocator`. Thanks to Felix Kallenborn for this
  contribution.
- NVIDIA/cub#335: Fixed a compile error affecting clang and NVRTC. Thanks to
  Jiading Guo for this contribution.
- NVIDIA/cub#351: Fixed some errors in the `cub::DeviceHistogram` documentation.

## Enhancements

- NVIDIA/cub#348: Add an example that demonstrates how to use dynamic shared
  memory with a CUB block algorithm. Thanks to Matthias Jouanneaux for this
  contribution.

# CUB 1.13.1 (CUDA Toolkit 11.5)

CUB 1.13.1 is a minor release accompanying the CUDA Toolkit 11.5.

This release provides a new hook for embedding the `cub::` namespace inside
a custom namespace. This is intended to work around various issues related to
linking multiple shared libraries that use CUB. The existing `CUB_NS_PREFIX` and
`CUB_NS_POSTFIX` macros already provided this capability; this update provides a
simpler mechanism that is extended to and integrated with Thrust. Simply define
`THRUST_CUB_WRAPPED_NAMESPACE` to a namespace name, and both `thrust::` and
`cub::` will be placed inside the new namespace. Using different wrapped
namespaces for each shared library will prevent issues like those reported in
NVIDIA/thrust#1401.

## New Features

- NVIDIA/cub#326: Add `THRUST_CUB_WRAPPED_NAMESPACE` hooks.

# CUB 1.13.0 (NVIDIA HPC SDK 21.7)

CUB 1.13.0 is the major release accompanying the NVIDIA HPC SDK 21.7 release.

Notable new features include support for striped data arrangements in block
load/store utilities, `bfloat16` radix sort support, and fewer restrictions on
offset iterators in segmented device algorithms. Several bugs
in `cub::BlockShuffle`, `cub::BlockDiscontinuity`, and `cub::DeviceHistogram`
have been addressed. The amount of code generated in `cub::DeviceScan` has been
greatly reduced, leading to significant compile-time improvements when targeting
multiple PTX architectures.

This release also includes several user-contributed documentation fixes that
will be reflected in CUB's online documentation in the coming weeks.

## Breaking Changes

- NVIDIA/cub#320: Deprecated `cub::TexRefInputIterator<T, UNIQUE_ID>`. Use
  `cub::TexObjInputIterator<T>` as a replacement.

## New Features

- NVIDIA/cub#274: Add `BLOCK_LOAD_STRIPED` and `BLOCK_STORE_STRIPED`
  functionality to `cub::BlockLoadAlgorithm` and `cub::BlockStoreAlgorithm`.
  Thanks to Matthew Nicely (@mnicely) for this contribution.
- NVIDIA/cub#291: `cub::DeviceSegmentedRadixSort` and
  `cub::DeviceSegmentedReduce` now support different types for begin/end
  offset iterators. Thanks to Sergey Pavlov (@psvvsp) for this contribution.
- NVIDIA/cub#306: Add `bfloat16` support to `cub::DeviceRadixSort`. Thanks to
  Xiang Gao (@zasdfgbnm) for this contribution.
- NVIDIA/cub#320: Introduce a new `CUB_IGNORE_DEPRECATED_API` macro that
  disables deprecation warnings on Thrust and CUB APIs.

## Bug Fixes

- NVIDIA/cub#277: Fixed sanitizer warnings in `RadixSortScanBinsKernels`. Thanks
  to Andy Adinets (@canonizer) for this contribution.
- NVIDIA/cub#287: `cub::DeviceHistogram` now correctly handles cases
  where `OffsetT` is not an `int`. Thanks to Dominique LaSalle (@nv-dlasalle)
  for this contribution.
- NVIDIA/cub#311: Fixed several bugs and added tests for the `cub::BlockShuffle`
  collective operations.
- NVIDIA/cub#312: Eliminate unnecessary kernel instantiations when
  compiling `cub::DeviceScan`. Thanks to Elias Stehle (@elstehle) for this
  contribution.
- NVIDIA/cub#319: Fixed out-of-bounds memory access on debugging builds
  of `cub::BlockDiscontinuity::FlagHeadsAndTails`.
- NVIDIA/cub#320: Fixed harmless missing return statement warning in
  unreachable `cub::TexObjInputIterator` code path.

## Other Enhancements

- Several documentation fixes are included in this release.
    - NVIDIA/cub#275: Fixed comments describing the `cub::If` and `cub::Equals`
      utilities. Thanks to Rukshan Jayasekara (@rukshan99) for this
      contribution.
    - NVIDIA/cub#290: Documented that `cub::DeviceSegmentedReduce` will produce
      consistent results run-to-run on the same device for pseudo-associated
      reduction operators. Thanks to Himanshu (@himanshu007-creator) for this
      contribution.
    - NVIDIA/cub#298: `CONTRIBUTING.md` now refers to Thrust's build
      instructions for developer builds, which is the preferred way to build the
      CUB test harness. Thanks to Xiang Gao (@zasdfgbnm) for contributing.
    - NVIDIA/cub#301: Expand `cub::DeviceScan` documentation to include in-place
      support and add tests. Thanks to Xiang Gao (@zasdfgbnm) for this
      contribution.
    - NVIDIA/cub#307: Expand `cub::DeviceRadixSort` and `cub::BlockRadixSort`
      documentation to clarify stability, in-place support, and type-specific
      bitwise transformations. Thanks to Himanshu (@himanshu007-creator) for
      contributing.
    - NVIDIA/cub#316: Move `WARP_TIME_SLICING` documentation to the correct
      location. Thanks to Peter Han (@peter9606) for this contribution.
    - NVIDIA/cub#321: Update URLs from deprecated github.com to preferred
      github.io. Thanks to Lilo Huang (@lilohuang) for this contribution.

# CUB 1.12.1 (CUDA Toolkit 11.4)

CUB 1.12.1 is a trivial patch release that slightly changes the phrasing of
a deprecation message.

# CUB 1.12.0 (NVIDIA HPC SDK 21.3)

## Summary

CUB 1.12.0 is a bugfix release accompanying the NVIDIA HPC SDK 21.3 and
the CUDA Toolkit 11.4.

Radix sort is now stable when both +0.0 and -0.0 are present in the input (they
are treated as equivalent).
Many compilation warnings and subtle overflow bugs were fixed in the device
algorithms, including a long-standing bug that returned invalid temporary
storage requirements when `num_items` was close to (but not
exceeding) `INT32_MAX`.
Support for Clang < 7.0 and MSVC < 2019 (aka 19.20/16.0/14.20) is now
deprecated.

## Breaking Changes

- NVIDIA/cub#256: Deprecate Clang < 7 and MSVC < 2019.

## New Features

- NVIDIA/cub#218: Radix sort now treats -0.0 and +0.0 as equivalent for floating
  point types, which is required for the sort to be stable. Thanks to Andy
  Adinets for this contribution.

## Bug Fixes

- NVIDIA/cub#247: Suppress newly triggered warnings in Clang. Thanks to Andrew
  Corrigan for this contribution.
- NVIDIA/cub#249: Enable stricter warning flags. This fixes a number of
  outstanding issues:
  - NVIDIA/cub#221: Overflow in `temp_storage_bytes` when `num_items` close to
    (but not over) `INT32_MAX`.
  - NVIDIA/cub#228: CUB uses non-standard C++ extensions that break strict
    compilers.
  - NVIDIA/cub#257: Warning when compiling `GridEvenShare` with unsigned
    offsets.
- NVIDIA/cub#258: Use correct `OffsetT` in `DispatchRadixSort::InitPassConfig`.
  Thanks to Felix Kallenborn for this contribution.
- NVIDIA/cub#259: Remove some problematic `__forceinline__` annotations.

## Other Enhancements

- NVIDIA/cub#123: Fix incorrect issue number in changelog. Thanks to Peet
  Whittaker for this contribution.

# CUB 1.11.0 (CUDA Toolkit 11.3)

## Summary

CUB 1.11.0 is a major release accompanying the CUDA Toolkit 11.3 release,
providing bugfixes and performance enhancements.

It includes a new `DeviceRadixSort` backend that improves performance by up to
2x on supported keys and hardware.

Our CMake package and build system continue to see improvements
with `add_subdirectory` support, installation rules, status messages, and other
features that make CUB easier to use from CMake projects.

The release includes several other bugfixes and modernizations, and received
updates from 11 contributors.

## Breaking Changes

- NVIDIA/cub#201: The intermediate accumulator type used when `DeviceScan` is
  invoked with different input/output types is now consistent
  with [P0571](https://wg21.link/P0571). This may produce different results for
  some edge cases when compared with earlier releases of CUB.

## New Features

- NVIDIA/cub#204: Faster `DeviceRadixSort`, up to 2x performance increase for
  32/64-bit keys on Pascal and up (SM60+). Thanks to Andy Adinets for this
  contribution.
- Unroll loops in `BlockRadixRank` to improve performance for 32-bit keys by
  1.5-2x on Clang CUDA. Thanks to Justin Lebar for this contribution.
- NVIDIA/cub#200: Allow CUB to be added to CMake projects via `add_subdirectory`.
- NVIDIA/cub#214: Optionally add install rules when included with
  CMake's `add_subdirectory`. Thanks to Kai Germaschewski for this contribution.

## Bug Fixes

- NVIDIA/cub#215: Fix integer truncation in `AgentReduceByKey`, `AgentScan`,
  and `AgentSegmentFixup`. Thanks to Rory Mitchell for this contribution.
- NVIDIA/cub#225: Fix compile-time regression when defining `CUB_NS_PREFIX`
  /`CUB_NS_POSTFIX` macro. Thanks to Elias Stehle for this contribution.
- NVIDIA/cub#210: Fix some edge cases in `DeviceScan`:
  - Use values from the input when padding temporary buffers. This prevents
    custom functors from getting unexpected values.
  - Prevent integer truncation when using large indices via the `DispatchScan`
    layer.
  - Use timesliced reads/writes for types > 128 bytes.
- NVIDIA/cub#217: Fix and add test for cmake package install rules. Thanks to
  Keith Kraus and Kai Germaschewski for testing and discussion.
- NVIDIA/cub#170, NVIDIA/cub#233: Update CUDA version checks to behave on Clang
  CUDA and `nvc++`. Thanks to Artem Belevich, Andrew Corrigan, and David Olsen
  for these contributions.
- NVIDIA/cub#220, NVIDIA/cub#216: Various fixes for Clang CUDA. Thanks to Andrew
  Corrigan for these contributions.
- NVIDIA/cub#231: Fix signedness mismatch warnings in unit tests.
- NVIDIA/cub#231: Suppress GPU deprecation warnings.
- NVIDIA/cub#214: Use semantic versioning rules for our CMake package's
  compatibility checks. Thanks to Kai Germaschewski for this contribution.
- NVIDIA/cub#214: Use `FindPackageHandleStandardArgs` to print standard status
  messages when our CMake package is found. Thanks to Kai Germaschewski for this
  contribution.
- NVIDIA/cub#207: Fix `CubDebug` usage
  in `CachingDeviceAllocator::DeviceAllocate`. Thanks to Andreas Hehn for this
  contribution.
- Fix documentation for `DevicePartition`. Thanks to ByteHamster for this
  contribution.
- Clean up unused code in `DispatchScan`. Thanks to ByteHamster for this
  contribution.

## Other Enhancements

- NVIDIA/cub#213: Remove tuning policies for unsupported hardware (<SM35).
- References to the old Github repository and branch names were updated.
  - Github's `thrust/cub` repository is now `NVIDIA/cub`
  - Development has moved from the `master` branch to the `main` branch.

# CUB 1.10.0 (NVIDIA HPC SDK 20.9, CUDA Toolkit 11.2)

## Summary

CUB 1.10.0 is the major release accompanying the NVIDIA HPC SDK 20.9 release
  and the CUDA Toolkit 11.2 release.
It drops support for C++03, GCC < 5, Clang < 6, and MSVC < 2017.
It also overhauls CMake support.
Finally, we now have a Code of Conduct for contributors:
https://github.com/NVIDIA/cub/blob/main/CODE_OF_CONDUCT.md

## Breaking Changes

- C++03 is no longer supported.
- GCC < 5, Clang < 6, and MSVC < 2017 are no longer supported.
- C++11 is deprecated.
  Using this dialect will generate a compile-time warning.
  These warnings can be suppressed by defining
    `CUB_IGNORE_DEPRECATED_CPP_DIALECT` or `CUB_IGNORE_DEPRECATED_CPP_11`.
  Suppression is only a short term solution.
  We will be dropping support for C++11 in the near future.
- CMake < 3.15 is no longer supported.
- The default branch on GitHub is now called `main`.

## Other Enhancements

- Added install targets to CMake builds.
- C++17 support.

## Bug Fixes

- NVIDIA/thrust#1244: Check for macro collisions with system headers during
    header testing.
- NVIDIA/thrust#1153: Switch to placement new instead of assignment to
    construct items in uninitialized memory.
  Thanks to Hugh Winkler for this contribution.
- NVIDIA/cub#38: Fix `cub::DeviceHistogram` for `size_t` `OffsetT`s.
  Thanks to Leo Fang for this contribution.
- NVIDIA/cub#35: Fix GCC-5 maybe-uninitialized warning.
  Thanks to Rong Ou for this contribution.
- NVIDIA/cub#36: Qualify namespace for `va_printf` in `_CubLog`.
  Thanks to Andrei Tchouprakov for this contribution.

# CUB 1.9.10-1 (NVIDIA HPC SDK 20.7, CUDA Toolkit 11.1)

## Summary

CUB 1.9.10-1 is the minor release accompanying the NVIDIA HPC SDK 20.7 release
  and the CUDA Toolkit 11.1 release.

## Bug Fixes

- NVIDIA/thrust#1217: Move static local in cub::DeviceCount to a separate
  host-only function because NVC++ doesn't support static locals in host-device
  functions.

# CUB 1.9.10 (NVIDIA HPC SDK 20.5)

## Summary

Thrust 1.9.10 is the release accompanying the NVIDIA HPC SDK 20.5 release.
It adds CMake `find_package` support.
C++03, C++11, GCC < 5, Clang < 6, and MSVC < 2017 are now deprecated.
Starting with the upcoming 1.10.0 release, C++03 support will be dropped
  entirely.

## Breaking Changes

- Thrust now checks that it is compatible with the version of CUB found
    in your include path, generating an error if it is not.
  If you are using your own version of CUB, it may be too old.
  It is recommended to simply delete your own version of CUB and use the
    version of CUB that comes with Thrust.
- C++03 and C++11 are deprecated.
  Using these dialects will generate a compile-time warning.
  These warnings can be suppressed by defining
    `CUB_IGNORE_DEPRECATED_CPP_DIALECT` (to suppress C++03 and C++11
    deprecation warnings) or `CUB_IGNORE_DEPRECATED_CPP_11` (to suppress C++11
    deprecation warnings).
  Suppression is only a short term solution.
  We will be dropping support for C++03 in the 1.10.0 release and C++11 in the
    near future.
- GCC < 5, Clang < 6, and MSVC < 2017 are deprecated.
  Using these compilers will generate a compile-time warning.
  These warnings can be suppressed by defining
  `CUB_IGNORE_DEPRECATED_COMPILER`.
  Suppression is only a short term solution.
  We will be dropping support for these compilers in the near future.

## New Features

- CMake `find_package` support.
  Just point CMake at the `cmake` folder in your CUB include directory
    (ex: `cmake -DCUB_DIR=/usr/local/cuda/include/cub/cmake/ .`) and then you
    can add CUB to your CMake project with `find_package(CUB REQUIRED CONFIG)`.

# CUB 1.9.9 (CUDA 11.0)

## Summary

CUB 1.9.9 is the release accompanying the CUDA Toolkit 11.0 release.
It introduces CMake support, version macros, platform detection machinery,
  and support for NVC++, which uses Thrust (and thus CUB) to implement
  GPU-accelerated C++17 Parallel Algorithms.
Additionally, the scan dispatch layer was refactored and modernized.
C++03, C++11, GCC < 5, Clang < 6, and MSVC < 2017 are now deprecated.
Starting with the upcoming 1.10.0 release, C++03 support will be dropped
  entirely.

## Breaking Changes

- Thrust now checks that it is compatible with the version of CUB found
    in your include path, generating an error if it is not.
  If you are using your own version of CUB, it may be too old.
  It is recommended to simply delete your own version of CUB and use the
    version of CUB that comes with Thrust.
- C++03 and C++11 are deprecated.
  Using these dialects will generate a compile-time warning.
  These warnings can be suppressed by defining
    `CUB_IGNORE_DEPRECATED_CPP_DIALECT` (to suppress C++03 and C++11
    deprecation warnings) or `CUB_IGNORE_DEPRECATED_CPP11` (to suppress C++11
    deprecation warnings).
  Suppression is only a short term solution.
  We will be dropping support for C++03 in the 1.10.0 release and C++11 in the
    near future.
- GCC < 5, Clang < 6, and MSVC < 2017 are deprecated.
  Using these compilers will generate a compile-time warning.
  These warnings can be suppressed by defining
    `CUB_IGNORE_DEPRECATED_COMPILER`.
  Suppression is only a short term solution.
  We will be dropping support for these compilers in the near future.

## New Features

- CMake support.
  Thanks to Francis Lemaire for this contribution.
- Refactorized and modernized scan dispatch layer.
  Thanks to Francis Lemaire for this contribution.
- Policy hooks for device-wide reduce, scan, and radix sort facilities
    to simplify tuning and allow users to provide custom policies.
  Thanks to Francis Lemaire for this contribution.
- `<cub/version.cuh>`: `CUB_VERSION`, `CUB_VERSION_MAJOR`, `CUB_VERSION_MINOR`,
    `CUB_VERSION_SUBMINOR`, and `CUB_PATCH_NUMBER`.
- Platform detection machinery:
  - `<cub/util_cpp_dialect.cuh>`: Detects the C++ standard dialect.
  - `<cub/util_compiler.cuh>`: host and device compiler detection.
  - `<cub/util_deprecated.cuh>`: `CUB_DEPRECATED`.
  - <cub/config.cuh>`: Includes `<cub/util_arch.cuh>`,
      `<cub/util_compiler.cuh>`, `<cub/util_cpp_dialect.cuh>`,
      `<cub/util_deprecated.cuh>`, `<cub/util_macro.cuh>`,
      `<cub/util_namespace.cuh>`
- `cub::DeviceCount` and `cub::DeviceCountUncached`, caching abstractions for
    `cudaGetDeviceCount`.

## Other Enhancements

- Lazily initialize the per-device CUDAattribute caches, because CUDA context
    creation is expensive and adds up with large CUDA binaries on machines with
    many GPUs.
  Thanks to the NVIDIA PyTorch team for bringing this to our attention.
- Make `cub::SwitchDevice` avoid setting/resetting the device if the current
    device is the same as the target device.

## Bug Fixes

- Add explicit failure parameter to CAS in the CUB attribute cache to workaround
    a GCC 4.8 bug.
- Revert a change in reductions that changed the signedness of the `lane_id`
    variable to suppress a warning, as this introduces a bug in optimized device
    code.
- Fix initialization in `cub::ExclusiveSum`.
  Thanks to Conor Hoekstra for this contribution.
- Fix initialization of the `std::array` in the CUB attribute cache.
- Fix `-Wsign-compare` warnings.
  Thanks to Elias Stehle for this contribution.
- Fix `test_block_reduce.cu` to build without parameters.
  Thanks to Francis Lemaire for this contribution.
- Add missing includes to `grid_even_share.cuh`.
  Thanks to Francis Lemaire for this contribution.
- Add missing includes to `thread_search.cuh`.
  Thanks to Francis Lemaire for this contribution.
- Add missing includes to `cub.cuh`.
  Thanks to Felix Kallenborn for this contribution.

# CUB 1.9.8-1 (NVIDIA HPC SDK 20.3)

## Summary

CUB 1.9.8-1 is a variant of 1.9.8 accompanying the NVIDIA HPC SDK 20.3 release.
It contains modifications necessary to serve as the implementation of NVC++'s
  GPU-accelerated C++17 Parallel Algorithms.

# CUB 1.9.8 (CUDA 11.0 Early Access)

## Summary

CUB 1.9.8 is the first release of CUB to be officially supported and included
  in the CUDA Toolkit.
When compiling CUB in C++11 mode, CUB now caches calls to CUDA attribute query
  APIs, which improves performance of these queries by 20x to 50x when they
  are called concurrently by multiple host threads.

## Enhancements

- (C++11 or later) Cache calls to `cudaFuncGetAttributes` and
    `cudaDeviceGetAttribute` within `cub::PtxVersion` and `cub::SmVersion`.
    These CUDA APIs acquire locks to CUDA driver/runtime mutex and perform
    poorly under contention; with the caching, they are 20 to 50x faster when
    called concurrently.
  Thanks to Bilge Acun for bringing this issue to our attention.
- `DispatchReduce` now takes an `OutputT` template parameter so that users can
    specify the intermediate type explicitly.
- Radix sort tuning policies updates to fix performance issues for element
    types smaller than 4 bytes.

## Bug Fixes

- Change initialization style from copy initialization to direct initialization
    (which is more permissive) in `AgentReduce` to allow a wider range of types
    to be used with it.
- Fix bad signed/unsigned comparisons in `WarpReduce`.
- Fix computation of valid lanes in warp-level reduction primitive to correctly
    handle the case where there are 0 input items per warp.

# CUB 1.8.0

## Summary

CUB 1.8.0 introduces changes to the `cub::Shuffle*` interfaces.

## Breaking Changes

- The interfaces of `cub::ShuffleIndex`, `cub::ShuffleUp`, and
    `cub::ShuffleDown` have been changed to allow for better computation of the
    PTX SHFL control constant for logical warps smaller than 32 threads.

## Bug Fixes

- #112: Fix `cub::WarpScan`'s broadcast of warp-wide aggregate for logical
    warps smaller than 32 threads.

# CUB 1.7.5

## Summary

CUB 1.7.5 adds support for radix sorting `__half` keys and improved sorting
  performance for 1 byte keys.
It was incorporated into Thrust 1.9.2.

## Enhancements

- Radix sort support for `__half` keys.
- Radix sort tuning policy updates to improve 1 byte key performance.

## Bug Fixes

- Syntax tweaks to mollify Clang.
- #127: `cub::DeviceRunLengthEncode::Encode` returns incorrect results.
- #128: 7-bit sorting passes fail for SM61 with large values.

# CUB 1.7.4

## Summary

CUB 1.7.4 is a minor release that was incorporated into Thrust 1.9.1-2.

## Bug Fixes

- #114: Can't pair non-trivially-constructible values in radix sort.
- #115: `cub::WarpReduce` segmented reduction is broken in CUDA 9 for logical
    warp sizes smaller than 32.

# CUB 1.7.3

## Summary

CUB 1.7.3 is a minor release.

## Bug Fixes

- #110: `cub::DeviceHistogram` null-pointer exception bug for iterator inputs.

# CUB 1.7.2

## Summary

CUB 1.7.2 is a minor release.

## Bug Fixes

- #108: Device-wide reduction is now "run-to-run" deterministic for
    pseudo-associative reduction operators (like floating point addition).

# CUB 1.7.1

## Summary

CUB 1.7.1 delivers improved radix sort performance on SM7x (Volta) GPUs and a
  number of bug fixes.

## Enhancements

- Radix sort tuning policies updated for SM7x (Volta).

## Bug Fixes

- #104: `uint64_t` `cub::WarpReduce` broken for CUB 1.7.0 on CUDA 8 and older.
- #103: Can't mix Thrust from CUDA 9.0 and CUB.
- #102: CUB pulls in `windows.h` which defines `min`/`max` macros that conflict
    with `std::min`/`std::max`.
- #99: Radix sorting crashes NVCC on Windows 10 for SM52.
- #98: cuda-memcheck: --tool initcheck failed with lineOfSight.
- #94: Git clone size.
- #93: Accept iterators for segment offsets.
- #87: CUB uses anonymous unions which is not valid C++.
- #44: Check for C++11 is incorrect for Visual Studio 2013.

# CUB 1.7.0

## Summary

CUB 1.7.0 brings support for CUDA 9.0 and SM7x (Volta) GPUs.
It is compatible with independent thread scheduling.
It was incorporated into Thrust 1.9.0-5.

## Breaking Changes

- Remove `cub::WarpAll` and `cub::WarpAny`.
  These functions served to emulate `__all` and `__any` functionality for
    SM1x devices, which did not have those operations.
  However, SM1x devices are now deprecated in CUDA, and the interfaces of these
    two functions are now lacking the lane-mask needed for collectives to run on
    SM7x and newer GPUs which have independent thread scheduling.

## Other Enhancements

- Remove any assumptions of implicit warp synchronization to be compatible with
    SM7x's (Volta) independent thread scheduling.

## Bug Fixes

- #86: Incorrect results with reduce-by-key.

# CUB 1.6.4

## Summary

CUB 1.6.4 improves radix sorting performance for SM5x (Maxwell) and SM6x
  (Pascal) GPUs.

## Enhancements

- Radix sort tuning policies updated for SM5x (Maxwell) and SM6x (Pascal) -
    3.5B and 3.4B 32 byte keys/s on TitanX and GTX 1080, respectively.

## Bug Fixes

- Restore fence work-around for scan (reduce-by-key, etc.) hangs in CUDA 8.5.
- #65: `cub::DeviceSegmentedRadixSort` should allow inputs to have
    pointer-to-const type.
- Mollify Clang device-side warnings.
- Remove out-dated MSVC project files.

# CUB 1.6.3

## Summary

CUB 1.6.3 improves support for Windows, changes
  `cub::BlockLoad`/`cub::BlockStore` interface to take the local data type,
  and enhances radix sort performance for SM6x (Pascal) GPUs.

## Breaking Changes

- `cub::BlockLoad` and `cub::BlockStore` are now templated by the local data
    type, instead of the `Iterator` type.
  This allows for output iterators having `void` as their `value_type` (e.g.
    discard iterators).

## Other Enhancements

- Radix sort tuning policies updated for SM6x (Pascal) GPUs - 6.2B 4 byte
    keys/s on GP100.
- Improved support for Windows (warnings, alignment, etc).

## Bug Fixes

- #74: `cub::WarpReduce` executes reduction operator for out-of-bounds items.
- #72: `cub:InequalityWrapper::operator` should be non-const.
- #71: `cub::KeyValuePair` won't work if `Key` has non-trivial constructor.
- #69: cub::BlockStore::Store` doesn't compile if `OutputIteratorT::value_type`
    isn't `T`.
- #68: `cub::TilePrefixCallbackOp::WarpReduce` doesn't permit PTX arch
    specialization.

# CUB 1.6.2 (previously 1.5.5)

## Summary

CUB 1.6.2 (previously 1.5.5) improves radix sort performance for SM6x (Pascal)
  GPUs.

## Enhancements

- Radix sort tuning policies updated for SM6x (Pascal) GPUs.

## Bug Fixes

- Fix AArch64 compilation of `cub::CachingDeviceAllocator`.

# CUB 1.6.1 (previously 1.5.4)

## Summary

CUB 1.6.1 (previously 1.5.4) is a minor release.

## Bug Fixes

- Fix radix sorting bug introduced by scan refactorization.

# CUB 1.6.0 (previously 1.5.3)

## Summary

CUB 1.6.0 changes the scan and reduce interfaces.
Exclusive scans now accept an "initial value" instead of an "identity value".
Scans and reductions now support differing input and output sequence types.
Additionally, many bugs have been fixed.

## Breaking Changes

- Device/block/warp-wide exclusive scans have been revised to now accept an
    "initial value" (instead of an "identity value") for seeding the computation
    with an arbitrary prefix.
- Device-wide reductions and scans can now have input sequence types that are
    different from output sequence types (as long as they are convertible).

## Other Enhancements

- Reduce repository size by moving the doxygen binary to doc repository.
- Minor reduction in `cub::BlockScan` instruction counts.

## Bug Fixes

- Issue #55: Warning in `cub/device/dispatch/dispatch_reduce_by_key.cuh`.
- Issue #59: `cub::DeviceScan::ExclusiveSum` can't prefix sum of float into
    double.
- Issue #58: Infinite loop in `cub::CachingDeviceAllocator::NearestPowerOf`.
- Issue #47: `cub::CachingDeviceAllocator` needs to clean up CUDA global error
    state upon successful retry.
- Issue #46: Very high amount of needed memory from the
    `cub::DeviceHistogram::HistogramEven`.
- Issue #45: `cub::CachingDeviceAllocator` fails with debug output enabled

# CUB 1.5.2

## Summary

CUB 1.5.2 enhances `cub::CachingDeviceAllocator` and improves scan performance
  for SM5x (Maxwell).

## Enhancements

- Improved medium-size scan performance on SM5x (Maxwell).
- Refactored `cub::CachingDeviceAllocator`:
  - Now spends less time locked.
  - Uses C++11's `std::mutex` when available.
  - Failure to allocate a block from the runtime will retry once after
  		freeing cached allocations.
  - Now respects max-bin, fixing an issue where blocks in excess of max-bin
      were still being retained in the free cache.

## Bug fixes:

- Fix for generic-type reduce-by-key `cub::WarpScan` for SM3x and newer GPUs.

# CUB 1.5.1

## Summary

CUB 1.5.1 is a minor release.

## Bug Fixes

- Fix for incorrect `cub::DeviceRadixSort` output for some small problems on
    SM52 (Mawell) GPUs.
- Fix for macro redefinition warnings when compiling `thrust::sort`.

# CUB 1.5.0

CUB 1.5.0 introduces segmented sort and reduction primitives.

## New Features:

- Segmented device-wide operations for device-wide sort and reduction primitives.

## Bug Fixes:

- #36: `cub::ThreadLoad` generates compiler errors when loading from
    pointer-to-const.
- #29: `cub::DeviceRadixSort::SortKeys<bool>` yields compiler errors.
- #26: Misaligned address after `cub::DeviceRadixSort::SortKeys`.
- #25: Fix for incorrect results and crashes when radix sorting 0-length
    problems.
- Fix CUDA 7.5 issues on SM52 GPUs with SHFL-based warp-scan and
    warp-reduction on non-primitive data types (e.g. user-defined structs).
- Fix small radix sorting problems where 0 temporary bytes were required and
    users code was invoking `malloc(0)` on some systems where that returns
    `NULL`.
  CUB assumed the user was asking for the size again and not running the sort.

# CUB 1.4.1

## Summary

CUB 1.4.1 is a minor release.

## Enhancements

- Allow `cub::DeviceRadixSort` and `cub::BlockRadixSort` on bool types.

## Bug Fixes

- Fix minor CUDA 7.0 performance regressions in `cub::DeviceScan` and
    `cub::DeviceReduceByKey`.
- Remove requirement for callers to define the `CUB_CDP` macro
    when invoking CUB device-wide rountines using CUDA dynamic parallelism.
- Fix headers not being included in the proper order (or missing includes)
    for some block-wide functions.

# CUB 1.4.0

## Summary

CUB 1.4.0 adds `cub::DeviceSpmv`, `cub::DeviceRunLength::NonTrivialRuns`,
  improves `cub::DeviceHistogram`, and introduces support for SM5x (Maxwell)
  GPUs.

## New Features:

- `cub::DeviceSpmv` methods for multiplying sparse matrices by
    dense vectors, load-balanced using a merge-based parallel decomposition.
- `cub::DeviceRadixSort` sorting entry-points that always return
    the sorted output into the specified buffer, as opposed to the
    `cub::DoubleBuffer` in which it could end up in either buffer.
- `cub::DeviceRunLengthEncode::NonTrivialRuns` for finding the starting
    offsets and lengths of all non-trivial runs (i.e., length > 1) of keys in
    a given sequence.
  Useful for top-down partitioning algorithms like MSD sorting of very-large
    keys.

## Other Enhancements

- Support and performance tuning for SM5x (Maxwell) GPUs.
- Updated cub::DeviceHistogram implementation that provides the same
    "histogram-even" and "histogram-range" functionality as IPP/NPP.
  Provides extremely fast and, perhaps more importantly, very uniform
    performance response across diverse real-world datasets, including
    pathological (homogeneous) sample distributions.

# CUB 1.3.2

## Summary

CUB 1.3.2 is a minor release.

## Bug Fixes

- Fix `cub::DeviceReduce` where reductions of small problems (small enough to
    only dispatch a single thread block) would run in the default stream (stream
    zero) regardless of whether an alternate stream was specified.

# CUB 1.3.1

## Summary

CUB 1.3.1 is a minor release.

## Bug Fixes

- Workaround for a benign WAW race warning reported by cuda-memcheck
    in `cub::BlockScan` specialized for `BLOCK_SCAN_WARP_SCANS` algorithm.
- Fix bug in `cub::DeviceRadixSort` where the algorithm may sort more
    key bits than the caller specified (up to the nearest radix digit).
- Fix for ~3% `cub::DeviceRadixSort` performance regression on SM2x (Fermi) and
    SM3x (Kepler) GPUs.

# CUB 1.3.0

## Summary

CUB 1.3.0 improves how thread blocks are expressed in block- and warp-wide
  primitives and adds an enhanced version of `cub::WarpScan`.

## Breaking Changes

- CUB's collective (block-wide, warp-wide) primitives underwent a minor
    interface refactoring:
  - To provide the appropriate support for multidimensional thread blocks,
      The interfaces for collective classes are now template-parameterized by
      X, Y, and Z block dimensions (with `BLOCK_DIM_Y` and `BLOCK_DIM_Z` being
      optional, and `BLOCK_DIM_X` replacing `BLOCK_THREADS`).
    Furthermore, the constructors that accept remapped linear
      thread-identifiers have been removed: all primitives now assume a
      row-major thread-ranking for multidimensional thread blocks.
  - To allow the host program (compiled by the host-pass) to accurately
      determine the device-specific storage requirements for a given collective
      (compiled for each device-pass), the interfaces for collective classes
      are now (optionally) template-parameterized by the desired PTX compute
      capability.
    This is useful when aliasing collective storage to shared memory that has
      been allocated dynamically by the host at the kernel call site.
  - Most CUB programs having typical 1D usage should not require any
      changes to accomodate these updates.

## New Features

- Added "combination" `cub::WarpScan` methods for efficiently computing
    both inclusive and exclusive prefix scans (and sums).

## Bug Fixes

- Fix for bug in `cub::WarpScan` (which affected `cub::BlockScan` and
    `cub::DeviceScan`) where incorrect results (e.g., NAN) would often be
    returned when parameterized for floating-point types (fp32, fp64).
- Workaround for ptxas error when compiling with with -G flag on Linux (for
    debug instrumentation).
- Fixes for certain scan scenarios using custom scan operators where code
    compiled for SM1x is run on newer GPUs of higher compute-capability: the
    compiler could not tell which memory space was being used collective
    operations and was mistakenly using global ops instead of shared ops.

# CUB 1.2.3

## Summary

CUB 1.2.3 is a minor release.

## Bug Fixes

- Fixed access violation bug in `cub::DeviceReduce::ReduceByKey` for
    non-primitive value types.
- Fixed code-snippet bug in `ArgIndexInputIteratorT` documentation.

# CUB 1.2.2

## Summary

CUB 1.2.2 adds a new variant of `cub::BlockReduce` and MSVC project solections
  for examples.

## New Features

- MSVC project solutions for device-wide and block-wide examples
- New algorithmic variant of cub::BlockReduce for improved performance
    when using commutative operators (e.g., numeric addition).

## Bug Fixes

- Inclusion of Thrust headers in a certain order prevented CUB device-wide
    primitives from working properly.

# CUB 1.2.0

## Summary

CUB 1.2.0 adds `cub::DeviceReduce::ReduceByKey` and
  `cub::DeviceReduce::RunLengthEncode` and support for CUDA 6.0.

## New Features

- `cub::DeviceReduce::ReduceByKey`.
- `cub::DeviceReduce::RunLengthEncode`.

## Other Enhancements

- Improved `cub::DeviceScan`, `cub::DeviceSelect`, `cub::DevicePartition`
    performance.
- Documentation and testing:
  - Added performance-portability plots for many device-wide primitives.
  - Explain that iterator (in)compatibilities with CUDA 5.0 (and older) and
      Thrust 1.6 (and older).
- Revised the operation of temporary tile status bookkeeping for
    `cub::DeviceScan` (and similar) to be safe for current code run on future
    platforms (now uses proper fences).

## Bug Fixes

- Fix `cub::DeviceScan` bug where Windows alignment disagreements between host
    and device regarding user-defined data types would corrupt tile status.
- Fix `cub::BlockScan` bug where certain exclusive scans on custom data types
    for the `BLOCK_SCAN_WARP_SCANS` variant would return incorrect results for
    the first thread in the block.
- Added workaround to make `cub::TexRefInputIteratorT` work with CUDA 6.0.

# CUB 1.1.1

## Summary

CUB 1.1.1 introduces texture and cache modifier iterators, descending sorting,
  `cub::DeviceSelect`, `cub::DevicePartition`, `cub::Shuffle*`, and
  `cub::MaxSMOccupancy`.
Additionally, scan and sort performance for older GPUs has been improved and
  many bugs have been fixed.

## Breaking Changes

- Refactored block-wide I/O (`cub::BlockLoad` and `cub::BlockStore`), removing
    cache-modifiers from their interfaces.
  `cub::CacheModifiedInputIterator` and `cub::CacheModifiedOutputIterator`
    should now be used with `cub::BlockLoad` and `cub::BlockStore` to effect that
    behavior.

## New Features

- `cub::TexObjInputIterator`, `cub::TexRefInputIterator`,
    `cub::CacheModifiedInputIterator`, and `cub::CacheModifiedOutputIterator`
    types for loading & storing arbitrary types through the cache hierarchy.
  They are compatible with Thrust.
- Descending sorting for `cub::DeviceRadixSort` and `cub::BlockRadixSort`.
- Min, max, arg-min, and arg-max operators for `cub::DeviceReduce`.
- `cub::DeviceSelect` (select-unique, select-if, and select-flagged).
- `cub::DevicePartition` (partition-if, partition-flagged).
- Generic `cub::ShuffleUp`, `cub::ShuffleDown`, and `cub::ShuffleIndex` for
    warp-wide communication of arbitrary data types (SM3x and up).
- `cub::MaxSmOccupancy` for accurately determining SM occupancy for any given
    kernel function pointer.

## Other Enhancements

- Improved `cub::DeviceScan` and `cub::DeviceRadixSort` performance for older
    GPUs (SM1x to SM3x).
- Renamed device-wide `stream_synchronous` param to `debug_synchronous` to
    avoid confusion about usage.
- Documentation improvements:
  - Added simple examples of device-wide methods.
  - Improved doxygen documentation and example snippets.
- Improved test coverege to include up to 21,000 kernel variants and 851,000
    unit tests (per architecture, per platform).

## Bug Fixes

- Fix misc `cub::DeviceScan, BlockScan, DeviceReduce, and BlockReduce bugs when
    operating on non-primitive types for older architectures SM1x.
- SHFL-based scans and reductions produced incorrect results for multi-word
    types (size > 4B) on Linux.
- For `cub::WarpScan`-based scans, not all threads in the first warp were
    entering the prefix callback functor.
- `cub::DeviceRadixSort` had a race condition with key-value pairs for pre-SM35
    architectures.
- `cub::DeviceRadixSor` bitfield-extract behavior with long keys on 64-bit
    Linux was incorrect.
- `cub::BlockDiscontinuity` failed to compile for types other than
    `int32_t`/`uint32_t`.
- CUDA Dynamic Parallelism (CDP, e.g. device-callable) versions of device-wide
    methods now report the same temporary storage allocation size requirement as
    their host-callable counterparts.

# CUB 1.0.2

## Summary

CUB 1.0.2 is a minor release.

## Bug Fixes

- Corrections to code snippet examples for `cub::BlockLoad`, `cub::BlockStore`,
    and `cub::BlockDiscontinuity`.
- Cleaned up unnecessary/missing header includes.
  You can now safely include a specific .cuh (instead of `cub.cuh`).
- Bug/compilation fixes for `cub::BlockHistogram`.

# CUB 1.0.1

## Summary

CUB 1.0.1 adds `cub::DeviceRadixSort` and `cub::DeviceScan`.
Numerous other performance and correctness fixes and included.

## Breaking Changes

- New collective interface idiom (specialize/construct/invoke).

## New Features

- `cub::DeviceRadixSort`.
  Implements short-circuiting for homogenous digit passes.
- `cub::DeviceScan`.
  Implements single-pass "adaptive-lookback" strategy.

## Other Enhancements

- Significantly improved documentation (with example code snippets).
- More extensive regression test suit for aggressively testing collective
    variants.
- Allow non-trially-constructed types (previously unions had prevented aliasing
    temporary storage of those types).
- Improved support for SM3x SHFL (collective ops now use SHFL for types larger
    than 32 bits).
- Better code generation for 64-bit addressing within
    `cub::BlockLoad`/`cub::BlockStore`.
- `cub::DeviceHistogram` now supports histograms of arbitrary bins.
- Updates to accommodate CUDA 5.5 dynamic parallelism.

## Bug Fixes

- Workarounds for SM10 codegen issues in uncommonly-used
    `cub::WarpScan`/`cub::WarpReduce` specializations.

# CUB 0.9.4

## Summary

CUB 0.9.3 is a minor release.

## Enhancements

- Various documentation updates and corrections.

## Bug Fixes

- Fixed compilation errors for SM1x.
- Fixed compilation errors for some WarpScan entrypoints on SM3x and up.

# CUB 0.9.3

## Summary

CUB 0.9.3 adds histogram algorithms and work management utility descriptors.

## New Features

- `cub::DevicHistogram256`.
- `cub::BlockHistogram256`.
- `cub::BlockScan` algorithm variant `BLOCK_SCAN_RAKING_MEMOIZE`, which
    trades more register consumption for less shared memory I/O.
- `cub::GridQueue`, `cub::GridEvenShare`, work management utility descriptors.

## Other Enhancements

- Updates to `cub::BlockRadixRank` to use `cub::BlockScan`, which improves
    performance on SM3x by using SHFL.
- Allow types other than builtin types to be used in `cub::WarpScan::*Sum`
    methods if they only have `operator+` overloaded.
  Previously they also required to support assignment from `int(0)`.
- Update `cub::BlockReduce`'s `BLOCK_REDUCE_WARP_REDUCTIONS` algorithm to work
    even when block size is not an even multiple of warp size.
- Refactoring of `cub::DeviceAllocator` interface and
    `cub::CachingDeviceAllocator` implementation.

# CUB 0.9.2

## Summary

CUB 0.9.2 adds `cub::WarpReduce`.

## New Features

- `cub::WarpReduce`, which uses the SHFL instruction when applicable.
  `cub::BlockReduce` now uses this `cub::WarpReduce` instead of implementing
    its own.

## Enhancements

- Documentation updates and corrections.

## Bug Fixes

- Fixes for 64-bit Linux compilation warnings and errors.

# CUB 0.9.1

## Summary

CUB 0.9.1 is a minor release.

## Bug Fixes

- Fix for ambiguity in `cub::BlockScan::Reduce` between generic reduction and
    summation.
  Summation entrypoints are now called `::Sum()`, similar to the
    convention in `cub::BlockScan`.
- Small edits to documentation and download tracking.

# CUB 0.9.0

## Summary

Initial preview release.
CUB is the first durable, high-performance library of cooperative block-level,
  warp-level, and thread-level primitives for CUDA kernel programming.

