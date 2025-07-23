<!--
******************************************************************************
* 
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/-->

# Release Notes <!-- omit in toc -->
This document contains changes of oneTBB compared to the last release.

## Table of Contents <!-- omit in toc -->
- [New Features](new-features)
- [Preview Features](#preview-features)
- [Known Limitations](#known-limitations)
- [Issues Fixed](#issues-fixed)
- [Open-Source Contributions Integrated](#open-source-contributions-integrated)

## :tada: New Features
- The oneTBB repository migrated to the new [UXL Foundation](https://github.com/uxlfoundation/oneTBB) organization.
- ``blocked_nd_range`` is now a fully supported feature.
- Introduced the ``ONETBB_SPEC_VERSION`` macro to specify the version of oneAPI specification implemented by the current version of the library.


## :rocket: Preview Features
- Added the explicit deduction guides to ``blocked_nd_range`` to support C++17 Class Template Argument Deduction.
- Extended ``task_arena`` API to select TBB workers leave policy and to hint the start and the end of parallel computations.


## :rotating_light: Known Limitations
- The ``oneapi::tbb::info`` namespace interfaces might unexpectedly change the process affinity mask on Windows* OS systems (see https://github.com/open-mpi/hwloc/issues/366 for details) when using hwloc version lower than 2.5.
- Using a hwloc version other than 1.11, 2.0, or 2.5 may cause an undefined behavior on Windows OS. See https://github.com/open-mpi/hwloc/issues/477 for details.
- The NUMA topology may be detected incorrectly on Windows* OS machines where the number of NUMA node threads exceeds the size of 1 processor group.
- On Windows OS on ARM64*, when compiling an application using oneTBB with the Microsoft* Compiler, the compiler issues a warning C4324 that a structure was padded due to the alignment specifier. Consider suppressing the warning by specifying /wd4324 to the compiler command line.
- C++ exception handling mechanism on Windows* OS on ARM64* might corrupt memory if an exception is thrown from any oneTBB parallel algorithm (see Windows* OS on ARM64* compiler issue: https://developercommunity.visualstudio.com/t/ARM64-incorrect-stack-unwinding-for-alig/1544293.
- When CPU resource coordination is enabled, tasks from a lower-priority ``task_arena`` might be executed before tasks from a higher-priority ``task_arena``.
- Using oneTBB on WASM* may cause applications to run in a single thread. See [Limitations of WASM Support](https://github.com/uxlfoundation/oneTBB/blob/master/WASM_Support.md#limitations).

> **_NOTE:_**  To see known limitations that impact all versions of oneTBB, refer to [oneTBB Documentation](https://uxlfoundation.github.io/oneTBB/main/intro/limitations.html).


## :hammer: Issues Fixed
- Fixed deadlock when using `tbb::concurrent_vector::grow_by()` (https://github.com/uxlfoundation/oneTBB/issues/1531).
- Fixed assertion in the Debug version of oneTBB on systems with multiple processor groups.
- Fixed issues with Flow Graph priorities when using limited concurrency nodes (https://github.com/uxlfoundation/oneTBB/issues/1595).
- Improved support of ``tbb::task_arena::constraints`` functionality on Windows* systems with multiple processor groups.
- Fixed ``concurrent_queue`` and ``concurrent_bounded_queue`` capacity preserving on copying, moving, and swapping (https://github.com/uxlfoundation/oneTBB/issues/1598).
- Fixed ``parallel_for_each`` compilation issues on GCC 9 in C++20 mode (https://github.com/uxlfoundation/oneTBB/issues/1552).


## :octocat: Open-Source Contributions Integrated
- Fixed linkage errors when the application is built with the hidden symbols visibility. Contributed by Vladislav Shchapov (https://github.com/uxlfoundation/oneTBB/pull/1114).
- On Linux* OS, for external thread, determined stack size using POSIX* API instead of relying on the stack size of a worker thread. Contributed by bongkyu7-kim (https://github.com/uxlfoundation/oneTBB/pull/1485).
- Added a CMake option to use relative paths instead of full paths in debug information. Contributed by Fang Xu (https://github.com/uxlfoundation/oneTBB/pull/1401).
- Improved OpenBSD* support by removing the use of direct syscalls. Contributed by Brad Smith (https://github.com/uxlfoundation/oneTBB/pull/1499).
- Fixed build issues on ARM64* when using Bazel. Contributed by snadampal (https://github.com/uxlfoundation/oneTBB/pull/1571).
- Suppressed deprecation warnings for CMake versions earlier than 3.10 when using the latest CMake. Contributed by Vladislav Shchapov (https://github.com/uxlfoundation/oneTBB/pull/1585).
