<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon'></a>

Thrust: Code at the speed of light
==================================

Thrust is a C++ parallel programming library which resembles the C++ Standard
Library. Thrust's **high-level** interface greatly enhances
programmer **productivity** while enabling performance portability between
GPUs and multicore CPUs. **Interoperability** with established technologies
(such as CUDA, TBB, and OpenMP) facilitates integration with existing
software. Develop **high-performance** applications rapidly with Thrust!

Thrust is included in the NVIDIA HPC SDK and the CUDA Toolkit.

Quick Start: Using Thrust From Your Project
-------------------------------------------

To use Thrust from your project, first recursively clone the Thrust Github repository:

```
git clone --recursive https://github.com/NVIDIA/thrust.git
```

Thrust is a header-only library; there is no need to build or install the project
unless you want to run the Thrust unit tests.

For CMake-based projects, we provide a CMake package for use with
`find_package`. See the [CMake README](thrust/cmake/README.md) for more
information. Thrust can also be added via `add_subdirectory` or tools like
the [CMake Package Manager](https://github.com/TheLartians/CPM.cmake).

For non-CMake projects, compile with:
- The Thrust include path (`-I<thrust repo root>/thrust`)
- The CUB include path, if using the CUDA device system (`-I<thrust repo root>/dependencies/cub/`)
- By default, the CPP host system and CUDA device system are used. 
  These can be changed using compiler definitions:
  - `-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_XXX`,
     where `XXX` is `CPP` (serial, default), `OMP` (OpenMP), or `TBB` (Intel TBB)
  - `-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_XXX`, where `XXX` is 
    `CPP`, `OMP`, `TBB`, or `CUDA` (default).

Examples
--------

Thrust is best explained through examples. The following source code
generates random numbers serially and then transfers them to a parallel
device where they are sorted.

```c++
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

int main(void)
{
  // generate 32M random numbers serially
  thrust::host_vector<int> h_vec(32 << 20);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer data to the device
  thrust::device_vector<int> d_vec = h_vec;

  // sort data on the device (846M keys per second on GeForce GTX 480)
  thrust::sort(d_vec.begin(), d_vec.end());

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  return 0;
}
```

This code sample computes the sum of 100 random numbers in parallel:

```c++
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <algorithm>
#include <cstdlib>

int main(void)
{
  // generate random data serially
  thrust::host_vector<int> h_vec(100);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer to device and compute sum
  thrust::device_vector<int> d_vec = h_vec;
  int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
  return 0;
}
```

CI Status
---------

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-gpu-build/CXX_TYPE=gcc,CXX_VER=7,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-gpu-build/CXX_TYPE=gcc,CXX_VER=7,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20GCC%207%20build%20and%20device%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20GCC%209%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=8,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=8,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20GCC%208%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=7,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=7,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20GCC%207%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=6,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=6,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20GCC%206%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=5,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=gcc,CXX_VER=5,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20GCC%205%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=clang,CXX_VER=9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=clang,CXX_VER=9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20Clang%209%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=clang,CXX_VER=8,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=clang,CXX_VER=8,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20Clang%208%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=clang,CXX_VER=7,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=clang,CXX_VER=7,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20Clang%207%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=icc,CXX_VER=latest,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=icc,CXX_VER=latest,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=cuda,SDK_VER=11.0-devel/badge/icon?subject=NVCC%2011.0%20%2B%20ICC%20build%20and%20host%20tests'></a>

<a href='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=nvcxx,CXX_VER=20.9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=nvhpc,SDK_VER=20.9-devel/'><img src='https://gpuci.gpuopenanalytics.com/job/nvidia/job/thrust/job/prb/job/thrust-cpu-build/CXX_TYPE=nvcxx,CXX_VER=20.9,OS_TYPE=ubuntu,OS_VER=20.04,SDK_TYPE=nvhpc,SDK_VER=20.9-devel/badge/icon?subject=NVC%2B%2B%2020.9%20build%20and%20host%20tests'></a>

Supported Compilers
-------------------

Thrust is regularly tested using the specified versions of the following
compilers. Unsupported versions may emit deprecation warnings, which can be
silenced by defining THRUST_IGNORE_DEPRECATED_COMPILER during compilation.

- NVCC 11.0+
- NVC++ 20.9+
- GCC 5+
- Clang 7+
- MSVC 2019+ (19.20/16.0/14.20)

Releases
--------

Thrust is distributed with the NVIDIA HPC SDK and the CUDA Toolkit in addition
to GitHub.

See the [changelog](CHANGELOG.md) for details about specific releases.

| Thrust Release    | Included In                             |
| ----------------- | --------------------------------------- |
| 1.13.0            | NVIDIA HPC SDK 21.7                     |
| 1.12.1            | CUDA Toolkit 11.4                       |
| 1.12.0            | NVIDIA HPC SDK 21.3                     |
| 1.11.0            | CUDA Toolkit 11.3                       |
| 1.10.0            | NVIDIA HPC SDK 20.9 & CUDA Toolkit 11.2 |
| 1.9.10-1          | NVIDIA HPC SDK 20.7 & CUDA Toolkit 11.1 |
| 1.9.10            | NVIDIA HPC SDK 20.5                     |
| 1.9.9             | CUDA Toolkit 11.0                       |
| 1.9.8-1           | NVIDIA HPC SDK 20.3                     |
| 1.9.8             | CUDA Toolkit 11.0 Early Access          |
| 1.9.7-1           | CUDA Toolkit 10.2 for Tegra             |
| 1.9.7             | CUDA Toolkit 10.2                       |
| 1.9.6-1           | NVIDIA HPC SDK 20.3                     |
| 1.9.6             | CUDA Toolkit 10.1 Update 2              |
| 1.9.5             | CUDA Toolkit 10.1 Update 1              |
| 1.9.4             | CUDA Toolkit 10.1                       |
| 1.9.3             | CUDA Toolkit 10.0                       |
| 1.9.2             | CUDA Toolkit 9.2                        |
| 1.9.1-2           | CUDA Toolkit 9.1                        |
| 1.9.0-5           | CUDA Toolkit 9.0                        |
| 1.8.3             | CUDA Toolkit 8.0                        |
| 1.8.2             | CUDA Toolkit 7.5                        |
| 1.8.1             | CUDA Toolkit 7.0                        |
| 1.8.0             |                                         |
| 1.7.2             | CUDA Toolkit 6.5                        |
| 1.7.1             | CUDA Toolkit 6.0                        |
| 1.7.0             | CUDA Toolkit 5.5                        |
| 1.6.0             |                                         |
| 1.5.3             | CUDA Toolkit 5.0                        |
| 1.5.2             | CUDA Toolkit 4.2                        |
| 1.5.1             | CUDA Toolkit 4.1                        |
| 1.5.0             |                                         |
| 1.4.0             | CUDA Toolkit 4.0                        |
| 1.3.0             |                                         |
| 1.2.1             |                                         |
| 1.2.0             |                                         |
| 1.1.1             |                                         |
| 1.1.0             |                                         |
| 1.0.0             |                                         |

Development Process
-------------------

Thrust uses the [CMake build system](https://cmake.org/) to build unit tests,
examples, and header tests. To build Thrust as a developer, the following
recipe should be followed:

```
# Clone Thrust and CUB repos recursively:
git clone --recursive https://github.com/NVIDIA/thrust.git
cd thrust

# Create build directory:
mkdir build
cd build

# Configure -- use one of the following:
cmake ..   # Command line interface.
ccmake ..  # ncurses GUI (Linux only)
cmake-gui  # Graphical UI, set source/build directories in the app

# Build:
cmake --build . -j <num jobs>   # invokes make (or ninja, etc)

# Run tests and examples:
ctest
```

By default, a serial `CPP` host system, `CUDA` accelerated device system, and
C++14 standard are used. This can be changed in CMake. More information on
configuring your Thrust build and creating a pull request can be found in
[CONTRIBUTING.md](CONTRIBUTING.md).
