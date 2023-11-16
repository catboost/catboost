# x86-simd-sort

C++ header file library for SIMD based 16-bit, 32-bit and 64-bit data type
sorting on x86 processors. Source header files are available in src directory.
We currently only have AVX-512 based implementation of quicksort. This
repository also includes a test suite which can be built and run to test the
sorting algorithms for correctness. It also has benchmarking code to compare
its performance relative to std::sort.

## Algorithm details

The ideas and code are based on these two research papers [1] and [2]. On a
high level, the idea is to vectorize quicksort partitioning using AVX-512
compressstore instructions. If the array size is < 128, then use Bitonic
sorting network implemented on 512-bit registers.  The precise network
definitions depend on the size of the dtype and are defined in separate files:
`avx512-16bit-qsort.hpp`, `avx512-32bit-qsort.hpp` and
`avx512-64bit-qsort.hpp`. Article [4] is a good resource for bitonic sorting
network. The core implementations of the vectorized qsort functions
`avx512_qsort<T>(T*, int64_t)` are modified versions of avx2 quicksort
presented in the paper [2] and source code associated with that paper [3].

## Handling NAN in float and double arrays

If you expect your array to contain NANs, please be aware that the these
routines **do not preserve your NANs as you pass them**. The
`avx512_qsort<T>()` routine will put all your NAN's at the end of the sorted
array and replace them with `std::nan("1")`. Please take a look at
`avx512_qsort<float>()` and `avx512_qsort<double>()` functions for details.

## Example to include and build this in a C++ code

### Sample code `main.cpp`

```cpp
#include "src/avx512-32bit-qsort.hpp"

int main() {
    const int ARRSIZE = 10;
    std::vector<float> arr;

    /* Initialize elements is reverse order */
    for (int ii = 0; ii < ARRSIZE; ++ii) {
        arr.push_back(ARRSIZE - ii);
    }

    /* call avx512 quicksort */
    avx512_qsort<float>(arr.data(), ARRSIZE);
    return 0;
}

```

### Build using gcc

```
gcc main.cpp -mavx512f -mavx512dq -O3
```

This is a header file only library and we do not provide any compile time and
run time checks which is recommended while including this your source code. A
slightly modified version of this source code has been contributed to
[NumPy](https://github.com/numpy/numpy) (see this [pull
request](https://github.com/numpy/numpy/pull/22315) for details). This NumPy
pull request is a good reference for how to include and build this library with
your source code.

## Build requirements

None, its header files only. However you will need `make` or `meson` to build
the unit tests and benchmarking suite. You will need a relatively modern
compiler to build.

```
gcc >= 8.x
```

### Build using Make

`make` command builds two executables:
- `testexe`: runs a bunch of tests written in ./tests directory.
- `benchexe`: measures performance of these algorithms for various data types
  and compares them to std::sort.

You can use `make test` and `make bench` to build just the `testexe` and
`benchexe` respectively.

### Build using Meson

You can also build `testexe` and `benchexe` using Meson/Ninja with the following
command:

```
meson setup builddir && cd builddir && ninja
```

## Requirements and dependencies

The sorting routines relies only on the C++ Standard Library and requires a
relatively modern compiler to build (gcc 8.x and above). Since they use the
AVX-512 instruction set, they can only run on processors that have AVX-512.
Specifically, the 32-bit and 64-bit require AVX-512F and AVX-512DQ instruction
set. The 16-bit sorting requires the AVX-512F, AVX-512BW and AVX-512 VMBI2
instruction set. The test suite is written using the Google test framework.

## References

* [1] Fast and Robust Vectorized In-Place Sorting of Primitive Types
    https://drops.dagstuhl.de/opus/volltexte/2021/13775/

* [2] A Novel Hybrid Quicksort Algorithm Vectorized using AVX-512 on Intel
Skylake https://arxiv.org/pdf/1704.08579.pdf

* [3] https://github.com/simd-sorting/fast-and-robust: SPDX-License-Identifier: MIT

* [4] http://mitp-content-server.mit.edu:18180/books/content/sectbyfn?collid=books_pres_0&fn=Chapter%2027.pdf&id=8030

