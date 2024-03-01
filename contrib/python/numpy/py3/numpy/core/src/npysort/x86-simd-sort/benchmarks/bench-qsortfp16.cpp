#include "avx512fp16-16bit-qsort.hpp"
#include "cpuinfo.h"
#include "rand_array.h"
#include <benchmark/benchmark.h>

template <typename T>
static void avx512_qsort(benchmark::State &state)
{
    if (cpu_has_avx512fp16()) {
        // Perform setup here
        size_t ARRSIZE = state.range(0);
        std::vector<T> arr;
        std::vector<T> arr_bkp;

        /* Initialize elements */
        for (size_t jj = 0; jj < ARRSIZE; ++jj) {
            _Float16 temp = (float)rand() / (float)(RAND_MAX);
            arr.push_back(temp);
        }
        arr_bkp = arr;

        /* call avx512 quicksort */
        for (auto _ : state) {
            avx512_qsort<T>(arr.data(), ARRSIZE);
            state.PauseTiming();
            arr = arr_bkp;
            state.ResumeTiming();
        }
    }
    else {
        state.SkipWithMessage("Requires AVX512-FP16 ISA");
    }
}

template <typename T>
static void stdsort(benchmark::State &state)
{
    if (cpu_has_avx512fp16()) {
        // Perform setup here
        size_t ARRSIZE = state.range(0);
        std::vector<T> arr;
        std::vector<T> arr_bkp;

        for (size_t jj = 0; jj < ARRSIZE; ++jj) {
            _Float16 temp = (float)rand() / (float)(RAND_MAX);
            arr.push_back(temp);
        }
        arr_bkp = arr;

        /* call std::sort */
        for (auto _ : state) {
            std::sort(arr.begin(), arr.end());
            state.PauseTiming();
            arr = arr_bkp;
            state.ResumeTiming();
        }
    }
    else {
        state.SkipWithMessage("Requires AVX512-FP16 ISA");
    }
}

// Register the function as a benchmark
BENCHMARK(avx512_qsort<_Float16>)->Arg(10000)->Arg(1000000);
BENCHMARK(stdsort<_Float16>)->Arg(10000)->Arg(1000000);

template <typename T>
static void avx512_qselect(benchmark::State &state)
{
    if (cpu_has_avx512fp16()) {
        // Perform setup here
        int64_t K = state.range(0);
        size_t ARRSIZE = 10000;
        std::vector<T> arr;
        std::vector<T> arr_bkp;

        /* Initialize elements */
        for (size_t jj = 0; jj < ARRSIZE; ++jj) {
            _Float16 temp = (float)rand() / (float)(RAND_MAX);
            arr.push_back(temp);
        }
        arr_bkp = arr;

        /* call avx512 quickselect */
        for (auto _ : state) {
            avx512_qselect<T>(arr.data(), K, ARRSIZE);

            state.PauseTiming();
            arr = arr_bkp;
            state.ResumeTiming();
        }
    }
    else {
        state.SkipWithMessage("Requires AVX512-FP16 ISA");
    }
}

template <typename T>
static void stdnthelement(benchmark::State &state)
{
    if (cpu_has_avx512fp16()) {
        // Perform setup here
        int64_t K = state.range(0);
        size_t ARRSIZE = 10000;
        std::vector<T> arr;
        std::vector<T> arr_bkp;

        /* Initialize elements */
        for (size_t jj = 0; jj < ARRSIZE; ++jj) {
            _Float16 temp = (float)rand() / (float)(RAND_MAX);
            arr.push_back(temp);
        }
        arr_bkp = arr;

        /* call std::nth_element */
        for (auto _ : state) {
            std::nth_element(arr.begin(), arr.begin() + K, arr.end());

            state.PauseTiming();
            arr = arr_bkp;
            state.ResumeTiming();
        }
    }
    else {
        state.SkipWithMessage("Requires AVX512-FP16 ISA");
    }
}

// Register the function as a benchmark
BENCHMARK(avx512_qselect<_Float16>)->Arg(10)->Arg(100)->Arg(1000)->Arg(5000);
BENCHMARK(stdnthelement<_Float16>)->Arg(10)->Arg(100)->Arg(1000)->Arg(5000);

template <typename T>
static void avx512_partial_qsort(benchmark::State &state)
{
    if (cpu_has_avx512fp16()) {
        // Perform setup here
        int64_t K = state.range(0);
        size_t ARRSIZE = 10000;
        std::vector<T> arr;
        std::vector<T> arr_bkp;

        /* Initialize elements */
        for (size_t jj = 0; jj < ARRSIZE; ++jj) {
            _Float16 temp = (float)rand() / (float)(RAND_MAX);
            arr.push_back(temp);
        }
        arr_bkp = arr;

        /* call avx512_partial_qsort */
        for (auto _ : state) {
            avx512_partial_qsort<T>(arr.data(), K, ARRSIZE);

            state.PauseTiming();
            arr = arr_bkp;
            state.ResumeTiming();
        }
    }
    else {
        state.SkipWithMessage("Requires AVX512-FP16 ISA");
    }
}

template <typename T>
static void stdpartialsort(benchmark::State &state)
{
    if (cpu_has_avx512fp16()) {
        // Perform setup here
        int64_t K = state.range(0);
        size_t ARRSIZE = 10000;
        std::vector<T> arr;
        std::vector<T> arr_bkp;

        /* Initialize elements */
        for (size_t jj = 0; jj < ARRSIZE; ++jj) {
            _Float16 temp = (float)rand() / (float)(RAND_MAX);
            arr.push_back(temp);
        }
        arr_bkp = arr;

        /* call std::partial_sort */
        for (auto _ : state) {
            std::partial_sort(arr.begin(), arr.begin() + K, arr.end());

            state.PauseTiming();
            arr = arr_bkp;
            state.ResumeTiming();
        }
    }
    else {
        state.SkipWithMessage("Requires AVX512-FP16 ISA");
    }
}

// Register the function as a benchmark
BENCHMARK(avx512_partial_qsort<_Float16>)
        ->Arg(10)
        ->Arg(100)
        ->Arg(1000)
        ->Arg(5000);
BENCHMARK(stdpartialsort<_Float16>)->Arg(10)->Arg(100)->Arg(1000)->Arg(5000);
