/*******************************************
 * * Copyright (C) 2022 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

#include "avx512fp16-16bit-qsort.hpp"
#include "cpuinfo.h"
#include "rand_array.h"
#include <gtest/gtest.h>
#include <vector>

TEST(avx512_qsort_float16, test_arrsizes)
{
    if (cpu_has_avx512fp16()) {
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii < 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<_Float16> arr;
        std::vector<_Float16> sortedarr;

        for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
            /* Random array */
            for (size_t jj = 0; jj < arrsizes[ii]; ++jj) {
                _Float16 temp = (float)rand() / (float)(RAND_MAX);
                arr.push_back(temp);
                sortedarr.push_back(temp);
            }
            /* Sort with std::sort for comparison */
            std::sort(sortedarr.begin(), sortedarr.end());
            avx512_qsort<_Float16>(arr.data(), arr.size());
            ASSERT_EQ(sortedarr, arr);
            arr.clear();
            sortedarr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512fp16 ISA";
    }
}

TEST(avx512_qsort_float16, test_special_floats)
{
    if (cpu_has_avx512fp16()) {
        const int arrsize = 1111;
        std::vector<_Float16> arr;
        std::vector<_Float16> sortedarr;
        Fp16Bits temp;
        for (size_t jj = 0; jj < arrsize; ++jj) {
            temp.f_ = (float)rand() / (float)(RAND_MAX);
            switch (rand() % 10) {
                case 0: temp.i_ = 0xFFFF; break;
                case 1: temp.i_ = X86_SIMD_SORT_INFINITYH; break;
                case 2: temp.i_ = X86_SIMD_SORT_NEGINFINITYH; break;
                default: break;
            }
            arr.push_back(temp.f_);
            sortedarr.push_back(temp.f_);
        }
        /* Cannot use std::sort because it treats NAN differently */
        avx512_qsort_fp16(reinterpret_cast<uint16_t *>(sortedarr.data()),
                          sortedarr.size());
        avx512_qsort<_Float16>(arr.data(), arr.size());
        // Cannot rely on ASSERT_EQ since it returns false if there are NAN's
        if (memcmp(arr.data(), sortedarr.data(), arrsize * 2) != 0) {
            ASSERT_EQ(sortedarr, arr);
        }
        arr.clear();
        sortedarr.clear();
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512fp16 ISA";
    }
}

TEST(avx512_qselect_float16, test_arrsizes)
{
    if (cpu_has_avx512fp16()) {
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii < 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<_Float16> arr;
        std::vector<_Float16> sortedarr;
        std::vector<_Float16> psortedarr;

        for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
            /* Random array */
            for (size_t jj = 0; jj < arrsizes[ii]; ++jj) {
                _Float16 temp = (float)rand() / (float)(RAND_MAX);
                arr.push_back(temp);
                sortedarr.push_back(temp);
            }
            /* Sort with std::sort for comparison */
            std::sort(sortedarr.begin(), sortedarr.end());
            for (size_t k = 0; k < arr.size(); ++k) {
                psortedarr = arr;
                avx512_qselect<_Float16>(
                        psortedarr.data(), k, psortedarr.size());
                /* index k is correct */
                ASSERT_EQ(sortedarr[k], psortedarr[k]);
                /* Check left partition */
                for (size_t jj = 0; jj < k; jj++) {
                    ASSERT_LE(psortedarr[jj], psortedarr[k]);
                }
                /* Check right partition */
                for (size_t jj = k + 1; jj < arr.size(); jj++) {
                    ASSERT_GE(psortedarr[jj], psortedarr[k]);
                }
                psortedarr.clear();
            }
            arr.clear();
            sortedarr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512fp16 ISA";
    }
}

TEST(avx512_partial_qsort_float16, test_ranges)
{
    if (cpu_has_avx512fp16()) {
        int64_t arrsize = 1024;
        int64_t nranges = 500;

        std::vector<_Float16> arr;
        std::vector<_Float16> sortedarr;
        std::vector<_Float16> psortedarr;

        /* Random array */
        for (size_t ii = 0; ii < arrsize; ++ii) {
            _Float16 temp = (float)rand() / (float)(RAND_MAX);
            arr.push_back(temp);
            sortedarr.push_back(temp);
        }
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());

        for (size_t ii = 0; ii < nranges; ++ii) {
            psortedarr = arr;

            int k = get_uniform_rand_array<int64_t>(1, arrsize, 1).front();

            /* Sort the range and verify all the required elements match the presorted set */
            avx512_partial_qsort<_Float16>(
                    psortedarr.data(), k, psortedarr.size());
            for (size_t jj = 0; jj < k; jj++) {
                ASSERT_EQ(sortedarr[jj], psortedarr[jj]);
            }

            psortedarr.clear();
        }

        arr.clear();
        sortedarr.clear();
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512fp16 ISA";
    }
}
