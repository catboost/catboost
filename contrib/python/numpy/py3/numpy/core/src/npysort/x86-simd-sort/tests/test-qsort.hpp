/*******************************************
 * * Copyright (C) 2022 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

#include "test-qsort-common.h"

template <typename T>
class avx512_sort : public ::testing::Test {
};
TYPED_TEST_SUITE_P(avx512_sort);

TYPED_TEST_P(avx512_sort, test_random)
{
    if (cpu_has_avx512bw()) {
        if ((sizeof(TypeParam) == 2) && (!cpu_has_avx512_vbmi2())) {
            GTEST_SKIP() << "Skipping this test, it requires avx512_vbmi2";
        }
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii < 1024; ++ii) {
            arrsizes.push_back((TypeParam)ii);
        }
        std::vector<TypeParam> arr;
        std::vector<TypeParam> sortedarr;
        for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
            /* Random array */
            arr = get_uniform_rand_array<TypeParam>(arrsizes[ii]);
            sortedarr = arr;
            /* Sort with std::sort for comparison */
            std::sort(sortedarr.begin(), sortedarr.end());
            avx512_qsort<TypeParam>(arr.data(), arr.size());
            ASSERT_EQ(sortedarr, arr) << "Array size = " << arrsizes[ii];
            arr.clear();
            sortedarr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw";
    }
}

TYPED_TEST_P(avx512_sort, test_reverse)
{
    if (cpu_has_avx512bw()) {
        if ((sizeof(TypeParam) == 2) && (!cpu_has_avx512_vbmi2())) {
            GTEST_SKIP() << "Skipping this test, it requires avx512_vbmi2";
        }
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii < 1024; ++ii) {
            arrsizes.push_back((TypeParam)(ii + 1));
        }
        std::vector<TypeParam> arr;
        std::vector<TypeParam> sortedarr;
        for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
            /* reverse array */
            for (int jj = 0; jj < arrsizes[ii]; ++jj) {
                arr.push_back((TypeParam)(arrsizes[ii] - jj));
            }
            sortedarr = arr;
            /* Sort with std::sort for comparison */
            std::sort(sortedarr.begin(), sortedarr.end());
            avx512_qsort<TypeParam>(arr.data(), arr.size());
            ASSERT_EQ(sortedarr, arr) << "Array size = " << arrsizes[ii];
            arr.clear();
            sortedarr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw";
    }
}

TYPED_TEST_P(avx512_sort, test_constant)
{
    if (cpu_has_avx512bw()) {
        if ((sizeof(TypeParam) == 2) && (!cpu_has_avx512_vbmi2())) {
            GTEST_SKIP() << "Skipping this test, it requires avx512_vbmi2";
        }
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii < 1024; ++ii) {
            arrsizes.push_back((TypeParam)(ii + 1));
        }
        std::vector<TypeParam> arr;
        std::vector<TypeParam> sortedarr;
        for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
            /* constant array */
            for (int jj = 0; jj < arrsizes[ii]; ++jj) {
                arr.push_back(ii);
            }
            sortedarr = arr;
            /* Sort with std::sort for comparison */
            std::sort(sortedarr.begin(), sortedarr.end());
            avx512_qsort<TypeParam>(arr.data(), arr.size());
            ASSERT_EQ(sortedarr, arr) << "Array size = " << arrsizes[ii];
            arr.clear();
            sortedarr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw";
    }
}

TYPED_TEST_P(avx512_sort, test_small_range)
{
    if (cpu_has_avx512bw()) {
        if ((sizeof(TypeParam) == 2) && (!cpu_has_avx512_vbmi2())) {
            GTEST_SKIP() << "Skipping this test, it requires avx512_vbmi2";
        }
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii < 1024; ++ii) {
            arrsizes.push_back((TypeParam)(ii + 1));
        }
        std::vector<TypeParam> arr;
        std::vector<TypeParam> sortedarr;
        for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
            arr = get_uniform_rand_array<TypeParam>(arrsizes[ii], 20, 1);
            sortedarr = arr;
            /* Sort with std::sort for comparison */
            std::sort(sortedarr.begin(), sortedarr.end());
            avx512_qsort<TypeParam>(arr.data(), arr.size());
            ASSERT_EQ(sortedarr, arr) << "Array size = " << arrsizes[ii];
            arr.clear();
            sortedarr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw";
    }
}

TYPED_TEST_P(avx512_sort, test_max_value_at_end_of_array)
{
    if (!cpu_has_avx512bw()) {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw ISA";
    }
    if ((sizeof(TypeParam) == 2) && (!cpu_has_avx512_vbmi2())) {
        GTEST_SKIP() << "Skipping this test, it requires avx512_vbmi2";
    }
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 1; ii <= 1024; ++ii) {
        arrsizes.push_back(ii);
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    for (auto &size : arrsizes) {
        arr = get_uniform_rand_array<TypeParam>(size);
        if (std::numeric_limits<TypeParam>::has_infinity) {
            arr[size - 1] = std::numeric_limits<TypeParam>::infinity();
        }
        else {
            arr[size - 1] = std::numeric_limits<TypeParam>::max();
        }
        sortedarr = arr;
        avx512_qsort(arr.data(), arr.size());
        std::sort(sortedarr.begin(), sortedarr.end());
        EXPECT_EQ(sortedarr, arr) << "Array size = " << size;
        arr.clear();
        sortedarr.clear();
    }
}

REGISTER_TYPED_TEST_SUITE_P(avx512_sort,
                            test_random,
                            test_reverse,
                            test_constant,
                            test_small_range,
                            test_max_value_at_end_of_array);
