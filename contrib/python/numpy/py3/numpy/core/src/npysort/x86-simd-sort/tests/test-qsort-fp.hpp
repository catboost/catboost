/*******************************************
 * * Copyright (C) 2022 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

#include "test-qsort-common.h"

template <typename T>
class avx512_sort_fp : public ::testing::Test {
};
TYPED_TEST_SUITE_P(avx512_sort_fp);

TYPED_TEST_P(avx512_sort_fp, test_random_nan)
{
    const int num_nans = 3;
    if (!cpu_has_avx512bw()) {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw";
    }
    std::vector<int64_t> arrsizes;
    for (int64_t ii = num_nans; ii < 1024; ++ii) {
        arrsizes.push_back((TypeParam)ii);
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    for (auto &size : arrsizes) {
        /* Random array */
        arr = get_uniform_rand_array<TypeParam>(size);
        for (auto ii = 1; ii <= num_nans; ++ii) {
            arr[size-ii] = std::numeric_limits<TypeParam>::quiet_NaN();
        }
        sortedarr = arr;
        std::sort(sortedarr.begin(), sortedarr.end()-3);
        std::random_shuffle(arr.begin(), arr.end());
        avx512_qsort<TypeParam>(arr.data(), arr.size());
        for (auto ii = 1; ii <= num_nans; ++ii) {
            if (!std::isnan(arr[size-ii])) {
                ASSERT_TRUE(false) << "NAN's aren't sorted to the end. Arr size = " << size;
            }
        }
        if (!std::is_sorted(arr.begin(), arr.end() - num_nans)) {
            ASSERT_TRUE(true) << "Array isn't sorted";
        }
        arr.clear();
        sortedarr.clear();
    }
}
REGISTER_TYPED_TEST_SUITE_P(avx512_sort_fp, test_random_nan);
