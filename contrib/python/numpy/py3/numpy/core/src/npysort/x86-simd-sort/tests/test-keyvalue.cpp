/*******************************************
 * * Copyright (C) 2022 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

#include "avx512-64bit-keyvaluesort.hpp"
#include "cpuinfo.h"
#include "rand_array.h"
#include <gtest/gtest.h>
#include <vector>
#define inf X86_SIMD_SORT_INFINITY

template <typename K, typename V = uint64_t>
struct sorted_t {
    K key;
    K value;
};

template <typename K, typename V = uint64_t>
bool compare(sorted_t<K, V> a, sorted_t<K, V> b)
{
    return a.key == b.key ? a.value < b.value : a.key < b.key;
}

template <typename K>
class KeyValueSort : public ::testing::Test {
};

TYPED_TEST_SUITE_P(KeyValueSort);

TYPED_TEST_P(KeyValueSort, test_64bit_random_data)
{
    if (cpu_has_avx512bw()) {
        std::vector<int64_t> keysizes;
        for (int64_t ii = 0; ii < 1024; ++ii) {
            keysizes.push_back((TypeParam)ii);
        }
        std::vector<TypeParam> keys;
        std::vector<uint64_t> values;
        std::vector<sorted_t<TypeParam, uint64_t>> sortedarr;

        for (size_t ii = 0; ii < keysizes.size(); ++ii) {
            /* Random array */
            keys = get_uniform_rand_array_with_uniquevalues<TypeParam>(
                    keysizes[ii]);
            values = get_uniform_rand_array<uint64_t>(keysizes[ii]);
            for (size_t i = 0; i < keys.size(); i++) {
                sorted_t<TypeParam, uint64_t> tmp_s;
                tmp_s.key = keys[i];
                tmp_s.value = values[i];
                sortedarr.emplace_back(tmp_s);
            }
            /* Sort with std::sort for comparison */
            std::sort(sortedarr.begin(),
                      sortedarr.end(),
                      compare<TypeParam, uint64_t>);
            avx512_qsort_kv<TypeParam>(keys.data(), values.data(), keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                ASSERT_EQ(keys[i], sortedarr[i].key);
                ASSERT_EQ(values[i], sortedarr[i].value);
            }
            keys.clear();
            values.clear();
            sortedarr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw";
    }
}

TEST(KeyValueSort, test_inf_at_endofarray)
{
    std::vector<double> key = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, inf};
    std::vector<double> key_sorted
            = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, inf};
    std::vector<uint64_t> val = {7, 6, 5, 4, 3, 2, 1, 0, 8};
    std::vector<uint64_t> val_sorted = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    avx512_qsort_kv(key.data(), val.data(), key.size());
    ASSERT_EQ(key, key_sorted);
    ASSERT_EQ(val, val_sorted);
}

REGISTER_TYPED_TEST_SUITE_P(KeyValueSort, test_64bit_random_data);

using TypesKv = testing::Types<double, uint64_t, int64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(T, KeyValueSort, TypesKv);
