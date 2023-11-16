#include "test-qsort-common.h"

template <typename T>
class avx512_select : public ::testing::Test {
};
TYPED_TEST_SUITE_P(avx512_select);

TYPED_TEST_P(avx512_select, test_random)
{
    if (cpu_has_avx512bw()) {
        if ((sizeof(TypeParam) == 2) && (!cpu_has_avx512_vbmi2())) {
            GTEST_SKIP() << "Skipping this test, it requires avx512_vbmi2";
        }
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii < 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<TypeParam> arr;
        std::vector<TypeParam> sortedarr;
        std::vector<TypeParam> psortedarr;
        for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
            /* Random array */
            arr = get_uniform_rand_array<TypeParam>(arrsizes[ii]);
            sortedarr = arr;
            /* Sort with std::sort for comparison */
            std::sort(sortedarr.begin(), sortedarr.end());
            for (size_t k = 0; k < arr.size(); ++k) {
                psortedarr = arr;
                avx512_qselect<TypeParam>(
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
        GTEST_SKIP() << "Skipping this test, it requires avx512bw";
    }
}

TYPED_TEST_P(avx512_select, test_small_range)
{
    if (cpu_has_avx512bw()) {
        if ((sizeof(TypeParam) == 2) && (!cpu_has_avx512_vbmi2())) {
            GTEST_SKIP() << "Skipping this test, it requires avx512_vbmi2";
        }
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii < 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<TypeParam> arr;
        std::vector<TypeParam> sortedarr;
        std::vector<TypeParam> psortedarr;
        for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
            /* Random array */
            arr = get_uniform_rand_array<TypeParam>(arrsizes[ii], 20, 1);
            sortedarr = arr;
            /* Sort with std::sort for comparison */
            std::sort(sortedarr.begin(), sortedarr.end());
            for (size_t k = 0; k < arr.size(); ++k) {
                psortedarr = arr;
                avx512_qselect<TypeParam>(
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
        GTEST_SKIP() << "Skipping this test, it requires avx512bw";
    }
}

REGISTER_TYPED_TEST_SUITE_P(avx512_select, test_random, test_small_range);
