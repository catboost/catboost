#include "test-qsort-common.h"

template <typename T>
class avx512_partial_sort : public ::testing::Test {
};
TYPED_TEST_SUITE_P(avx512_partial_sort);

TYPED_TEST_P(avx512_partial_sort, test_ranges)
{
    int64_t arrsize = 1024;
    int64_t nranges = 500;

    if (cpu_has_avx512bw()) {
        if ((sizeof(TypeParam) == 2) && (!cpu_has_avx512_vbmi2())) {
            GTEST_SKIP() << "Skipping this test, it requires avx512_vbmi2";
        }
        std::vector<TypeParam> arr;
        std::vector<TypeParam> sortedarr;
        std::vector<TypeParam> psortedarr;
        /* Random array */
        arr = get_uniform_rand_array<TypeParam>(arrsize);
        sortedarr = arr;
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());

        for (size_t ii = 0; ii < nranges; ++ii) {
            psortedarr = arr;

            /* Pick a random number of elements to sort at the beginning of the array */
            int k = get_uniform_rand_array<int64_t>(1, arrsize, 1).front();

            /* Sort the range and verify all the required elements match the presorted set */
            avx512_partial_qsort<TypeParam>(
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
        GTEST_SKIP() << "Skipping this test, it requires avx512bw";
    }
}

REGISTER_TYPED_TEST_SUITE_P(avx512_partial_sort, test_ranges);
