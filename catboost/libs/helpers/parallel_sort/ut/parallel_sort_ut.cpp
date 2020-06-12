#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/parallel_sort/parallel_sort.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/algorithm.h>
#include <util/random/shuffle.h>

static TVector<ui32> GeneratePermutation(size_t size, TRandom& rnd) {
    TVector<ui32> permutation(size);
    std::iota(permutation.begin(), permutation.end(), 0);
    Shuffle(permutation.begin(), permutation.end(), rnd);
    return permutation;
}

static TVector<ui32> RandomSubset(ui32 size, ui32 num, TRandom& rnd) {
    TVector<ui32> result(num);
    for (size_t i = 0; i < num; ++i) {
        result[i] = rnd(size - num + 1);
    }
    Sort(result.begin(), result.end());
    for (size_t i = 0; i < num; ++i) {
        result[i] += i;
    }
    return result;
}

static TVector<ui32> RandomlyDivide(ui32 size, ui32 blocks, TRandom& rnd) {
    if (blocks == 1u) {
        return {size};
    }
    TVector<ui32> result = RandomSubset(size - 1, blocks - 1, rnd);
    result.push_back(size - result.back() - 1);
    for (ui32 i = result.size() - 2; i >= 1; i--) {
        result[i] -= result[i - 1];
    }
    result[0]++;
    return result;
}

static TVector<ui32> RandomVector(ui32 size, ui32 differentCount, TRandom& rnd) {
    TVector<ui32> counts = RandomlyDivide(size, differentCount, rnd);
    TVector<ui32> result;
    for (ui32 i = 0; i < differentCount; ++i) {
        for (ui32 j = 0; j < counts[i]; ++j) {
            result.push_back(i);
        }
    }
    Shuffle(result.begin(), result.end(), rnd);
    return result;
}

static bool CmpLess(ui32 left, ui32 right) {
    return left < right;
}

static bool CmpGreater(ui32 left, ui32 right) {
    return left > right;
}

Y_UNIT_TEST_SUITE(ParallelSortTests) {

    Y_UNIT_TEST(ParallelSortRandomPermutationTest) {
        TRandom rnd(239);
        size_t size = (size_t)(1e6 + 239);
        TVector<ui32> permutation = GeneratePermutation(size, rnd);
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(31);
        NCB::ParallelMergeSort(CmpLess, &permutation, &localExecutor);
        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(permutation[i], i);
        }
    }

    Y_UNIT_TEST(ParallelSortOneThreadSortRandomPermutationTest) {
        TRandom rnd(239 + 239);
        size_t size = (size_t)(1e6 + 239);
        TVector<ui32> permutation = GeneratePermutation(size, rnd);
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(1);
        NCB::ParallelMergeSort(CmpLess, &permutation, &localExecutor);
        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(permutation[i], i);
        }
    }

    Y_UNIT_TEST(ParallelSortOwnBufTest) {
        TRandom rnd(239 + 239 + 239);
        size_t size = (size_t)(1e6 + 239);
        TVector<ui32> permutation = GeneratePermutation(size, rnd);
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(31);
        TVector<ui32> buf(size);
        NCB::ParallelMergeSort(CmpLess, &permutation, &localExecutor, &buf);
        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(permutation[i], i);
        }
    }

    Y_UNIT_TEST(ParallelSortEmptyVectorTest) {
        TVector<ui32> permutation;
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(31);
        NCB::ParallelMergeSort(CmpLess, &permutation, &localExecutor);
        UNIT_ASSERT_VALUES_EQUAL(permutation.size(), 0u);
    }

    Y_UNIT_TEST(ParallelSortSmallVectorTest) {
        TRandom rnd(239);
        size_t size = 23;
        TVector<ui32> permutation = GeneratePermutation(size, rnd);
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(31);
        NCB::ParallelMergeSort(CmpLess, &permutation, &localExecutor);
        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(permutation[i], i);
        }
    }

    Y_UNIT_TEST(ParallelSortOtherCmpTest) {
        size_t size = (size_t)(1e6 + 239);
        TVector<ui32> permutation(size);
        std::iota(permutation.begin(), permutation.end(), 0);
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(31);
        NCB::ParallelMergeSort(CmpGreater, &permutation, &localExecutor);
        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(permutation[i], size - i - 1);
        }
    }

    Y_UNIT_TEST(ParallelSortEqualVectorTest) {
        TRandom rnd(239);
        size_t size = (size_t)(1e6 + 239);
        TVector<ui32> currentVector = RandomVector(size, 1u, rnd);
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(31);
        NCB::ParallelMergeSort(CmpLess, &currentVector, &localExecutor);
        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(currentVector[i], 0u);
        }
    }

    Y_UNIT_TEST(ParallelSortVectorWithVerySmallValuesTest) {
        TRandom rnd(239);
        size_t size = (size_t)(1e6 + 239);
        TVector<ui32> currentVector = RandomVector(size, 13u, rnd);
        NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(31);
        NCB::ParallelMergeSort(CmpLess, &currentVector, &localExecutor);
        for (size_t i = 0; i + 1 < size; ++i) {
            UNIT_ASSERT_GE(currentVector[i + 1], currentVector[i]);
        }
    }

    Y_UNIT_TEST(ParallelSortVectorWithSmallValuesTest) {
        TRandom rnd(239);
        size_t size = (size_t)(1e6 + 239);
        TVector<ui32> currentVector = RandomVector(size, size / 100u, rnd);
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(31);
        NCB::ParallelMergeSort(CmpLess, &currentVector, &localExecutor);
        for (size_t i = 0; i + 1 < size; ++i) {
            UNIT_ASSERT_GE(currentVector[i + 1], currentVector[i]);
        }
    }
}
