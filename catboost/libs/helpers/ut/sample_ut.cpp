#include <catboost/libs/helpers/sample.h>

#include <util/generic/hash_set.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(TSampleIndicesTest) {

    template<class T>
    void CheckOneCase(size_t n, size_t k) {
        TRestorableFastRng64 rand(0);
        const TVector<T> sampledIndices = SampleIndices<T>(n, k, &rand);

        UNIT_ASSERT_VALUES_EQUAL(sampledIndices.size(), k);

        THashSet<T> sampledIndicesSet;
        for (auto index : sampledIndices) {
            UNIT_ASSERT((size_t)index >= (T)0);
            UNIT_ASSERT((size_t)index < n);
            UNIT_ASSERT(!sampledIndicesSet.contains(index));
            sampledIndicesSet.insert(index);
        }
    }

    Y_UNIT_TEST(Simple) {
        CheckOneCase<int>(0, 0);
        CheckOneCase<int>(1, 0);
        CheckOneCase<int>(1, 1);
        CheckOneCase<int>(2, 2);
        CheckOneCase<ui32>(100, 100);
        CheckOneCase<ui32>(100, 0);
        CheckOneCase<ui32>(100, 1);
        CheckOneCase<ui64>(1000, 20);
        CheckOneCase<ui64>(935, 34);
        CheckOneCase<ui64>(935, 623);
        CheckOneCase<ui64>(935, 935);
    }
}
