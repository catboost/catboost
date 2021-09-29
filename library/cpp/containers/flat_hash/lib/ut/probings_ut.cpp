#include <library/cpp/containers/flat_hash/lib/probings.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace NFlatHash;

namespace {
    struct TDummySizeFitter {
        constexpr auto EvalIndex(size_t idx, size_t) const {
            return idx;
        }
    };

    constexpr TDummySizeFitter SIZE_FITTER;

    auto atLeast13 = [](size_t idx) { return idx >= 13; };
}

Y_UNIT_TEST_SUITE(TProbingsTest) {
    Y_UNIT_TEST(LinearProbingTest) {
        using TProbing = TLinearProbing;
        UNIT_ASSERT_EQUAL(TProbing::FindBucket(SIZE_FITTER, 1, 0, atLeast13), 13);
    }

    Y_UNIT_TEST(QuadraticProbingTest) {
        using TProbing = TQuadraticProbing;
        UNIT_ASSERT_EQUAL(TProbing::FindBucket(SIZE_FITTER, 1, 0, atLeast13), 17);
    }

    Y_UNIT_TEST(DenseProbingTest) {
        using TProbing = TDenseProbing;
        UNIT_ASSERT_EQUAL(TProbing::FindBucket(SIZE_FITTER, 1, 0, atLeast13), 16);
    }
}
