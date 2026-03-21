#include <library/cpp/dot_product/dot_product_avx2.h>
#include <library/cpp/dot_product/dot_product_simple.h>
#include <library/cpp/dot_product/dot_product_sse.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/system/cpu_id.h>
#include <util/generic/vector.h>

using TriWayFunc = TTriWayDotProduct<float> (*)
        (const float* lhs, const float* rhs, size_t length, bool computeRR) noexcept;

namespace {

void TestTriWay(size_t length, TriWayFunc func) {
    const float lhsMultiplier = 0.25f;
    const float rhsMultiplier = -0.5f;

    TVector<float> lhs(length);
    TVector<float> rhs(length);
    for (size_t i = 0; i < length; ++i) {
        lhs[i] = static_cast<float>(i) * lhsMultiplier;
        rhs[i] = static_cast<float>(length - i) * rhsMultiplier;
    }

    const bool computeRRVariants[] = {false, true};
    for (const bool computeRR : computeRRVariants) {
        const auto expected = TriWayDotProductSimple(lhs.data(), rhs.data(), length, computeRR);
        const auto actual = func(lhs.data(), rhs.data(), length, computeRR);

        UNIT_ASSERT_DOUBLES_EQUAL(expected.LL, actual.LL, std::abs(0.00001 * expected.LL));
        UNIT_ASSERT_DOUBLES_EQUAL(expected.LR, actual.LR, std::abs(0.00001 * expected.LR));
        UNIT_ASSERT_DOUBLES_EQUAL(expected.RR, actual.RR, std::abs(0.00001 * expected.RR));
    }
}

} // anonymous namespace

Y_UNIT_TEST_SUITE(TriWayDotProductTest) {

Y_UNIT_TEST(TestTriWayAvx2Maybe) {
    // depending on compiler flags and machine it might:
    // * test AVX2 version as desired
    // * test SSE version
    // * test nothing
    // but that's fine
    if (!NX86::HaveAVX2() || !NX86::HaveFMA()) {
        return;
    }

    for (size_t length = 640; length < 656; ++length) {
        TestTriWay(length, TriWayDotProductAvx2);
    }
}

Y_UNIT_TEST(TestTriWaySse) {
    if (!NX86::HaveSSE2()) {
        return;
    }

    for (size_t length = 640; length < 656; ++length) {
        TestTriWay(length, TriWayDotProductSse);
    }
}

}

