#include <catboost/libs/helpers/math_utils.h>

#include <limits>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(TMathUtilsTest) {
    Y_UNIT_TEST(TestEqualWithNans) {
        UNIT_ASSERT(EqualWithNans(0.1f, 0.1f));
        UNIT_ASSERT(!EqualWithNans(0.1f, 0.0f));

        auto floatNan = std::numeric_limits<float>::quiet_NaN();
        auto floatInf = std::numeric_limits<float>::infinity();

        UNIT_ASSERT(EqualWithNans(floatNan, floatNan));
        UNIT_ASSERT(EqualWithNans(floatNan, -floatNan));
        UNIT_ASSERT(!EqualWithNans(floatNan, 0.0f));
        UNIT_ASSERT(!EqualWithNans(floatInf, 0.0f));
        UNIT_ASSERT(!EqualWithNans(floatInf, 1.0f));
        UNIT_ASSERT(!EqualWithNans(floatNan, floatInf));
        UNIT_ASSERT(EqualWithNans(floatInf, floatInf));
        UNIT_ASSERT(!EqualWithNans(floatInf, -floatInf));

        UNIT_ASSERT(EqualWithNans(0.1, 0.1));
        UNIT_ASSERT(!EqualWithNans(0.1, 0.0));

        auto doubleNan = std::numeric_limits<double>::quiet_NaN();
        auto doubleInf = std::numeric_limits<double>::infinity();
        UNIT_ASSERT(EqualWithNans(doubleNan, doubleNan));
        UNIT_ASSERT(EqualWithNans(doubleNan, -doubleNan));
        UNIT_ASSERT(!EqualWithNans(doubleNan, 0.0));
        UNIT_ASSERT(!EqualWithNans(doubleInf, 0.0));
        UNIT_ASSERT(!EqualWithNans(doubleInf, 1.0));
        UNIT_ASSERT(!EqualWithNans(doubleNan, doubleInf));
        UNIT_ASSERT(EqualWithNans(doubleInf, doubleInf));
        UNIT_ASSERT(!EqualWithNans(doubleInf, -doubleInf));
    }
}
