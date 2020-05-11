#include <catboost/private/libs/quantization_schema/quantize.h>
#include <catboost/private/libs/options/enums.h>

#include <library/cpp/unittest/registar.h>

#include <limits>

Y_UNIT_TEST_SUITE(QuantizationTests) {
    Y_UNIT_TEST(TestQuantizeOnNans) {
        const float borders[] = {1.f};
        const float nan_ = std::numeric_limits<float>::quiet_NaN();

        UNIT_ASSERT_VALUES_EQUAL(0, NCB::Quantize(nan_, borders, ENanMode::Min));
        UNIT_ASSERT_VALUES_EQUAL(1, NCB::Quantize(nan_, borders, ENanMode::Max));
        UNIT_ASSERT_VALUES_EQUAL(0, NCB::Quantize(nan_, borders, ENanMode::Forbidden));
    }
}
