#include <library/cpp/testing/unittest/registar.h>

#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/quantization/utils.h>

#include <limits>

Y_UNIT_TEST_SUITE(TQuantizationUtilsTests) {
    Y_UNIT_TEST(TestBinarizeOnNans) {
        const float borders[] = {1.f};
        const float nan_ = std::numeric_limits<float>::quiet_NaN();

        UNIT_ASSERT_VALUES_EQUAL(0, NCB::Binarize<ui32>(ENanMode::Min, borders, nan_));
        UNIT_ASSERT_VALUES_EQUAL(1, NCB::Binarize<ui32>(ENanMode::Max, borders, nan_));
        UNIT_ASSERT_VALUES_EQUAL(0, NCB::Binarize<ui32>(ENanMode::Forbidden, borders, nan_));
    }
}
