#include <library/cpp/testing/unittest/registar.h>
#include <catboost/private/libs/algo_helpers/quantile_selection.h>

Y_UNIT_TEST_SUITE(TQuantileSelectionTest) {
    Y_UNIT_TEST(QuantileSelectionFirst) {
        UNIT_ASSERT_DOUBLES_EQUAL(select_quantile({1., 2.}, {-1., 0., 1.}, 0.5), -1., 1e-6);
    }
    Y_UNIT_TEST(QuantileSelectionSecond) {
        UNIT_ASSERT_DOUBLES_EQUAL(select_quantile({1., 2.}, {-1., 0., 1.}, 1.5), 0., 1e-6);
    }
    Y_UNIT_TEST(QuantileSelectionThird) {
        UNIT_ASSERT_DOUBLES_EQUAL(select_quantile({1., 2.}, {-1., 0., 1.}, 2.5), 1., 1e-6);
    }
    Y_UNIT_TEST(QuantileSelectionOnBoundary) {
        UNIT_ASSERT_DOUBLES_EQUAL(select_quantile({1., 2.}, {-1., 0., 1.}, 1.0), -1., 1e-6);
    }
    Y_UNIT_TEST(QuantileSelectionSingleSegment) {
        UNIT_ASSERT_DOUBLES_EQUAL(select_quantile({}, {0.5}, 42.0), 0.5, 1e-6);
    }
    Y_UNIT_TEST(QuantileSelectionFourSegments) {
        UNIT_ASSERT_DOUBLES_EQUAL(select_quantile({1., 2., 3.}, {0.1, 0.2, 0.3, 0.4}, 2.5), 0.3, 1e-6);
    }
}
