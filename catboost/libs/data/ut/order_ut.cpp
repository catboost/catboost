#include <catboost/libs/data/order.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(EObjectsOrder) {
    Y_UNIT_TEST(Combine) {
        UNIT_ASSERT_VALUES_EQUAL(
            Combine(EObjectsOrder::Ordered, EObjectsOrder::Ordered),
            EObjectsOrder::Ordered
        );
        UNIT_ASSERT_VALUES_EQUAL(
            Combine(EObjectsOrder::Ordered, EObjectsOrder::RandomShuffled),
            EObjectsOrder::RandomShuffled
        );
        UNIT_ASSERT_VALUES_EQUAL(
            Combine(EObjectsOrder::Ordered, EObjectsOrder::Undefined),
            EObjectsOrder::Undefined
        );
        UNIT_ASSERT_VALUES_EQUAL(
            Combine(EObjectsOrder::RandomShuffled, EObjectsOrder::Ordered),
            EObjectsOrder::RandomShuffled
        );
        UNIT_ASSERT_VALUES_EQUAL(
            Combine(EObjectsOrder::RandomShuffled, EObjectsOrder::RandomShuffled),
            EObjectsOrder::RandomShuffled
        );
        UNIT_ASSERT_VALUES_EQUAL(
            Combine(EObjectsOrder::RandomShuffled, EObjectsOrder::Undefined),
            EObjectsOrder::RandomShuffled
        );
        UNIT_ASSERT_VALUES_EQUAL(
            Combine(EObjectsOrder::Undefined, EObjectsOrder::Ordered),
            EObjectsOrder::Undefined
        );
        UNIT_ASSERT_VALUES_EQUAL(
            Combine(EObjectsOrder::Undefined, EObjectsOrder::RandomShuffled),
            EObjectsOrder::RandomShuffled
        );
        UNIT_ASSERT_VALUES_EQUAL(
            Combine(EObjectsOrder::Undefined, EObjectsOrder::Undefined),
            EObjectsOrder::Undefined
        );
    }
}

