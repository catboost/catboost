#include "enum_cast.h"

#include "enum_cast_ut.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestEnumCast) {
    Y_UNIT_TEST(SafeCastToEnumTest) {
        UNIT_ASSERT_VALUES_EQUAL(SafeCastToEnum<EIntEnum>(0), EIntEnum::Zero);
        UNIT_ASSERT_VALUES_EQUAL(SafeCastToEnum<EIntEnum>(1), EIntEnum::One);
        UNIT_ASSERT_VALUES_EQUAL(SafeCastToEnum<EIntEnum>(2), EIntEnum::Two);
        UNIT_ASSERT_EXCEPTION(SafeCastToEnum<EIntEnum>(3), TBadCastException);

        UNIT_ASSERT_VALUES_EQUAL(SafeCastToEnum<EUcharEnum>(0), EUcharEnum::Zero);
        UNIT_ASSERT_VALUES_EQUAL(SafeCastToEnum<EUcharEnum>(1), EUcharEnum::One);
        UNIT_ASSERT_VALUES_EQUAL(SafeCastToEnum<EUcharEnum>(2), EUcharEnum::Two);
        UNIT_ASSERT_EXCEPTION_CONTAINS(
            SafeCastToEnum<EUcharEnum>(3), TBadCastException,
            "Unexpected enum");
        int val1 = 256;
        UNIT_ASSERT_EXCEPTION_CONTAINS(
            SafeCastToEnum<EUcharEnum>(val1), TBadCastException,
            "Unexpected enum");
        int val2 = -1;
        UNIT_ASSERT_EXCEPTION_CONTAINS(
            SafeCastToEnum<EUcharEnum>(val2), TBadCastException,
            "Unexpected enum");
        int val3 = 2;
        UNIT_ASSERT_VALUES_EQUAL(SafeCastToEnum<EUcharEnum>(val3), EUcharEnum::Two);

        UNIT_ASSERT_VALUES_EQUAL(SafeCastToEnum<EBoolEnum>(false), EBoolEnum::False);
        UNIT_ASSERT_VALUES_EQUAL(SafeCastToEnum<EBoolEnum>(true), EBoolEnum::True);

        UNIT_ASSERT_VALUES_EQUAL(SafeCastToEnum<EUnscopedIntEnum>(2), UIE_TWO);
        UNIT_ASSERT_EXCEPTION_CONTAINS(
            SafeCastToEnum<EUnscopedIntEnum>(3), TBadCastException,
            "Unexpected enum");
        UNIT_ASSERT_EXCEPTION_CONTAINS(
            SafeCastToEnum<EUnscopedIntEnum>(9), TBadCastException,
            "Unexpected enum");

        UNIT_ASSERT_VALUES_EQUAL(SafeCastToEnum<ECharEnum>(static_cast<unsigned int>(0)), ECharEnum::Zero);
        UNIT_ASSERT_VALUES_EQUAL(SafeCastToEnum<ECharEnum>(static_cast<short>(-1)), ECharEnum::MinusOne);
        UNIT_ASSERT_VALUES_EQUAL(SafeCastToEnum<ECharEnum>(static_cast<int>(-2)), ECharEnum::MinusTwo);
        UNIT_ASSERT_EXCEPTION_CONTAINS(
            SafeCastToEnum<ECharEnum>(static_cast<int>(2)), TBadCastException,
            "Unexpected enum");
    }
} // Y_UNIT_TEST_SUITE(TestEnumCast)
