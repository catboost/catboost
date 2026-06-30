#include <catboost/cuda/cuda_lib/helpers.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/yexception.h>


using namespace NHelpers;


Y_UNIT_TEST_SUITE(TParseRangeStringTest) {

    Y_UNIT_TEST(TestBad) {
        UNIT_ASSERT_EXCEPTION(ParseRangeString("", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString(" ", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("0 ", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString(" 0", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("0 - 1", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("x", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("-3", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("x-2", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("0:0:-11", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("0:--:2", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("::", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString(":1:", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString(":1:3", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("2:4:", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("3-2", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("2:10-6:3", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("3--2", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("0-3:8-6", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("0-3:-8-6", 10), TBadArgumentException);
    }

    Y_UNIT_TEST(TestSingleId) {
        auto result = ParseRangeString("1", 100);
        UNIT_ASSERT_EQUAL(result, TSet<ui32>{1});
    }

    Y_UNIT_TEST(TestSingleIdOutOfLimit) {
        UNIT_ASSERT_EXCEPTION(ParseRangeString("20", 10), TBadArgumentException);
    }

    Y_UNIT_TEST(TestSeveralSeparateIds) {
        auto result = ParseRangeString("0:1:3", 10);
        UNIT_ASSERT_EQUAL(result, (TSet<ui32>{0, 1, 3}));
    }

    Y_UNIT_TEST(TestBadSeveralSeparateIds) {
        UNIT_ASSERT_EXCEPTION(ParseRangeString("0:-2:3", 10), TBadArgumentException);
    }

    Y_UNIT_TEST(TestRange) {
        {
            auto result = ParseRangeString("0-0", 10);
            UNIT_ASSERT_EQUAL(result, (TSet<ui32>{0}));
        }
        {
            auto result = ParseRangeString("1-2", 10);
            UNIT_ASSERT_EQUAL(result, (TSet<ui32>{1, 2}));
        }
        {
            auto result = ParseRangeString("3-6", 10);
            UNIT_ASSERT_EQUAL(result, (TSet<ui32>{3, 4, 5, 6}));
        }
    }

    Y_UNIT_TEST(TestRangeOutOfLimit) {
        UNIT_ASSERT_EXCEPTION(ParseRangeString("0-20", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("10-20", 10), TBadArgumentException);
        UNIT_ASSERT_EXCEPTION(ParseRangeString("20-23", 10), TBadArgumentException);
    }

    Y_UNIT_TEST(TestRanges) {
        {
            auto result = ParseRangeString("0-0:2-4", 10);
            UNIT_ASSERT_EQUAL(result, (TSet<ui32>{0, 2, 3, 4}));
        }
        {
            auto result = ParseRangeString("0-2:5-5", 10);
            UNIT_ASSERT_EQUAL(result, (TSet<ui32>{0, 1, 2, 5}));
        }
        {
            auto result = ParseRangeString("0-1:1-1", 10);
            UNIT_ASSERT_EQUAL(result, (TSet<ui32>{0, 1}));
        }
        {
            auto result = ParseRangeString("1-4:2-5", 10);
            UNIT_ASSERT_EQUAL(result, (TSet<ui32>{1, 2, 3, 4, 5}));
        }
        {
            auto result = ParseRangeString("1-1:1-1", 10);
            UNIT_ASSERT_EQUAL(result, (TSet<ui32>{1}));
        }
        {
            auto result = ParseRangeString("0-1:2-4:6-7", 10);
            UNIT_ASSERT_EQUAL(result, (TSet<ui32>{0, 1, 2, 3, 4, 6, 7}));
        }
        {
            auto result = ParseRangeString("6-7:2-4:0-1", 10);
            UNIT_ASSERT_EQUAL(result, (TSet<ui32>{0, 1, 2, 3, 4, 6, 7}));
        }
    }

}
