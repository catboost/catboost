#include "mem.h"
#include "multi.h"
#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestMultiInput) {
    struct TTestCase {
        TMemoryInput Input1;
        TMemoryInput Input2;
        TMultiInput MultiInput;
        TTestCase(const TStringBuf in1, const TStringBuf in2)
            : Input1(in1)
            , Input2(in2)
            , MultiInput(&Input1, &Input2)
        {
        }
        void TestReadToResult(char c, size_t expectedRetval,
                              const TString& expectedValue,
                              const TString& initValue = "") {
            TString t = initValue;
            UNIT_ASSERT_VALUES_EQUAL(MultiInput.ReadTo(t, c), expectedRetval);
            UNIT_ASSERT_VALUES_EQUAL(t, expectedValue);
        }
    };

    Y_UNIT_TEST(TestReadTo) {
        TString t;

        TTestCase simpleCase("0123456789abc", "defghijk");
        simpleCase.TestReadToResult('7', 8, "0123456");
        simpleCase.TestReadToResult('f', 8, "89abcde");
        simpleCase.TestReadToResult('z', 5, "ghijk");
    }

    Y_UNIT_TEST(TestReadToBetweenStreams) {
        TTestCase case1("0123456789abc", "defghijk");
        case1.TestReadToResult('c', 13, "0123456789ab");
        case1.TestReadToResult('k', 8, "defghij");
        case1.TestReadToResult('z', 0, "TRASH", "TRASH");

        TTestCase case2("0123456789abc", "defghijk");
        case2.TestReadToResult('d', 14, "0123456789abc");
        case2.TestReadToResult('j', 6, "efghi");
        case2.TestReadToResult('k', 1, "", "TRASH");

        TTestCase case3("0123456789abc", "defghijk");
        case3.TestReadToResult('e', 15, "0123456789abcd");
        case3.TestReadToResult('j', 5, "fghi");
        case3.TestReadToResult('k', 1, "", "TRASH");
    }
} // Y_UNIT_TEST_SUITE(TestMultiInput)
