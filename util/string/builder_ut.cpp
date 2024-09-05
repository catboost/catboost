#include "builder.h"

#include <library/cpp/testing/unittest/registar.h>

static void TestEquals(const TString& expected, const TString& actual) {
    UNIT_ASSERT_VALUES_EQUAL(expected, actual);
}

struct TClassWithStreamOperator {
    ui32 Id;
    TString Name;

    TClassWithStreamOperator(ui32 id, const TString& name)
        : Id(id)
        , Name(name)
    {
    }
};

IOutputStream& operator<<(IOutputStream& out, const TClassWithStreamOperator& value) {
    return out << value.Id << " " << value.Name;
}

Y_UNIT_TEST_SUITE(TStringBuilderTest) {
    Y_UNIT_TEST(TestStringBuilder) {
        TestEquals("", TStringBuilder());
        TestEquals("a", TStringBuilder() << "a");
        TestEquals("a1", TStringBuilder() << "a" << 1);
        TestEquals("value: 123 name", TStringBuilder() << "value: " << TClassWithStreamOperator(123, "name"));
    }

    Y_UNIT_TEST(TestStringBuilderOut) {
        TString s;
        TStringOutput out(s);
        TStringBuilder sb;
        sb << "a";
        out << sb;
        TestEquals("a", s);
    }

    Y_UNIT_TEST(TestStringBuilderRValue) {
        struct TRValueAcceptTester {
            static bool IsRValue(const TString&) {
                return false;
            }

            static bool IsRValue(TString&&) {
                return true;
            }
        };

        UNIT_ASSERT(TRValueAcceptTester::IsRValue(TStringBuilder() << "a" << 1));

        TStringBuilder b;
        UNIT_ASSERT(!TRValueAcceptTester::IsRValue(b << "a" << 1));
        TStringBuilder b2;
        UNIT_ASSERT(!TRValueAcceptTester::IsRValue(b2 << "a" << 1 << TStringBuilder() << "a"));
        UNIT_ASSERT_VALUES_EQUAL("a1a", b2);

        UNIT_ASSERT(TRValueAcceptTester::IsRValue(TStringBuilder() << b2));
        UNIT_ASSERT_VALUES_EQUAL("a1a", TStringBuilder() << b2);
    }
} // Y_UNIT_TEST_SUITE(TStringBuilderTest)
