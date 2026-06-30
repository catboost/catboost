#include "ztstrbuf.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TZtStringBufTest) {
    Y_UNIT_TEST(EmptyString) {
        TZtStringBuf s0{};
        UNIT_ASSERT_VALUES_EQUAL(s0, TString{""});
        UNIT_ASSERT_VALUES_EQUAL(s0.c_str(), TString{""});
    }

    Y_UNIT_TEST(Constness) {
        constexpr TZtStringBuf s0{"bar"};
        static_assert(s0[0] == 'b');
        static_assert(s0.data()[s0.size()] == '\0');
        static_assert(s0.data()[2] == 'r');
        UNIT_ASSERT_VALUES_EQUAL(s0, TString{"bar"});
    }

    Y_UNIT_TEST(FromString) {
        TString str0{"foo"};
        TZtStringBuf s0 = str0;
        UNIT_ASSERT_VALUES_EQUAL(s0, "foo");
        std::string str1{"bar"};
        TZtStringBuf s1 = str1;
        UNIT_ASSERT_VALUES_EQUAL(s1, "bar");
    }
}
