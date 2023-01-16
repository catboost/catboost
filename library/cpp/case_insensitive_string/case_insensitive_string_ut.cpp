#include "case_insensitive_string.h"

#include <util/generic/string_ut.h>

class TCaseInsensitiveStringTest : public TTestBase, private TStringTestImpl<TCaseInsensitiveString, TTestData<char>> {
public:
    UNIT_TEST_SUITE(TCaseInsensitiveStringTest);
    UNIT_TEST(TestOperators);
    UNIT_TEST(TestOperatorsCI);

    UNIT_TEST_SUITE_END();
};

UNIT_TEST_SUITE_REGISTRATION(TCaseInsensitiveStringTest);

Y_UNIT_TEST_SUITE(TCaseInsensitiveStringTestEx) {
    Y_UNIT_TEST(BasicTString) {
        TCaseInsensitiveString foo("foo");
        TCaseInsensitiveString FOO("FOO");
        TCaseInsensitiveString Bar("Bar");
        TCaseInsensitiveString bAR("bAR");

        UNIT_ASSERT_EQUAL(foo, FOO);
        UNIT_ASSERT_EQUAL(Bar, bAR);

        constexpr TCaseInsensitiveStringBuf foobar("foobar");
        UNIT_ASSERT(foobar.StartsWith(foo));
        UNIT_ASSERT(foobar.StartsWith(FOO));
        UNIT_ASSERT(foobar.EndsWith(Bar));
        UNIT_ASSERT(foobar.EndsWith(bAR));
        UNIT_ASSERT(foobar.Contains(FOO));
        UNIT_ASSERT(foobar.Contains(Bar));
    }

    Y_UNIT_TEST(BasicStdString) {
        using TCaseInsensitiveStdString = std::basic_string<char, TCaseInsensitiveCharTraits>;
        using TCaseInsensitiveStringView = std::basic_string_view<char, TCaseInsensitiveCharTraits>;

        TCaseInsensitiveStdString foo("foo");
        TCaseInsensitiveStdString FOO("FOO");
        TCaseInsensitiveStdString Bar("Bar");
        TCaseInsensitiveStdString bAR("bAR");

        UNIT_ASSERT_EQUAL(foo, FOO);
        UNIT_ASSERT_EQUAL(Bar, bAR);

        constexpr TCaseInsensitiveStringView foobar("foobar");
        UNIT_ASSERT(foobar.starts_with(foo));
        UNIT_ASSERT(foobar.starts_with(FOO));
        UNIT_ASSERT(foobar.ends_with(Bar));
        UNIT_ASSERT(foobar.ends_with(bAR));
        //TODO: test contains after C++23
    }

/*
    Y_UNIT_TEST(TestSplit) {
        TCaseInsensitiveStringBuf input("splitAmeAbro");
        TVector<TCaseInsensitiveStringBuf> expected{"split", "me", "bro"};

        TVector<TCaseInsensitiveStringBuf> split = StringSplitter(input).Split('a');

        UNIT_ASSERT_VALUES_EQUAL(split, expected);
    }
*/
}
