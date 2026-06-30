#include <library/cpp/case_insensitive_string/case_insensitive_string.h>
#include <library/cpp/case_insensitive_string/ut_gtest/util/locale_guard.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <util/generic/string.h>

TEST(CaseInsensitiveString, CompareAscii) {
    {
        TCaseInsensitiveStringBuf s1 = "Some Text";
        TCaseInsensitiveStringBuf s2 = "somE texT";
        EXPECT_EQ(s1, s2);
    }
    {
        TCaseInsensitiveStringBuf s1 = "aBCd";
        TCaseInsensitiveStringBuf s2 = "AbcE";
        EXPECT_LT(s1, s2);
    }
    {
        // Also works with null bytes
        TCaseInsensitiveStringBuf s1("aBC\0d", 5);
        TCaseInsensitiveStringBuf s2("Abc\0E", 5);
        EXPECT_LT(s1, s2);
    }
}

TEST(CaseInsensitiveString, CompareLocaleDependent) {
    TLocaleGuard loc("ru_RU.CP1251");
    if (loc.Error()) {
        GTEST_SKIP() << "ru_RU.CP1251 locale is not available: " << loc.Error();
    }
    {
        TCaseInsensitiveStringBuf s1 = "\xc0\xc1\xc2";  // "АБВ"
        TCaseInsensitiveStringBuf s2 = "\xe0\xe1\xe2";  // "абв"
        EXPECT_EQ(s1, s2);
    }
    {
        TCaseInsensitiveStringBuf s1 = "\xc0\xc1\xc3";  // "АБГ"
        TCaseInsensitiveStringBuf s2 = "\xe0\xe1\xe2";  // "абв"
        EXPECT_GT(s1, s2);
    }
}

TEST(CaseInsensitiveAsciiString, CompareAsciiWithoutNullBytes) {
    {
        TCaseInsensitiveAsciiStringBuf s1 = "Some Text";
        TCaseInsensitiveAsciiStringBuf s2 = "somE texT";
        EXPECT_EQ(s1, s2);
    }
    {
        TCaseInsensitiveAsciiStringBuf s1 = "aBCd";
        TCaseInsensitiveAsciiStringBuf s2 = "AbcE";
        EXPECT_LT(s1, s2);
    }
}

TEST(CaseInsensitiveAsciiString, UnspecifiedBehaviorForNonAsciiStrings) {
    TLocaleGuard loc("ru_RU.CP1251");
    if (loc.Error()) {
        GTEST_SKIP() << "ru_RU.CP1251 locale is not available: " << loc.Error();
    }
    // - strncasecmp in glibc is locale-dependent.
    // - not sure about strnicmp (probably also locale-dependent).
    // - strncasecmp in musl and bionic ignores locales.
    // - ASan interceptor for strncasecmp ignores locales
    {
        TCaseInsensitiveAsciiStringBuf s1 = "\xc0\xc1\xc2";  // "АБВ"
        TCaseInsensitiveAsciiStringBuf s2 = "\xe0\xe1\xe2";  // "абв"
        EXPECT_TRUE(s1 == s2 || s1 < s2);
    }
    {
        TCaseInsensitiveAsciiStringBuf s1 = "\xc0\xc1\xc3";  // "АБГ"
        TCaseInsensitiveAsciiStringBuf s2 = "\xe0\xe1\xe2";  // "абв"
        EXPECT_TRUE(s1 > s2 || s1 < s2);
    }
}

TEST(CaseInsensitiveAsciiString, DoesNotWorkWithNullBytes) {
    TCaseInsensitiveAsciiStringBuf s1("aBC\0d", 5);
    TCaseInsensitiveAsciiStringBuf s2("Abc\0E", 5);
    EXPECT_EQ(s1, s2);
}
