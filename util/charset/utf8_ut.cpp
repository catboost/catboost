#include "utf8.h"
#include "wide.h"

#include <util/stream/file.h>
#include <util/ysaveload.h>

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/env.h>

Y_UNIT_TEST_SUITE(TUtfUtilTest) {
    Y_UNIT_TEST(TestUTF8Len) {
        UNIT_ASSERT_EQUAL(GetNumberOfUTF8Chars("привет!"), 7);
    }

    Y_UNIT_TEST(TestToLowerUtfString) {
        UNIT_ASSERT_VALUES_EQUAL(ToLowerUTF8("xyz XYZ ПРИВЕТ!"), "xyz xyz привет!");

        UNIT_ASSERT_VALUES_EQUAL(ToLowerUTF8(TStringBuf("xyz")), "xyz");

        {
            TString s = "привет!";
            TString q = "ПРИВЕТ!";
            TString tmp;
            UNIT_ASSERT(ToLowerUTF8Impl(s.data(), s.size(), tmp) == false);
            UNIT_ASSERT(ToLowerUTF8Impl(q.data(), q.size(), tmp) == true);
        }

        {
            const char* weird = "\xC8\xBE"; // 'Ⱦ', U+023E. strlen(weird)==2, strlen(tolower_utf8(weird)) is 3
            const char* turkI = "İ";        // strlen("İ") == 2, strlen(tolower_utf8("İ") == 1
            TStringBuf chars[] = {"f", "F", "Б", "б", weird, turkI};
            const int N = Y_ARRAY_SIZE(chars);
            // try all combinations of these letters.
            int numberOfVariants = 1;
            for (int len = 0; len <= 4; ++len) {
                for (int i = 0; i < numberOfVariants; ++i) {
                    TString s;
                    int k = i;
                    for (int j = 0; j < len; ++j) {
                        // Treat 'i' like number in base-N system with digits from 'chars'-array
                        s += chars[k % N];
                        k /= N;
                    }

                    TUtf16String tmp = UTF8ToWide(s);
                    tmp.to_lower();

                    UNIT_ASSERT_VALUES_EQUAL(ToLowerUTF8(s), WideToUTF8(tmp));
                }
                numberOfVariants *= N;
            }
        }
    }

    Y_UNIT_TEST(TestToUpperUtfString) {
        UNIT_ASSERT_VALUES_EQUAL(ToUpperUTF8("xyz XYZ привет!"), "XYZ XYZ ПРИВЕТ!");

        UNIT_ASSERT_VALUES_EQUAL(ToUpperUTF8(TStringBuf("XYZ")), "XYZ");

        {
            TString s = "ПРИВЕТ!";
            TString q = "привет!";
            TString tmp;
            UNIT_ASSERT(ToUpperUTF8Impl(s.data(), s.size(), tmp) == false);
            UNIT_ASSERT(ToUpperUTF8Impl(q.data(), q.size(), tmp) == true);
        }

        {
            const char* weird = "\xC8\xBE"; // 'Ⱦ', U+023E. strlen(weird)==2, strlen(ToUpper_utf8(weird)) is 3
            const char* turkI = "İ";        // strlen("İ") == 2, strlen(ToUpper_utf8("İ") == 1
            TStringBuf chars[] = {"F", "f", "б", "Б", turkI, weird};
            const int N = Y_ARRAY_SIZE(chars);
            // try all combinations of these letters.
            int numberOfVariants = 1;
            for (int len = 0; len <= 4; ++len) {
                for (int i = 0; i < numberOfVariants; ++i) {
                    TString s;
                    int k = i;
                    for (int j = 0; j < len; ++j) {
                        // Treat 'i' like number in base-N system with digits from 'chars'-array
                        s += chars[k % N];
                        k /= N;
                    }

                    TUtf16String tmp = UTF8ToWide(s);
                    tmp.to_upper();

                    UNIT_ASSERT_VALUES_EQUAL(ToUpperUTF8(s), WideToUTF8(tmp));
                }
                numberOfVariants *= N;
            }
        }
    }

    Y_UNIT_TEST(TestUTF8ToWide) {
        TFileInput in(ArcadiaSourceRoot() + TStringBuf("/util/charset/ut/utf8/test1.txt"));

        TString text = in.ReadAll();
        UNIT_ASSERT(WideToUTF8(UTF8ToWide(text)) == text);
    }

    Y_UNIT_TEST(TestInvalidUTF8) {
        TVector<TString> testData;
        TFileInput input(ArcadiaSourceRoot() + TStringBuf("/util/charset/ut/utf8/invalid_UTF8.bin"));
        Load(&input, testData);

        for (const auto& text : testData) {
            UNIT_ASSERT_EXCEPTION(UTF8ToWide(text), yexception);
        }
    }

    Y_UNIT_TEST(TestUTF8ToWideScalar) {
        TFileInput in(ArcadiaSourceRoot() + TStringBuf("/util/charset/ut/utf8/test1.txt"));

        TString text = in.ReadAll();
        TUtf16String wtextSSE = UTF8ToWide(text);
        TUtf16String wtextScalar = TUtf16String::Uninitialized(text.size());
        const unsigned char* textBegin = reinterpret_cast<const unsigned char*>(text.c_str());
        wchar16* wtextBegin = wtextScalar.begin();
        ::NDetail::UTF8ToWideImplScalar<false>(textBegin, textBegin + text.size(), wtextBegin);
        UNIT_ASSERT(wtextBegin == wtextScalar.begin() + wtextSSE.size());
        UNIT_ASSERT(textBegin == reinterpret_cast<const unsigned char*>(text.end()));
        wtextScalar.remove(wtextSSE.size());
        UNIT_ASSERT(wtextScalar == wtextSSE);
    }

    Y_UNIT_TEST(TestUtf8TruncateInplace) {
        TString s = "Съешь ещё этих мягких французских булок, да выпей же чаю.";
        Utf8TruncateInplace(s, 0u);
        UNIT_ASSERT_EQUAL(s, "");

        s = "Съешь ещё этих мягких французских булок, да выпей же чаю.";
        Utf8TruncateInplace(s, 10u);
        UNIT_ASSERT_EQUAL(s, "Съешь");

        s = "Съешь ещё этих мягких французских булок, да выпей же чаю.";
        TString s_copy = s;
        Utf8TruncateInplace(s, s.size());
        UNIT_ASSERT_EQUAL(s, s_copy);

        Utf8TruncateInplace(s, Max());
        UNIT_ASSERT_EQUAL(s, s_copy);
    }

    Y_UNIT_TEST(TestUtf8TruncateCorrupted) {
        const TString s = "Съешь ещё этих мягких французских булок, да выпей же чаю.";
        TStringBuf corrupted{s, 0u, 21u};
        UNIT_ASSERT_EXCEPTION_CONTAINS(Y_UNUSED(Utf8Truncate(corrupted, 21u)), yexception, "invalid UTF-8 char");
        UNIT_ASSERT_NO_EXCEPTION(Y_UNUSED(Utf8TruncateRobust(corrupted, 21u)));
        TStringBuf fixed = Utf8TruncateRobust(corrupted, 21u);
        UNIT_ASSERT_LE(fixed.size(), 21u);
        UNIT_ASSERT_EQUAL(fixed, "Съешь ещё э");
    }

    Y_UNIT_TEST(TestUtf8CutInvalidSuffixInplace) {
        TString s = "Съешь ещё этих мягких французских булок, да выпей же чаю.";
        s.resize(21);
        UNIT_ASSERT_UNEQUAL(s, "Съешь ещё э");
        Utf8TruncateInplaceRobust(s, s.size());
        UNIT_ASSERT_EQUAL(s, "Съешь ещё э");
    }

    Y_UNIT_TEST(TestUtf8CutInvalidSuffix) {
        TStringBuf sb = "Съешь ещё этих мягких французских булок, да выпей же чаю."sv;
        UNIT_ASSERT_EQUAL(Utf8TruncateRobust(sb, sb.size()), sb);
        UNIT_ASSERT_EQUAL(Utf8TruncateRobust(sb.substr(0, 21), sb.size()), "Съешь ещё э"sv);
    }

    Y_UNIT_TEST(TestUtf8TruncateCornerCases) {
        UNIT_ASSERT_EQUAL(Utf8Truncate("①②③"sv, 4).size(), 3);
        UNIT_ASSERT_VALUES_EQUAL(Utf8Truncate("foobar"sv, Max()), "foobar"sv);
    }

} // Y_UNIT_TEST_SUITE(TUtfUtilTest)
