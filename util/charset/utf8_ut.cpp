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
            const char* turkI = "İ";        //strlen("İ") == 2, strlen(tolower_utf8("İ") == 1
            TStringBuf chars[] = {"f", "F", "Б", "б", weird, turkI};
            const int N = Y_ARRAY_SIZE(chars);
            //try all combinations of these letters.
            int numberOfVariants = 1;
            for (int len = 0; len <= 4; ++len) {
                for (int i = 0; i < numberOfVariants; ++i) {
                    TString s;
                    int k = i;
                    for (int j = 0; j < len; ++j) {
                        //Treat 'i' like number in base-N system with digits from 'chars'-array
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
            const char* turkI = "İ";        //strlen("İ") == 2, strlen(ToUpper_utf8("İ") == 1
            TStringBuf chars[] = {"F", "f", "б", "Б", turkI, weird};
            const int N = Y_ARRAY_SIZE(chars);
            //try all combinations of these letters.
            int numberOfVariants = 1;
            for (int len = 0; len <= 4; ++len) {
                for (int i = 0; i < numberOfVariants; ++i) {
                    TString s;
                    int k = i;
                    for (int j = 0; j < len; ++j) {
                        //Treat 'i' like number in base-N system with digits from 'chars'-array
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
}
