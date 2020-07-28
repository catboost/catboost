#include <library/cpp/testing/unittest/registar.h>

#include <util/charset/wide.h>

#include <library/cpp/unicode/normalization/normalization.h>

Y_UNIT_TEST_SUITE(TUnicodeNormalizationTest) {
    template <NUnicode::ENormalization NormType>
    void TestInit() {
        NUnicode::TNormalizer<NormType> normalizer;
        TString s("упячка detected");
        TUtf16String w;
        UTF8ToWide(s, w);
        normalizer.Normalize(w);
    }

    Y_UNIT_TEST(TestInitNFD) {
        TestInit<NUnicode::NFD>();
    }

    Y_UNIT_TEST(TestInitNFC) {
        TestInit<NUnicode::NFC>();
    }

    Y_UNIT_TEST(TestInitNFKD) {
        TestInit<NUnicode::NFKD>();
    }

    Y_UNIT_TEST(TestInitNFKC) {
        TestInit<NUnicode::NFKC>();
    }
}
