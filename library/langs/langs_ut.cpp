#include "langs.h"

#include <library/cpp/unittest/registar.h>

#include <util/system/yassert.h>

class TDocCodesTest: public TTestBase {
private:
    UNIT_TEST_SUITE(TDocCodesTest);
    UNIT_TEST(TestNames);
    UNIT_TEST_SUITE_END();

public:
    void TestNames();
    void TestSimpleExamples();

private:
    void TestName(ELanguage language, const char* name) {
        ELanguage reversed = LanguageByName(name);
        UNIT_ASSERT(language == reversed);
        ELanguage reversedStrict = LanguageByNameStrict(name);
        UNIT_ASSERT(language == reversedStrict);
        ELanguage reversedOrDie = LanguageByNameOrDie(name);
        UNIT_ASSERT(language == reversedOrDie);
    }

    void TestWrongName(const char* name) {
        ELanguage reversed = LanguageByName(name);
        UNIT_ASSERT(reversed == LANG_UNK);
        ELanguage reversedStrict = LanguageByNameStrict(name);
        UNIT_ASSERT(reversedStrict == LANG_MAX);
        UNIT_ASSERT_EXCEPTION(LanguageByNameOrDie(name), yexception);
    }

    void TestBiblioName(ELanguage language) {
        const char* name = NameByLanguage(language);

        UNIT_ASSERT(name != nullptr);
        UNIT_ASSERT(strlen(name) > 2);

        TestName(language, name);
    }

    void TestIsoName(ELanguage language) {
        const char* name = IsoNameByLanguage(language);

        UNIT_ASSERT(name != nullptr);
        UNIT_ASSERT(strlen(name) == 0 || strlen(name) == 2 ||
                    !strcmp(name, "mis") || !strcmp(name, "udm") ||
                    !strcmp(name, "mrj") || !strcmp(name, "mhr") ||
                    !strcmp(name, "sjn") || !strcmp(name, "ceb") ||
                    !strcmp(name, "koi") || !strcmp(name, "pap") ||
                    !strcmp(name, "sah") || !strncmp(name, "bas-", 4) ||
                    !strcmp(name, "uz-Cyrl") || !strcmp(name, "kk-Latn") ||
                    !strcmp(name, "tr-ipa") || !strcmp(name, "emj"));

        if (strlen(name))
            TestName(language, name);
    }

    void TestFullName(ELanguage language) {
        const char* name = FullNameByLanguage(language);
        UNIT_ASSERT(name != nullptr);
        UNIT_ASSERT(strlen(name) >= 3); // The shortest language name is "Lao"

        if (strlen(name))
            TestName(language, name);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TDocCodesTest);

void TDocCodesTest::TestNames() {
    TestWrongName(nullptr);
    TestWrongName("");
    TestWrongName("A wrong language name");
    TestWrongName("cockney-en");

    for (size_t i = 0; i != LANG_MAX; ++i) {
        ELanguage language = static_cast<ELanguage>(i);
        TestBiblioName(language);
        TestIsoName(language);
        TestFullName(language);
    }

    TestName(LANG_RUS, "ru_RU");
    TestName(LANG_ENG, "en-cockney");
    TestName(LANG_CHI, "zh-Hant_HK");
}

void TDocCodesTest::TestSimpleExamples() {
    UNIT_ASSERT(NameByLanguage(LANG_RUS) == TString("rus"));
    UNIT_ASSERT(IsoNameByLanguage(LANG_RUS) == TString("ru"));
    UNIT_ASSERT(FullNameByLanguage(LANG_RUS) == TString("Russian"));
}
