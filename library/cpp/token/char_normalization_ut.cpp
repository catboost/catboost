#include <library/cpp/unittest/registar.h>
#include <util/charset/wide.h>
#include "charfilter.h"

class TCharNormalizationTest: public TTestBase {
    UNIT_TEST_SUITE(TCharNormalizationTest);
    UNIT_TEST(TestAll);
    UNIT_TEST_SUITE_END();

public:
    void TestAll();

private:
    static bool CheckEq(const char* mod, const TUtf16String& s);
    static bool TestNormalization(const char* orig,
                                  const char* newNoG,
                                  const char* oldNoG,
                                  const char* newG,
                                  const char* oldG);

    static bool TestPolish();
    static bool TestNorwegian();
    static bool TestGerman();
    static bool TestRUB();
    static bool TestKazakh();
};

UNIT_TEST_SUITE_REGISTRATION(TCharNormalizationTest);

static TUtf16String ctw(const char* s) {
    Y_ASSERT(s);
    return UTF8ToWide<true>(s, strlen(s));
}

bool TCharNormalizationTest::CheckEq(const char* mod, const TUtf16String& s) {
    const TUtf16String sMod = ctw(mod);
    if (sMod == s)
        return true;

    Cerr << "\n"
         << "EXPECTED:"
         << "\n"
         << "\t" << mod << "\n"
         << "RECEIVED:"
         << "\n"
         << "\t" << s << Endl;
    return false;
}

bool TCharNormalizationTest::TestNormalization(const char* orig,
                                               const char* newNoG, const char* oldNoG,
                                               const char* newG, const char* oldG) {
    const TUtf16String sOrig = ctw(orig);
    return CheckEq(newNoG, NormalizeUnicode(sOrig, false, true)) && CheckEq(oldNoG, NormalizeUnicode(sOrig, false, false)) && CheckEq(newG, NormalizeUnicode(sOrig, true, true)) && CheckEq(oldG, NormalizeUnicode(sOrig, true, false));
}

bool TCharNormalizationTest::TestPolish() {
    const char* orig = "Wisła najdłuższa rzeka Polski";
    const char* dNew = "wisla najdluzsza rzeka polski";
    const char* dOld = "wisła najdłuzsza rzeka polski";
    return TestNormalization(orig, dNew, dOld, dNew, dOld);
}

bool TCharNormalizationTest::TestNorwegian() {
    const char* orig = "Bjørn Erlend Dæhlie";
    const char* dNew = "bjorn erlend daehlie";
    const char* dOld = "bjørn erlend dæhlie";
    return TestNormalization(orig, dNew, dOld, dNew, dOld);
}

bool TCharNormalizationTest::TestGerman() {
    const char* orig = "er läßt Sie grüßen";
    const char* dNoG = "er lasst sie grussen";
    const char* dG = "er laesst sie gruessen";
    return TestNormalization(orig, dNoG, dNoG, dG, dG);
}

bool TCharNormalizationTest::TestRUB() {
    const char* orig = "Зялёны воўк и хімічний європейський їжак";
    const char* d = "зялены воук и хімічний європейський їжак";
    return TestNormalization(orig, d, d, d, d);
}

bool TCharNormalizationTest::TestKazakh() {
    const char* orig = "Қазақстан, Өзбекстан, Қытай, Моңғолия";
    const char* dNew = "казакстан, озбекстан, кытай, монголия";
    const char* dOld = "қазақстан, өзбекстан, қытай, моңғолия";
    return TestNormalization(orig, dNew, dOld, dNew, dOld);
}

void TCharNormalizationTest::TestAll() {
    UNIT_ASSERT(TestPolish());
    UNIT_ASSERT(TestNorwegian());
    UNIT_ASSERT(TestGerman());
    UNIT_ASSERT(TestRUB());
    UNIT_ASSERT(TestKazakh());
}
