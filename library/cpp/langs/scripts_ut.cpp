#include "scripts.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/system/yassert.h>

class TScriptsTest: public TTestBase {
private:
    UNIT_TEST_SUITE(TScriptsTest);
    UNIT_TEST(TestNames);
    UNIT_TEST_SUITE_END();

public:
    void TestNames();

private:
    void TestName(EScript script, const char* name) {
        EScript reversed = ScriptByName(name);
        UNIT_ASSERT(script == reversed);
    }

    void TestWrongName(const char* name) {
        EScript reversed = ScriptByName(name);
        UNIT_ASSERT(reversed == SCRIPT_UNKNOWN);
        UNIT_ASSERT_EXCEPTION(ScriptByNameOrDie(name), yexception);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TScriptsTest);

void TScriptsTest::TestNames() {
    TestName(SCRIPT_LATIN, "Latn");
    TestName(SCRIPT_CYRILLIC, "cyrl");
    TestName(SCRIPT_HAN, "HANS");
    TestName(SCRIPT_LATIN, "latin");
    TestName(SCRIPT_CYRILLIC, "CYRILLIC");

    TestWrongName(nullptr);
    TestWrongName("");
    TestWrongName("A wrong scipt name");
    TestWrongName("English");

    // Roundtrip tests
    //
    for (size_t i = 0; i != SCRIPT_MAX; ++i) {
        EScript script = static_cast<EScript>(i);
        TString isoName = IsoNameByScript(script);
        UNIT_ASSERT(isoName.size() == 4);
        UNIT_ASSERT(ScriptByName(isoName) == script);
    }
}
