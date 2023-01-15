#include <library/cpp/unittest/registar.h>
#include <util/string/hex.h>
#include <library/langmask/langmask.h>
#include <library/langmask/serialization/langmask.h>
#include <library/langmask/proto/langmask.pb.h>

class TLangMaskTest: public TTestBase {
    UNIT_TEST_SUITE(TLangMaskTest);
    UNIT_TEST(TestSerializationText);
    UNIT_TEST(TestSerializationProto);
    UNIT_TEST(TestHumanReadableRepr);
    UNIT_TEST_SUITE_END();

public:
    void TestSerializationText();
    void TestSerializationProto();
    void TestHumanReadableRepr();
};

UNIT_TEST_SUITE_REGISTRATION(TLangMaskTest);

void TLangMaskTest::TestSerializationText() {
    const TLangMask mask1{LANG_RUS, LANG_ENG, LANG_UKR, LANG_SJN};
    const TString& str1 = mask1.ToString();
    const TLangMask mask2 = TLangMask::GetFromString(str1);
    const TString& str2 = mask2.ToString();
    UNIT_ASSERT(mask1 == mask2);
    UNIT_ASSERT(str1 == str2);
}

void TLangMaskTest::TestSerializationProto() {
    const TLangMask mask1{LANG_RUS, LANG_ENG, LANG_UKR, LANG_SJN};
    {
        NProto::TLangMask proto;
        Serialize(proto, mask1, false);
        const TLangMask mask2 = Deserialize(proto);
        UNIT_ASSERT(mask1 == mask2);
        UNIT_ASSERT(mask1.ToString() == mask2.ToString());
    }
    {
        NProto::TLangMask proto;
        Serialize(proto, mask1, true);
        const TLangMask mask2 = Deserialize(proto);
        UNIT_ASSERT(mask1 == mask2);
        UNIT_ASSERT(mask1.ToString() == mask2.ToString());
    }
}

void TLangMaskTest::TestHumanReadableRepr() {
    TLangMask mask(LANG_RUS, LANG_ENG);
    TString humanReadableMask = NLanguageMasks::ToString(mask);
    UNIT_ASSERT(humanReadableMask == "rus,eng");
}
