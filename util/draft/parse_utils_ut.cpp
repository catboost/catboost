#include <library/unittest/registar.h>

#include "parse_utils.h"

class TUtilDraftParseUtilsTest: public TTestBase {
    UNIT_TEST_SUITE(TUtilDraftParseUtilsTest);
    UNIT_TEST(testAddToNameValueHash);
    UNIT_TEST(testIsNonBMPUTF8);
    UNIT_TEST(testToLowerUTF8);
    UNIT_TEST_SUITE_END();

private:
    template <class T>
    static void CheckHash(const yhash<TString, T>& hm, const char* k, const T* v);

public:
    void testAddToNameValueHash();
    void testIsNonBMPUTF8();
    void testToLowerUTF8();
};

UNIT_TEST_SUITE_REGISTRATION(TUtilDraftParseUtilsTest);

template <class T>
void TUtilDraftParseUtilsTest::CheckHash(const yhash<TString, T>& hm, const char* k, const T* v) {
    typename yhash<TString, T>::const_iterator it = hm.find(k);
    if (v) {
        UNIT_ASSERT(it != hm.end());
        UNIT_ASSERT_EQUAL(it->second, *v);
    } else {
        UNIT_ASSERT(it == hm.end());
    }
}

void TUtilDraftParseUtilsTest::testAddToNameValueHash() {
    const char* nvString = "a=4\tb=0.03\tc=12\trt=0.23";

    yhash<TString, float> nvMap;
    AddToNameValueHash(nvString, nvMap);

    float v;
    v = 4.f;
    CheckHash(nvMap, "a", &v);
    v = 0.03f;
    CheckHash(nvMap, "b", &v);
    v = 12.f;
    CheckHash(nvMap, "c", &v);
    v = 0.23f;
    CheckHash(nvMap, "rt", &v);
    CheckHash(nvMap, "d", (const float*)nullptr);
    CheckHash(nvMap, "rtd", (const float*)nullptr);
}

void TUtilDraftParseUtilsTest::testIsNonBMPUTF8() {
    const char* query1 = "utf\x20\xd0\xb7\xd0\xb0\xd0\xbf\xd1\x80\xd0\xbe\xd1\x81";
    UNIT_ASSERT(!IsNonBMPUTF8(query1));

    const char* query2 = "\xf4\x80\x89\x84\xf4\x80\x89\x87\xf4\x80\x88\xba";
    UNIT_ASSERT(IsNonBMPUTF8(query2));
}

void TUtilDraftParseUtilsTest::testToLowerUTF8() {
    TString query1 = "\xd0\x97\xd0\xb0\xd0\xbf\xd1\x80\xd0\xbe\xd1\x81\x20\x55\x54\x46\x38\x20\xd1\x81\xd0\xbe"
                     "\x20\xd0\xa1\xd0\xbc\xd0\xb5\xd0\xa8\xd0\xb0\xd0\xbd\xd0\xbd\xd1\x8b\xd0\xbc"
                     "\x20\xd1\x80\xd0\xb5\xd0\xb3\xd0\xb8\xd1\x81\xd1\x82\xd1\x80\xd0\xbe\xd0\xbc";
    UNIT_ASSERT_EQUAL(ToLowerUTF8(query1), "\xd0\xb7\xd0\xb0\xd0\xbf\xd1\x80\xd0\xbe\xd1\x81\x20\x75\x74\x66\x38\x20\xd1\x81\xd0\xbe"
                                           "\x20\xd1\x81\xd0\xbc\xd0\xb5\xd1\x88\xd0\xb0\xd0\xbd\xd0\xbd\xd1\x8b\xd0\xbc"
                                           "\x20\xd1\x80\xd0\xb5\xd0\xb3\xd0\xb8\xd1\x81\xd1\x82\xd1\x80\xd0\xbe\xd0\xbc");
}
