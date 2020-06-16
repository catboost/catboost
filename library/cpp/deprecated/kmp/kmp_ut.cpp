#include "kmp.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/output.h>

static TVector<int> FindAll(const TString& pattern, const TString& string) {
    TVector<int> result;
    TKMPMatcher kmp(pattern);
    const char* pResult;
    const char* begin = string.begin();
    const char* end = string.end();
    while (kmp.SubStr(begin, end, pResult)) {
        result.push_back(int(pResult - string.data()));
        begin = pResult + pattern.size();
    }
    return result;
}

class TTestKMP: public TTestBase {
    UNIT_TEST_SUITE(TTestKMP);
    UNIT_TEST(Test);
    UNIT_TEST(TestStream);
    UNIT_TEST_SUITE_END();

public:
    void Test() {
        TVector<int> ans = {0, 2};
        UNIT_ASSERT_EQUAL(FindAll("a", "aba"), ans);
        ans = {0};
        UNIT_ASSERT_EQUAL(FindAll("aba", "aba"), ans);
        ans.clear();
        UNIT_ASSERT_EQUAL(FindAll("abad", "aba"), ans);
        ans = {0, 2};
        UNIT_ASSERT_EQUAL(FindAll("ab", "abab"), ans);
    }

    class TKMPSimpleCallback: public TKMPStreamMatcher<int>::ICallback {
    private:
        int* Begin;
        int* End;
        int Count;

    public:
        TKMPSimpleCallback(int* begin, int* end)
            : Begin(begin)
            , End(end)
            , Count(0)
        {
        }

        void OnMatch(const int* begin, const int* end) override {
            UNIT_ASSERT_EQUAL(end - begin, End - Begin);
            const int* p0 = Begin;
            const int* p1 = begin;
            while (p0 < End) {
                UNIT_ASSERT_EQUAL(*p0, *p1);
                ++p0;
                ++p1;
            }
            ++Count;
        }

        int GetCount() const {
            return Count;
        }
    };

    void TestStream() {
        int pattern[] = {2, 3};
        int data[] = {1, 2, 3, 5, 2, 2, 3, 2, 4, 3, 2};
        TKMPSimpleCallback callback(pattern, pattern + 2);
        TKMPStreamMatcher<int> matcher(pattern, pattern + 2, &callback);
        for (auto& i : data)
            matcher.Push(i);
        UNIT_ASSERT_EQUAL(2, callback.GetCount());
    }
};

UNIT_TEST_SUITE_REGISTRATION(TTestKMP);
