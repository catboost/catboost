#include "tempbuf.h"

#include <utility>

#include <library/cpp/testing/unittest/registar.h>

class TTempBufTest: public TTestBase {
    UNIT_TEST_SUITE(TTempBufTest);
    UNIT_TEST(TestCreate);
    UNIT_TEST(TestOps);
    UNIT_TEST(TestMoveCtor);
    UNIT_TEST(TestAppend);
    UNIT_TEST(TestProceed);
    UNIT_TEST_SUITE_END();

public:
    void TestCreate();
    void TestOps();
    void TestMoveCtor();
    void TestProceed();

    void TestAppend() {
        TTempBuf tmp;

        tmp.Append("a", 1);
        tmp.Append("bc", 2);
        tmp.Append("def", 3);

        UNIT_ASSERT_EQUAL(tmp.Filled(), 6);
        UNIT_ASSERT_EQUAL(TString(tmp.Data(), tmp.Filled()), "abcdef");
    }
};

UNIT_TEST_SUITE_REGISTRATION(TTempBufTest);

void TTempBufTest::TestCreate() {
    const size_t num = 1000000;
    size_t tmp = 0;
    const size_t len = 4096;

    for (size_t i = 0; i < num; ++i) {
        TTempBuf buf(len);

        tmp += (size_t)buf.Data();
    }

    UNIT_ASSERT(tmp != 0);
}

void TTempBufTest::TestOps() {
    TTempBuf tmp(201);

    tmp.Proceed(100);

    UNIT_ASSERT_EQUAL(tmp.Current() - tmp.Data(), 100);
    UNIT_ASSERT(tmp.Left() >= 101);
    UNIT_ASSERT(tmp.Size() >= 201);
    UNIT_ASSERT_EQUAL(tmp.Filled(), 100);

    tmp.Reset();

    UNIT_ASSERT_EQUAL(tmp.Current(), tmp.Data());
    UNIT_ASSERT(tmp.Left() >= 201);
    UNIT_ASSERT(tmp.Size() >= 201);
    UNIT_ASSERT_EQUAL(tmp.Filled(), 0);
}

void TTempBufTest::TestMoveCtor() {
    TTempBuf src;
    UNIT_ASSERT(!src.IsNull());

    src.Proceed(10);

    TTempBuf dst(std::move(src));

    UNIT_ASSERT(src.IsNull());
    UNIT_ASSERT(!dst.IsNull());
    UNIT_ASSERT_EQUAL(dst.Filled(), 10);
}

void TTempBufTest::TestProceed() {
    TTempBuf src;

    char* data = src.Proceed(100);
    UNIT_ASSERT_EQUAL(data, src.Data());
    UNIT_ASSERT_EQUAL(data + 100, src.Current());
    UNIT_ASSERT_EQUAL(100, src.Filled());

    char* second = src.Proceed(100);
    UNIT_ASSERT_EQUAL(data + 100, second);
    UNIT_ASSERT_EQUAL(data + 200, src.Current());
    UNIT_ASSERT_EQUAL(200, src.Filled());
}
