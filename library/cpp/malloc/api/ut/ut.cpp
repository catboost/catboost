#include <library/cpp/testing/unittest/registar.h>

#include <library/cpp/malloc/api/malloc.h>

Y_UNIT_TEST_SUITE(MallocApi) {
    Y_UNIT_TEST(ToStream) {
        TStringStream ss;
        ss << NMalloc::MallocInfo();
    }
}
