#include <library/unittest/registar.h>

#include <library/malloc/api/malloc.h>

SIMPLE_UNIT_TEST_SUITE(MallocApi) {
    SIMPLE_UNIT_TEST(ToStream) {
        TStringStream ss;
        ss << NMalloc::MallocInfo();
    }
}
