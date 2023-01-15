#include "sanitizers.h"
#include "sys_alloc.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(Sanitizers) {
    Y_UNIT_TEST(MarkAsIntentionallyLeaked) {
        auto* p1 = new i32[100];
        NSan::MarkAsIntentionallyLeaked(p1);

        auto* p2 = y_allocate(123);
        NSan::MarkAsIntentionallyLeaked(p2);
    }

} // Y_UNIT_TEST_SUITE(Sanitizers)
