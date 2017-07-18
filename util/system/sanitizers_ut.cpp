#include "sanitizers.h"
#include "sys_alloc.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(Sanitizers) {
    SIMPLE_UNIT_TEST(MarkAsIntentionallyLeaked) {
        auto* p1 = new i32[100];
        NSan::MarkAsIntentionallyLeaked(p1);

        auto* p2 = y_allocate(123);
        NSan::MarkAsIntentionallyLeaked(p2);
    }

} // SIMPLE_UNIT_TEST_SUITE(Sanitizers)
