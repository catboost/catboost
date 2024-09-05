#undef NDEBUG
// yassert.h must be included before all headers
#include "yassert.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(YassertTest) {
    Y_UNIT_TEST(TestAcsLikeFunctionCall) {
        if (true) {
            Y_ASSERT(true); // this cannot be compiled if Y_ASSERT is "if (!cond) { ... }"
        } else {
            Y_ASSERT(false);
        }

        bool var = false;
        if (false) {
            Y_ASSERT(false);
        } else {
            var = true; // this is unreachable if Y_ASSERT is "if (!cond) { ... }"
        }
        UNIT_ASSERT(var);
    }

    Y_UNIT_TEST(TestFailCompiles) {
        if (false) {
            Y_ABORT("%d is a lucky number", 7);
            Y_ABORT();
        }
    }

    Y_UNIT_TEST(TestVerify) {
        Y_ABORT_UNLESS(true, "hi %s", "there");
        Y_ABORT_UNLESS(true);
    }

    Y_UNIT_TEST(TestExceptionVerify) {
        UNIT_ASSERT_EXCEPTION(
            []() { Y_ABORT_UNLESS([]() {throw yexception{} << "check"; return false; }(), "hi %s", "there"); }(),
            yexception);
    }
} // Y_UNIT_TEST_SUITE(YassertTest)
