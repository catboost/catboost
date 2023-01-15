#include <contrib/libs/python/ut/lib/test.h>
#include <library/cpp/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestPy3Binding) {
    Y_UNIT_TEST(version) {
         TTestPyInvoker invoker;
         UNIT_ASSERT_EQUAL(invoker.GetVersion()[0], '2');
    }
}

