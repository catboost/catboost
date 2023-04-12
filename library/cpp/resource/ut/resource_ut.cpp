#include <library/cpp/resource/resource.h>
#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestResource) {
    Y_UNIT_TEST(Test1) {
        UNIT_ASSERT_VALUES_EQUAL(NResource::Find("/x"), "na gorshke sidel korol\n");
    }

    Y_UNIT_TEST(TestHasFunc) {
        UNIT_ASSERT_VALUES_EQUAL(NResource::Has("/x"), true);
        UNIT_ASSERT_VALUES_EQUAL(NResource::Has("/y"), false);
    }
}
