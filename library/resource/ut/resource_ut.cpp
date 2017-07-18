#include <library/resource/resource.h>
#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TestResource) {
    SIMPLE_UNIT_TEST(Test1) {
        UNIT_ASSERT_VALUES_EQUAL(NResource::Find("/x"), "na gorshke sidel korol\n");
    }
}
