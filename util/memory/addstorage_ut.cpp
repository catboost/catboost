#include "addstorage.h"

#include <library/cpp/testing/unittest/registar.h>

class TAddStorageTest: public TTestBase {
    UNIT_TEST_SUITE(TAddStorageTest);
    UNIT_TEST(TestIt)
    UNIT_TEST_SUITE_END();

    class TClass: public TAdditionalStorage<TClass> {
    };

private:
    inline void TestIt() {
        THolder<TClass> c(new (100) TClass);

        UNIT_ASSERT_EQUAL(c->AdditionalDataLength(), 100);

        // test segfault
        memset(c->AdditionalData(), 0, c->AdditionalDataLength());
    }
};

UNIT_TEST_SUITE_REGISTRATION(TAddStorageTest);
