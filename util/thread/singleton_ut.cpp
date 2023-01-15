#include <library/cpp/testing/unittest/registar.h>

#include "singleton.h"

namespace {
    struct TFoo {
        int i;
        TFoo()
            : i(0)
        {
        }
    };
}

Y_UNIT_TEST_SUITE(Tls) {
    Y_UNIT_TEST(FastThread) {
        UNIT_ASSERT_VALUES_EQUAL(0, FastTlsSingleton<TFoo>()->i);
        FastTlsSingleton<TFoo>()->i += 3;
        UNIT_ASSERT_VALUES_EQUAL(3, FastTlsSingleton<TFoo>()->i);
    }
}
