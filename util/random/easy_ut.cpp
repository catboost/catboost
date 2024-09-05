#include "easy.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TEasyRndInterface) {
    Y_UNIT_TEST(Test1) {
        {
            ui32 x = 0;

            x = Random();

            if (!x) {
                x = Random();
            }

            UNIT_ASSERT(x != 0);
        }

        {
            ui64 x = 0;

            x = Random();

            UNIT_ASSERT(x != 0);
        }

        {
            long double x = 0;

            x = Random();

            UNIT_ASSERT(x != 0);
        }
    }
} // Y_UNIT_TEST_SUITE(TEasyRndInterface)
