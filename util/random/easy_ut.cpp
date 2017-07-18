#include "easy.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TEasyRndInterface) {
    SIMPLE_UNIT_TEST(Test1) {
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
}
