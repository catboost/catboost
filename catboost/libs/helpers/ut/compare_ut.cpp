#include <catboost/libs/helpers/compare.h>

#include <util/generic/ptr.h>

#include <library/unittest/registar.h>


Y_UNIT_TEST_SUITE(ArePointeesEqual) {
    Y_UNIT_TEST(Test) {
        {
            THolder<int> h1;
            THolder<int> h2;

            UNIT_ASSERT(ArePointeesEqual(h1, h2));
        }
        {
            THolder<int> h1;
            THolder<int> h2 = MakeHolder<int>(1);

            UNIT_ASSERT(!ArePointeesEqual(h1, h2));
        }
        {
            THolder<int> h1 = MakeHolder<int>(1);
            THolder<int> h2 = MakeHolder<int>(1);

            UNIT_ASSERT(ArePointeesEqual(h1, h2));
        }
        {
            THolder<int> h1 = MakeHolder<int>(1);
            THolder<int> h2 = MakeHolder<int>(2);

            UNIT_ASSERT(!ArePointeesEqual(h1, h2));
        }
    }
}
