#include "smart_ptr.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/str.h>

using namespace NNeh;

Y_UNIT_TEST_SUITE(TSmartPtr) {
    Y_UNIT_TEST(TUsageTest) {
        //validate smart ptr
        typedef TSharedPtrB<int> TSharedInt;
        typedef TWeakPtrB<int> TWeakInt;

        TWeakInt w01;
        TWeakInt w02;

        UNIT_ASSERT(w01.UseCount() == 0);

        {
            TSharedInt s1(new int(42));

            UNIT_ASSERT(s1.UseCount() == 1);
            {
                TSharedInt s2(s1);

                UNIT_ASSERT(s1.UseCount() == 2);
                UNIT_ASSERT(s2.UseCount() == 2);

                TWeakInt w1(s1);
                TWeakInt w2 = s2;

                UNIT_ASSERT(s1.UseCount() == 2);
                UNIT_ASSERT(w1.UseCount() == 2);

                TSharedInt s3(w1);

                UNIT_ASSERT(s3.UseCount() == 3);
                UNIT_ASSERT(s3.Get() != (int*)nullptr && *s3 == 42);

                {
                    TSharedInt s4 = w1;
                    UNIT_ASSERT(s4.UseCount() == 4);

                    w01 = s4;
                    w02 = w1;
                }

                UNIT_ASSERT(w01.UseCount() == 3);
                UNIT_ASSERT(*s3 == 42);

                s3 = TSharedInt();

                UNIT_ASSERT(s3.UseCount() == 0);
                UNIT_ASSERT(w1.UseCount() == 2);

                s3.Swap(s2);

                UNIT_ASSERT(s3.UseCount() == 2);
                UNIT_ASSERT(s2.UseCount() == 0);

                s3.Reset();

                UNIT_ASSERT(s3.UseCount() == 0);
                UNIT_ASSERT(s1.UseCount() == 1);
                UNIT_ASSERT(w01.UseCount() == 1);
            }

            UNIT_ASSERT(w01.UseCount() == 1);
            UNIT_ASSERT(s1.UseCount() == 1);
        }

        UNIT_ASSERT(w01.UseCount() == 0);
        UNIT_ASSERT(w02.UseCount() == 0);

        TSharedInt s0(w01);

        UNIT_ASSERT(s0.UseCount() == 0);
        UNIT_ASSERT(!s0 && s0.Get() == (int*)nullptr);
    }
}
