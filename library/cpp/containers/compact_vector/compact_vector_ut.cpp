#include <library/cpp/unittest/registar.h>

#include "compact_vector.h"

Y_UNIT_TEST_SUITE(TCompactVectorTest) {
    Y_UNIT_TEST(TestSimple1) {
    }

    Y_UNIT_TEST(TestSimple) {
        TCompactVector<ui32> vector;
        for (ui32 i = 0; i < 10000; ++i) {
            vector.PushBack(i + 20);
            UNIT_ASSERT_VALUES_EQUAL(i + 1, vector.Size());
        }
        for (ui32 i = 0; i < 10000; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(i + 20, vector[i]);
        }
    }

    Y_UNIT_TEST(TestInsert) {
        TCompactVector<ui32> vector;

        for (ui32 i = 0; i < 10; ++i) {
            vector.PushBack(i + 2);
        }

        vector.Insert(vector.Begin(), 99);

        UNIT_ASSERT_VALUES_EQUAL(11u, vector.Size());
        UNIT_ASSERT_VALUES_EQUAL(99u, vector[0]);
        for (ui32 i = 0; i < 10; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(i + 2, vector[i + 1]);
        }

        vector.Insert(vector.Begin() + 3, 77);

        UNIT_ASSERT_VALUES_EQUAL(12u, vector.Size());
        UNIT_ASSERT_VALUES_EQUAL(99u, vector[0]);
        UNIT_ASSERT_VALUES_EQUAL(2u, vector[1]);
        UNIT_ASSERT_VALUES_EQUAL(3u, vector[2]);
        UNIT_ASSERT_VALUES_EQUAL(77u, vector[3]);
        UNIT_ASSERT_VALUES_EQUAL(4u, vector[4]);
        UNIT_ASSERT_VALUES_EQUAL(5u, vector[5]);
        UNIT_ASSERT_VALUES_EQUAL(11u, vector[11]);
    }
}
