#include <library/cpp/testing/unittest/registar.h>

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

    Y_UNIT_TEST(TestInitializerListConstructor) {
        TCompactVector<ui32> vector = { 4, 8, 10, 3, 5};
        UNIT_ASSERT_VALUES_EQUAL(5u, vector.Size());

        UNIT_ASSERT_VALUES_EQUAL(4u, vector[0]);
        UNIT_ASSERT_VALUES_EQUAL(8u, vector[1]);
        UNIT_ASSERT_VALUES_EQUAL(10u, vector[2]);
        UNIT_ASSERT_VALUES_EQUAL(3u, vector[3]);
        UNIT_ASSERT_VALUES_EQUAL(5u, vector[4]);
    }

    Y_UNIT_TEST(TestIteratorConstructor) {
        TVector<ui32> origVector = { 4, 8, 10, 3, 5};
        TCompactVector<ui32> vector(origVector.begin(), origVector.end());
        UNIT_ASSERT_VALUES_EQUAL(5u, vector.Size());

        UNIT_ASSERT_VALUES_EQUAL(4u, vector[0]);
        UNIT_ASSERT_VALUES_EQUAL(8u, vector[1]);
        UNIT_ASSERT_VALUES_EQUAL(10u, vector[2]);
        UNIT_ASSERT_VALUES_EQUAL(3u, vector[3]);
        UNIT_ASSERT_VALUES_EQUAL(5u, vector[4]);
    }

    Y_UNIT_TEST(TestInitializerListCopyOperator) {
        TCompactVector<double> vector = { 4, 8, 10, 3, 5};
        UNIT_ASSERT_VALUES_EQUAL(5u, vector.Size());

        vector = { 11, 17, 23 };
        UNIT_ASSERT_VALUES_EQUAL(3u, vector.Size());

        UNIT_ASSERT_VALUES_EQUAL(11.0, vector[0]);
        UNIT_ASSERT_VALUES_EQUAL(17.0, vector[1]);
        UNIT_ASSERT_VALUES_EQUAL(23.0, vector[2]);
    }

    Y_UNIT_TEST(TestMoveConstructor) {
        TCompactVector<ui32> vector = { 4, 8, 10, 3, 5};
        auto it = vector.Begin();

        TCompactVector<ui32> vector2(std::move(vector));
        UNIT_ASSERT_VALUES_EQUAL(it, vector2.begin());

        UNIT_ASSERT_VALUES_EQUAL(5u, vector2.Size());

        UNIT_ASSERT_VALUES_EQUAL(4u, vector2[0]);
        UNIT_ASSERT_VALUES_EQUAL(8u, vector2[1]);
        UNIT_ASSERT_VALUES_EQUAL(10u, vector2[2]);
        UNIT_ASSERT_VALUES_EQUAL(3u, vector2[3]);
        UNIT_ASSERT_VALUES_EQUAL(5u, vector2[4]);
    }

    Y_UNIT_TEST(TestReverseIterators) {
        TCompactVector<std::string> vector = {
                "мама",
                "мыла",
                "раму"
        };

        TCompactVector<std::string> reverseVector(vector.rbegin(), vector.rend());
        UNIT_ASSERT_VALUES_EQUAL(3u, reverseVector.Size());

        UNIT_ASSERT_VALUES_EQUAL("раму", reverseVector[0]);
        UNIT_ASSERT_VALUES_EQUAL("мыла", reverseVector[1]);
        UNIT_ASSERT_VALUES_EQUAL("мама", reverseVector[2]);
    }

    Y_UNIT_TEST(TestErase) {
        TCompactVector<std::string> vector = {
                "мама",
                "утром",
                "мыла",
                "раму"
        };

        vector.erase(vector.begin() + 1);
        UNIT_ASSERT_VALUES_EQUAL(3u, vector.Size());

        UNIT_ASSERT_VALUES_EQUAL("мама", vector[0]);
        UNIT_ASSERT_VALUES_EQUAL("мыла", vector[1]);
        UNIT_ASSERT_VALUES_EQUAL("раму", vector[2]);
    }

    Y_UNIT_TEST(TestCopyAssignmentOperator) {
        TCompactVector<std::string> vector;
        const TCompactVector<std::string> vector2 = {
                "мама",
                "мыла",
                "раму"
        };

        vector = vector2;

        UNIT_ASSERT_VALUES_EQUAL(3u, vector.Size());

        UNIT_ASSERT_VALUES_EQUAL("мама", vector[0]);
        UNIT_ASSERT_VALUES_EQUAL("мыла", vector[1]);
        UNIT_ASSERT_VALUES_EQUAL("раму", vector[2]);

        UNIT_ASSERT_VALUES_EQUAL(vector[0], vector2[0]);
        UNIT_ASSERT_VALUES_EQUAL(vector[1], vector2[1]);
        UNIT_ASSERT_VALUES_EQUAL(vector[2], vector2[2]);
    }
}
