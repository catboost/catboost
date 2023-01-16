#include "stack_vec.h"

#include <library/cpp/testing/unittest/registar.h>

namespace {
    struct TNotCopyAssignable {
        const int Value;
    };

    static_assert(std::is_copy_constructible_v<TNotCopyAssignable>);
    static_assert(!std::is_copy_assignable_v<TNotCopyAssignable>);

    template <class T, size_t JunkPayloadSize>
    struct TThickAlloc: public std::allocator<T> {
        template <class U>
        struct rebind {
            using other = TThickAlloc<U, JunkPayloadSize>;
        };

        char Junk[JunkPayloadSize]{sizeof(T)};
    };

    template <class T>
    struct TStatefulAlloc: public std::allocator<T> {
        using TBase = std::allocator<T>;

        template <class U>
        struct rebind {
            using other = TStatefulAlloc<U>;
        };

        TStatefulAlloc(size_t* allocCount)
            : AllocCount(allocCount)
        {}

        size_t* AllocCount;

        T* allocate(size_t n)
        {
            *AllocCount += 1;
            return TBase::allocate(n);
        }
    };
}

Y_UNIT_TEST_SUITE(TStackBasedVectorTest) {
    Y_UNIT_TEST(TestCreateEmpty) {
        TStackVec<int> ints;
        UNIT_ASSERT_EQUAL(ints.size(), 0);
    }

    Y_UNIT_TEST(TestCreateNonEmpty) {
        TStackVec<int> ints(5);
        UNIT_ASSERT_EQUAL(ints.size(), 5);

        for (size_t i = 0; i < ints.size(); ++i) {
            UNIT_ASSERT_EQUAL(ints[i], 0);
        }
    }

    Y_UNIT_TEST(TestReallyOnStack) {
        const TStackVec<int> vec(5);

        UNIT_ASSERT(
            (const char*)&vec <= (const char*)&vec[0] &&
            (const char*)&vec[0] <= (const char*)&vec + sizeof(vec)
        );
    }

    Y_UNIT_TEST(TestFallback) {
        TSmallVec<int> ints;
        for (int i = 0; i < 14; ++i) {
            ints.push_back(i);
        }

        for (size_t i = 0; i < ints.size(); ++i) {
            UNIT_ASSERT_EQUAL(ints[i], (int)i);
        }

        for (int i = 14; i < 20; ++i) {
            ints.push_back(i);
        }

        for (size_t i = 0; i < ints.size(); ++i) {
            UNIT_ASSERT_EQUAL(ints[i], (int)i);
        }

        TSmallVec<int> ints2 = ints;

        for (size_t i = 0; i < ints2.size(); ++i) {
            UNIT_ASSERT_EQUAL(ints2[i], (int)i);
        }

        TSmallVec<int> ints3;
        ints3 = ints2;

        for (size_t i = 0; i < ints3.size(); ++i) {
            UNIT_ASSERT_EQUAL(ints3[i], (int)i);
        }
    }

    Y_UNIT_TEST(TestCappedSize) {
        TStackVec<int, 8, false> ints;
        ints.push_back(1);
        ints.push_back(2);

        auto intsCopy = ints;
        UNIT_ASSERT_VALUES_EQUAL(intsCopy.capacity(), 8);

        for (int i = 2; i != 8; ++i) {
            intsCopy.push_back(i);
        }
        // Just verify that the program did not crash.
    }

    Y_UNIT_TEST(TestCappedSizeWithNotCopyAssignable) {
        TStackVec<TNotCopyAssignable, 8, false> values;
        values.push_back({1});
        values.push_back({2});

        auto valuesCopy = values;
        UNIT_ASSERT_VALUES_EQUAL(valuesCopy.capacity(), 8);

        for (int i = 2; i != 8; ++i) {
            valuesCopy.push_back({i});
        }
        // Just verify that the program did not crash.
    }

    Y_UNIT_TEST(TestCustomAllocSize) {
        constexpr size_t n = 16384;
        using TVec = TStackVec<size_t, 1, true, TThickAlloc<size_t, n>>;
        UNIT_ASSERT_LT(sizeof(TVec), 1.5 * n);
    }

    Y_UNIT_TEST(TestStatefulAlloc) {
        size_t count = 0;
        TStackVec<size_t, 1, true, TStatefulAlloc<size_t>> vec{{ &count }};
        for (size_t i = 0; i < 5; ++i) {
            vec.push_back(1);
        }
        UNIT_ASSERT_VALUES_EQUAL(count, 3);
    }
}
