#include "stack_vec.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TStackBasedVectorTest) {
    SIMPLE_UNIT_TEST(TestCreateEmpty) {
        TStackVec<int> ints;
        UNIT_ASSERT_EQUAL(ints.size(), 0);
    }

    SIMPLE_UNIT_TEST(TestCreateNonEmpty) {
        TStackVec<int> ints(5);
        UNIT_ASSERT_EQUAL(ints.size(), 5);

        for (size_t i = 0; i < ints.size(); ++i) {
            UNIT_ASSERT_EQUAL(ints[i], 0);
        }
    }

    SIMPLE_UNIT_TEST(TestReallyOnStack) {
        TStackVec<int> ints(5);
#ifdef _LIBCPP_MEMORY // USE_STL_LIBCXX_TRUNK
        // Depends on libc++ std::vector layout, which is now __begin__, then __end__,
        // then __end_cap_ which is a __compressed_pair<pointer, allocator_type>
        UNIT_ASSERT_EQUAL((const char*)ints.data(), ((const char*)&ints) + 3 * sizeof(TStackVec<int>::pointer));
#else
        // Depends on STLPort's std::vector layout, which is now _M_start, _M_finish, then AllocProxy
        UNIT_ASSERT_EQUAL((const char*)ints.data(), ((const char*)&ints) + 2 * sizeof(TStackVec<int>::pointer));
#endif
    }

    SIMPLE_UNIT_TEST(TestFallback) {
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
}
