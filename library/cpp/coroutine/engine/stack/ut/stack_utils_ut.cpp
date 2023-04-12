#include <library/cpp/coroutine/engine/stack/stack_common.h>
#include <library/cpp/coroutine/engine/stack/stack_utils.h>
#include <library/cpp/testing/gtest/gtest.h>


using namespace testing;

namespace NCoro::NStack::Tests {

    TEST(StackUtilsTest, Allocation) {
        char *rawPtr, *alignedPtr = nullptr;
        for (size_t i : {1, 2, 3, 4, 11}) {
            EXPECT_TRUE(GetAlignedMemory(i, rawPtr, alignedPtr));
            EXPECT_TRUE(rawPtr);
            EXPECT_TRUE(alignedPtr);
            EXPECT_FALSE((size_t)alignedPtr & PageSizeMask);
            free(rawPtr);
        }
    }

#if !defined(_san_enabled_) && defined(_linux_)

    TEST(StackUtilsTest, RssReleaseOnePage) {
        char *rawPtr, *alignedPtr = nullptr;
        for (size_t i : {1, 2, 8}) {
            EXPECT_TRUE(GetAlignedMemory(i, rawPtr, alignedPtr));
            EXPECT_TRUE(rawPtr);
            EXPECT_TRUE(alignedPtr);
            EXPECT_FALSE((size_t)alignedPtr & PageSizeMask);

            ReleaseRss(alignedPtr, i); // allocator can provide reused memory with RSS memory on it
            EXPECT_EQ(CountMapped(alignedPtr, i), 0ul); // no RSS memory allocated

            *(alignedPtr + (i - 1) * PageSize) = 42; // map RSS memory
            EXPECT_EQ(CountMapped(alignedPtr, i), 1ul);

            ReleaseRss(alignedPtr, i);
            EXPECT_EQ(CountMapped(alignedPtr, i), 0ul) << "number of pages " << i; // no RSS memory allocated

            free(rawPtr);
        }
    }

    TEST(StackUtilsTest, RssReleaseSeveralPages) {
        char *rawPtr, *alignedPtr = nullptr;

        for (size_t i : {1, 2, 5, 8}) {
            EXPECT_TRUE(GetAlignedMemory(i, rawPtr, alignedPtr));
            EXPECT_TRUE(rawPtr);
            EXPECT_TRUE(alignedPtr);
            EXPECT_FALSE((size_t)alignedPtr & PageSizeMask);

            ReleaseRss(alignedPtr, i); // allocator can provide reused memory with RSS memory on it
            EXPECT_EQ(CountMapped(alignedPtr, i), 0ul); // no RSS memory allocated

            for (size_t page = 0; page < i; ++page) {
                *(alignedPtr + page * PageSize) = 42; // map RSS memory
                EXPECT_EQ(CountMapped(alignedPtr, page + 1), page + 1);
            }

            const size_t pagesToKeep = (i > 2) ? 2 : i;

            ReleaseRss(alignedPtr, i - pagesToKeep);
            EXPECT_EQ(CountMapped(alignedPtr, i), pagesToKeep) << "number of pages " << i; // no RSS memory allocated

            free(rawPtr);
        }
    }

#endif

}

