#include <library/cpp/coroutine/engine/stack/stack_common.h>
#include <library/cpp/coroutine/engine/stack/stack_guards.h>
#include <library/cpp/coroutine/engine/stack/stack_utils.h>
#include <library/cpp/testing/gtest/gtest.h>


using namespace testing;

namespace NCoro::NStack::Tests {

    template <class TGuard>
    class TGuardFixture : public Test {
    protected:
        TGuardFixture() : Guard_(GetGuard<TGuard>()) {}

        const TGuard& Guard_;
    };

    typedef Types<TCanaryGuard, TPageGuard> Implementations;
    TYPED_TEST_SUITE(TGuardFixture, Implementations);

    TYPED_TEST(TGuardFixture, GuardSize) {
        const auto size = this->Guard_.GetSize();
        EXPECT_GE(size, 64ul);
        EXPECT_FALSE(size & 63ul); // check 64-byte alignment
    }

    TYPED_TEST(TGuardFixture, GuardAlignedSize) {
        const auto size = this->Guard_.GetPageAlignedSize();
        EXPECT_GE(size, PageSize);
        EXPECT_FALSE(size & PageSizeMask); // check page-alignment
    }

    TYPED_TEST(TGuardFixture, StackWorkspace) {
        for (size_t sizeInPages : {2, 5, 12}) {
            char *rawPtr, *alignedPtr = nullptr;
            ASSERT_TRUE(GetAlignedMemory(sizeInPages, rawPtr, alignedPtr));
            auto workspace = this->Guard_.GetWorkspace(alignedPtr, sizeInPages * PageSize);
            EXPECT_EQ(workspace.size(), sizeInPages * PageSize - this->Guard_.GetSize()) << " size in pages " << sizeInPages;

            this->Guard_.Protect(alignedPtr, sizeInPages * PageSize, false);
            workspace = this->Guard_.GetWorkspace(alignedPtr, sizeInPages * PageSize);
            EXPECT_EQ(workspace.size(), sizeInPages * PageSize - this->Guard_.GetSize()) << " size in pages " << sizeInPages;

            this->Guard_.RemoveProtection(alignedPtr, sizeInPages * PageSize);
            workspace = this->Guard_.GetWorkspace(alignedPtr, sizeInPages * PageSize);
            EXPECT_EQ(workspace.size(), sizeInPages * PageSize - this->Guard_.GetSize()) << " size in pages " << sizeInPages;

            free(rawPtr);
        }
    }

    TYPED_TEST(TGuardFixture, SetRemoveProtectionWorks) {
        char *rawPtr, *alignedPtr = nullptr;
        constexpr size_t sizeInPages = 4;
        ASSERT_TRUE(GetAlignedMemory(sizeInPages + 1, rawPtr, alignedPtr));

        this->Guard_.Protect(alignedPtr, PageSize, false); // set previous guard
        alignedPtr += PageSize; // leave first page for previous guard
        this->Guard_.Protect(alignedPtr, sizeInPages * PageSize, true);

        EXPECT_TRUE(this->Guard_.CheckOverflow(alignedPtr));
        EXPECT_TRUE(this->Guard_.CheckOverride(alignedPtr, sizeInPages * PageSize));

        this->Guard_.RemoveProtection(alignedPtr, sizeInPages * PageSize);
        this->Guard_.RemoveProtection(alignedPtr - PageSize, PageSize); // remove previous guard

        free(rawPtr);
    }

    TEST(StackGuardTest, CanaryGuardTestOverflow) {
        const auto& guard = GetGuard<TCanaryGuard>();

        char *rawPtr, *alignedPtr = nullptr;
        constexpr size_t sizeInPages = 4;
        ASSERT_TRUE(GetAlignedMemory(sizeInPages + 1, rawPtr, alignedPtr));
        guard.Protect(alignedPtr, PageSize, false); // set previous guard
        alignedPtr += PageSize; // leave first page for previous guard
        guard.Protect(alignedPtr, sizeInPages * PageSize, true);

        EXPECT_TRUE(guard.CheckOverflow(alignedPtr));
        EXPECT_TRUE(guard.CheckOverride(alignedPtr, sizeInPages * PageSize));

        // Overwrite previous guard
        *(alignedPtr - 1) = 42;

        EXPECT_FALSE(guard.CheckOverflow(alignedPtr));

        free(rawPtr);
    }

    TEST(StackGuardTest, CanaryGuardTestOverride) {
        const auto& guard = GetGuard<TCanaryGuard>();

        char *rawPtr, *alignedPtr = nullptr;
        constexpr size_t sizeInPages = 4;
        ASSERT_TRUE(GetAlignedMemory(sizeInPages + 1, rawPtr, alignedPtr));
        guard.Protect(alignedPtr, PageSize, false); // set previous guard
        alignedPtr += PageSize; // leave first page for previous guard
        guard.Protect(alignedPtr, sizeInPages * PageSize, true);

        EXPECT_TRUE(guard.CheckOverflow(alignedPtr));
        EXPECT_TRUE(guard.CheckOverride(alignedPtr, sizeInPages * PageSize));

        // Overwrite guard
        *(alignedPtr + sizeInPages * PageSize - 1) = 42;

        EXPECT_FALSE(guard.CheckOverride(alignedPtr, sizeInPages * PageSize));

        free(rawPtr);
    }

    TEST(StackGuardDeathTest, PageGuardTestOverflow) {
        ASSERT_DEATH({
            const auto &guard = GetGuard<TPageGuard>();

            char* rawPtr = nullptr;
            char* alignedPtr = nullptr;
            constexpr size_t sizeInPages = 4;
            ASSERT_TRUE(GetAlignedMemory(sizeInPages + 1, rawPtr, alignedPtr));

            guard.Protect(alignedPtr, PageSize, false); // set previous guard
            alignedPtr += PageSize; // leave first page for previous guard
            guard.Protect(alignedPtr, sizeInPages * PageSize, true);

            // Overwrite previous guard, crash is here
            *(alignedPtr - 1) = 42;

            guard.RemoveProtection(alignedPtr, sizeInPages * PageSize);
            guard.RemoveProtection(alignedPtr - PageSize, PageSize); // remove previous guard

            free(rawPtr);
        }, "");
    }

    TEST(StackGuardDeathTest, PageGuardTestOverride) {
        ASSERT_DEATH({
            const auto &guard = GetGuard<TPageGuard>();

            char* rawPtr = nullptr;
            char* alignedPtr = nullptr;
            constexpr size_t sizeInPages = 4;
            ASSERT_TRUE(GetAlignedMemory(sizeInPages + 1, rawPtr, alignedPtr));
            guard.Protect(alignedPtr, PageSize, false); // set previous guard
            alignedPtr += PageSize; // leave first page for previous guard
            guard.Protect(alignedPtr, sizeInPages * PageSize, true);

            // Overwrite guard, crash is here
            *(alignedPtr + sizeInPages * PageSize - 1) = 42;

            guard.RemoveProtection(alignedPtr, sizeInPages * PageSize);
            guard.RemoveProtection(alignedPtr - PageSize, PageSize); // remove previous guard

            free(rawPtr);
        }, "");
    }

}
