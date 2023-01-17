#include <library/cpp/coroutine/engine/stack/stack_common.h>
#include <library/cpp/coroutine/engine/stack/stack_guards.h>
#include <library/cpp/coroutine/engine/stack/stack_pool.h>
#include <library/cpp/testing/gtest/gtest.h>


using namespace testing;

namespace NCoro::NStack::Tests {

    template <class TGuard>
    class TPoolFixture : public Test {
    protected:
        TPoolFixture() : Guard_(GetGuard<TGuard>()), Pool_(StackSize_, TPoolAllocatorSettings{1, 1, 8, 32}, Guard_) {}

        const size_t StackSize_ = PageSize * 4;
        const TGuard& Guard_;
        TPool<TGuard> Pool_;
    };

    typedef Types<TCanaryGuard, TPageGuard> Implementations;
    TYPED_TEST_SUITE(TPoolFixture, Implementations);

    TYPED_TEST(TPoolFixture, AllocAndFreeStack) {
        auto stack = this->Pool_.AllocStack("test_stack");
        this->Pool_.FreeStack(stack);
        EXPECT_FALSE(stack.GetRawMemory());
    }

    TYPED_TEST(TPoolFixture, FreedStackReused) {
        auto stack = this->Pool_.AllocStack("test_stack");
        auto rawPtr = stack.GetRawMemory();
        auto alignedPtr = stack.GetAlignedMemory();

        this->Pool_.FreeStack(stack);
        EXPECT_FALSE(stack.GetRawMemory());

        auto stack2 = this->Pool_.AllocStack("test_stack");
        EXPECT_EQ(rawPtr, stack2.GetRawMemory());
        EXPECT_EQ(alignedPtr, stack2.GetAlignedMemory());

        this->Pool_.FreeStack(stack2);
        EXPECT_FALSE(stack2.GetRawMemory());
    }

    TYPED_TEST(TPoolFixture, MruFreedStackReused) {
        auto stack = this->Pool_.AllocStack("test_stack");
        auto rawPtr = stack.GetRawMemory();
        auto alignedPtr = stack.GetAlignedMemory();
        auto stack2 = this->Pool_.AllocStack("test_stack");
        auto stack3 = this->Pool_.AllocStack("test_stack");

        this->Pool_.FreeStack(stack2);
        EXPECT_FALSE(stack2.GetRawMemory());

        this->Pool_.FreeStack(stack);
        EXPECT_FALSE(stack.GetRawMemory());

        auto stack4 = this->Pool_.AllocStack("test_stack");
        EXPECT_EQ(rawPtr, stack4.GetRawMemory());
        EXPECT_EQ(alignedPtr, stack4.GetAlignedMemory());

        this->Pool_.FreeStack(stack3);
        EXPECT_FALSE(stack.GetRawMemory());

        this->Pool_.FreeStack(stack4);
        EXPECT_FALSE(stack4.GetRawMemory());
    }

}
