#include <library/cpp/coroutine/engine/stack/stack_allocator.h>
#include <library/cpp/coroutine/engine/stack/stack_common.h>
#include <library/cpp/testing/gtest/gtest.h>


using namespace testing;

namespace NCoro::NStack::Tests {

    enum class EAllocator {
        Pool,    // allocates page-size aligned stacks from pools
        Simple  // uses malloc/free for each stack
    };

    class TAllocatorParamFixture : public TestWithParam< std::tuple<EGuard, EAllocator> > {
    protected: // methods
    void SetUp() override {
        EGuard guardType;
        EAllocator allocType;
        std::tie(guardType, allocType) = GetParam();

        TMaybe<TPoolAllocatorSettings> poolSettings;
        if (allocType == EAllocator::Pool) {
            poolSettings = TPoolAllocatorSettings{};
        }

        Allocator_ = GetAllocator(poolSettings, guardType);
    }

    protected: // data
        THolder<IAllocator> Allocator_;
    };


    TEST_P(TAllocatorParamFixture, StackAllocationAndRelease) {
        size_t stackSize = PageSize * 12;
        auto stack = Allocator_->AllocStack(stackSize, "test_stack");
#if defined(_san_enabled_) || !defined(NDEBUG)
        stackSize *= DebugOrSanStackMultiplier;
#endif

        // Correct stack should have
        EXPECT_EQ(stack.GetSize(), stackSize); // predefined size
        EXPECT_FALSE((size_t)stack.GetAlignedMemory() & PageSizeMask); // aligned pointer
        // Writable workspace
        auto workspace = Allocator_->GetStackWorkspace(stack.GetAlignedMemory(), stack.GetSize());
        for (size_t i = 0; i < workspace.size(); i += 512) {
            workspace[i] = 42;
        }
        EXPECT_TRUE(Allocator_->CheckStackOverflow(stack.GetAlignedMemory()));
        EXPECT_TRUE(Allocator_->CheckStackOverride(stack.GetAlignedMemory(), stack.GetSize()));

        Allocator_->FreeStack(stack);
        EXPECT_FALSE(stack.GetRawMemory());
    }

    INSTANTIATE_TEST_SUITE_P(AllocatorTestParams, TAllocatorParamFixture,
            Combine(Values(EGuard::Canary, EGuard::Page), Values(EAllocator::Pool, EAllocator::Simple)));


    // ------------------------------------------------------------------------
    // Test that allocated stack has guards
    //
    template<class AllocatorType>
    THolder<IAllocator> GetAllocator(EGuard guardType);

    struct TPoolTag {};
    struct TSimpleTag {};

    template<>
    THolder<IAllocator> GetAllocator<TPoolTag>(EGuard guardType) {
        TMaybe<TPoolAllocatorSettings> poolSettings = TPoolAllocatorSettings{};
        return GetAllocator(poolSettings, guardType);
    }

    template<>
    THolder<IAllocator> GetAllocator<TSimpleTag>(EGuard guardType) {
        TMaybe<TPoolAllocatorSettings> poolSettings;
        return GetAllocator(poolSettings, guardType);
    }


    template <class AllocatorType>
    class TAllocatorFixture : public Test {
    protected:
        TAllocatorFixture()
            : Allocator_(GetAllocator<AllocatorType>(EGuard::Page))
        {}

        const size_t StackSize_ = PageSize * 2;
        THolder<IAllocator> Allocator_;
    };

    typedef Types<TPoolTag, TSimpleTag> Implementations;
    TYPED_TEST_SUITE(TAllocatorFixture, Implementations);

    TYPED_TEST(TAllocatorFixture, StackOverflow) {
        ASSERT_DEATH({
            auto stack = this->Allocator_->AllocStack(this->StackSize_, "test_stack");

            // Overwrite previous guard, crash is here
            *(stack.GetAlignedMemory() - 1) = 42;
        }, "");
    }

    TYPED_TEST(TAllocatorFixture, StackOverride) {
        ASSERT_DEATH({
            auto stack = this->Allocator_->AllocStack(this->StackSize_, "test_stack");

            // Overwrite guard, crash is here
            *(stack.GetAlignedMemory() + stack.GetSize() - 1) = 42;
        }, "");
    }

}
