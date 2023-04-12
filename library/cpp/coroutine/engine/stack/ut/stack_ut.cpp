#include <library/cpp/coroutine/engine/stack/stack.h>
#include <library/cpp/coroutine/engine/stack/stack_common.h>
#include <library/cpp/coroutine/engine/stack/stack_guards.h>
#include <library/cpp/coroutine/engine/stack/stack_utils.h>
#include <library/cpp/testing/gtest/gtest.h>


using namespace testing;

namespace NCoro::NStack::Tests {

    constexpr size_t StackSizeInPages = 4;

    template <class TGuard>
    class TStackFixture : public Test {
    protected: // methods
        TStackFixture()
            : Guard_(GetGuard<TGuard>())
            , StackSize_(StackSizeInPages * PageSize)
        {}

        void SetUp() override {
            ASSERT_TRUE(GetAlignedMemory(StackSizeInPages, RawMemory_, AlignedMemory_));
            Stack_ = MakeHolder<NDetails::TStack>(RawMemory_, AlignedMemory_, StackSize_, "test_stack");
            Guard_.Protect(AlignedMemory_, StackSize_, false);
        }

        void TearDown() override {
            Guard_.RemoveProtection(AlignedMemory_, StackSize_);
            free(Stack_->GetRawMemory());
            Stack_->Reset();
            EXPECT_EQ(Stack_->GetRawMemory(), nullptr);
        }

    protected: // data
        const TGuard& Guard_;
        const size_t StackSize_ = 0;
        char* RawMemory_ = nullptr;
        char* AlignedMemory_ = nullptr;
        THolder<NDetails::TStack> Stack_;
    };

    typedef Types<TCanaryGuard, TPageGuard> Implementations;
    TYPED_TEST_SUITE(TStackFixture, Implementations);

    TYPED_TEST(TStackFixture, PointersAndSize) {
        EXPECT_EQ(this->Stack_->GetRawMemory(), this->RawMemory_);
        EXPECT_EQ(this->Stack_->GetAlignedMemory(), this->AlignedMemory_);
        EXPECT_EQ(this->Stack_->GetSize(), this->StackSize_);
    }

    TYPED_TEST(TStackFixture, WriteStack) {
        auto workspace = this->Guard_.GetWorkspace(this->Stack_->GetAlignedMemory(), this->Stack_->GetSize());
        for (size_t i = 0; i < workspace.size(); i += 512) {
            workspace[i] = 42;
        }
        EXPECT_TRUE(this->Guard_.CheckOverride(this->Stack_->GetAlignedMemory(), this->Stack_->GetSize()));
    }

}
