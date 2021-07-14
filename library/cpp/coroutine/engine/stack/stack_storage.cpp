#include "stack_storage.h"

#include "stack.h"
#include "stack_guards.h"
#include "stack_utils.h"

#include <library/cpp/coroutine/engine/impl.h>


namespace NCoro::NStack {

    constexpr uint64_t ReleasedToFull = 10;
    constexpr uint64_t ReleaseForEach = 16;

    TStorage::TStorage(TContExecutor* executor, uint64_t stackSize, uint64_t rssPagesToKeep)
        : Executor_(executor)
        , StackSize_(stackSize)
        , RssPagesToKeep_(rssPagesToKeep)
    {
        Y_ASSERT(StackSize_ && RssPagesToKeep_);
    }

    TStorage::~TStorage() {
        if (ReleaseCoro_) {
            Y_ASSERT(Executor_);
            ReleaseCoro_->Cancel();
            Executor_->Running()->Join(ReleaseCoro_, TInstant::Now());
        }
    }

    bool TStorage::IsEmpty() const noexcept {
        return Released_.empty() && Full_.empty();
    }

    uint64_t TStorage::Size() const noexcept {
        return Released_.size() + Full_.size();
    }

    void TStorage::ReturnStack(NDetails::TStack& stack) {
        thread_local uint64_t i = 0;
        if (++i % ReleaseForEach != 0) {
            Full_.push_back(stack.GetAlignedMemory());
        } else {
            ReleaseMemory(stack.GetAlignedMemory(), RssPagesToKeep_);
            Released_.push_back(stack.GetAlignedMemory());
        }
        stack.Reset();

        if (!Executor_) {
            return;
        }

        CheckReleaseCoro();
    }

    void TStorage::ReleaseMemory([[maybe_unused]] char* alignedStackMemory, [[maybe_unused]] uint64_t pagesToKeep) noexcept {
#if !defined(_san_enabled_) && defined(NDEBUG)
        uint64_t numOfPagesToFree = StackSize_ / PageSize;
        numOfPagesToFree -= pagesToKeep;
        ReleaseRss(alignedStackMemory, numOfPagesToFree);
#endif
    }

    void TStorage::CheckReleaseCoro() {
        Y_ASSERT(Executor_);
        if (!ReleaseCoro_ && Full_.size() > Released_.size() / ReleasedToFull) {
            ReleaseCoro_ = Executor_->Create<TStorage, &TStorage::RunReleaseCoro>(this, "coro_stack_release");
        }
    }

    void TStorage::RunReleaseCoro(TCont*) noexcept {
        while (ReleaseCoro_ && !ReleaseCoro_->Cancelled() && Full_.size() > Released_.size() / (2 * ReleasedToFull)) {
            void* stack = Full_.front(); // take oldest Full_
            Full_.pop_front();
            ReleaseMemory((char*)stack, 1); // leave only one page with guard
            Released_.push_front(stack);
            if (ReleaseCoro_ && !ReleaseCoro_->Cancelled()) {
                ReleaseCoro_->Yield();
            }
        }

        ReleaseCoro_ = nullptr;
    }

}
