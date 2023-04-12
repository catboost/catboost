#include "stack_storage.h"

#include "stack.h"
#include "stack_utils.h"

#include <library/cpp/coroutine/engine/impl.h>


namespace NCoro::NStack {

    TStorage::TStorage(size_t stackSize, size_t rssPagesToKeep, size_t releaseRate)
        : StackSize_(stackSize)
        , RssPagesToKeep_(rssPagesToKeep)
        , ReleaseRate_(releaseRate ? releaseRate : 1)
    {
        Y_ASSERT(StackSize_ && RssPagesToKeep_);
    }

    bool TStorage::IsEmpty() const noexcept {
        return Released_.empty() && Full_.empty();
    }

    size_t TStorage::Size() const noexcept {
        return Released_.size() + Full_.size();
    }

    void TStorage::ReturnStack(NDetails::TStack& stack) {
        thread_local size_t i = 0;
        if (++i % ReleaseRate_ != 0) {
            Full_.push_back(stack.GetAlignedMemory());
        } else {
            ReleaseMemory(stack.GetAlignedMemory(), RssPagesToKeep_);
            Released_.push_back(stack.GetAlignedMemory());
        }
        stack.Reset();
    }

    void TStorage::ReleaseMemory([[maybe_unused]] char* alignedStackMemory, [[maybe_unused]] size_t pagesToKeep) noexcept {
#if !defined(_san_enabled_) && defined(NDEBUG)
        size_t numOfPagesToFree = StackSize_ / PageSize;
        numOfPagesToFree -= pagesToKeep;
        ReleaseRss(alignedStackMemory, numOfPagesToFree);
#endif
    }

}
