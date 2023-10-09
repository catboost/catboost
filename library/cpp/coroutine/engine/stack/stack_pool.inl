#include "stack_storage.h"
#include "stack_utils.h"


namespace NCoro::NStack {

    template<typename TGuard>
    TPool<TGuard>::TPool(size_t stackSize, const TPoolAllocatorSettings& settings, const TGuard& guard)
        : StackSize_(stackSize)
        , RssPagesToKeep_(IsSmallStack() ? settings.SmallStackRssPagesToKeep : settings.RssPagesToKeep)
        , Guard_(guard)
        , ChunkSize_(Guard_.GetPageAlignedSize() + StackSize_ * settings.StacksPerChunk)
    {
        Y_ASSERT(RssPagesToKeep_);
        if (!RssPagesToKeep_) {
            RssPagesToKeep_ = 1; // at least guard should be kept
        }

        const size_t stackSizeInPages = stackSize / PageSize;
        Y_ASSERT(stackSizeInPages >= RssPagesToKeep_);
        if (stackSizeInPages < RssPagesToKeep_) {
            RssPagesToKeep_ = stackSizeInPages; // keep all stack pages
        }

        Y_ASSERT(StackSize_ && !(StackSize_ & PageSizeMask)); // stack size is not zero and page aligned
        Y_ASSERT(Guard_.GetSize() < StackSize_); // stack has enough space to place guard
        Y_ASSERT(stackSizeInPages >= RssPagesToKeep_);

        Storage_ = MakeHolder<TStorage>(StackSize_, RssPagesToKeep_, settings.ReleaseRate);

        AllocNewMemoryChunk();
    }

    template<typename TGuard>
    TPool<TGuard>::TPool(TPool&& other) noexcept = default;

    template<typename TGuard>
    TPool<TGuard>::~TPool() {
        if (!Memory_.empty()) {
            Y_ASSERT(NextToAlloc_ && StackSize_);

            for (const auto& chunk : Memory_) {
                Y_ASSERT(chunk.Raw && chunk.Aligned);

                if (Guard_.ShouldRemoveProtectionBeforeFree()) {
                    Guard_.RemoveProtection(chunk.Aligned, Guard_.GetPageAlignedSize()); // first page in chunk

                    const char* endOfStacksMemory = chunk.Aligned + ChunkSize_;
                    for (char* i = chunk.Aligned + Guard_.GetPageAlignedSize(); i < endOfStacksMemory; i += StackSize_) {
                        Guard_.RemoveProtection(i, StackSize_);
                    }
                }

                free(chunk.Raw);
            }
        }
    }

    template<typename TGuard>
    NDetails::TStack TPool<TGuard>::AllocStack(const char* name) {
        Y_ASSERT(!Memory_.empty());

        if (!Storage_->IsEmpty()) {
            return Storage_->GetStack(Guard_, name);
        } else {
            ++NumOfAllocated_;
            return AllocNewStack(name);
        }
    }

    template<typename TGuard>
    void TPool<TGuard>::FreeStack(NDetails::TStack& stack) {
        Y_ASSERT(Storage_->Size() < ((ChunkSize_ - Guard_.GetPageAlignedSize()) / StackSize_) * Memory_.size());
        Y_ASSERT(IsStackFromThisPool(stack));

        Storage_->ReturnStack(stack);
    }

    template<typename TGuard>
    size_t TPool<TGuard>::GetReleasedSize() const noexcept {
        return Storage_->GetReleasedSize();
    }
    template<typename TGuard>
    size_t TPool<TGuard>::GetFullSize() const noexcept {
        return Storage_->GetFullSize();
    }

    template<typename TGuard>
    void TPool<TGuard>::AllocNewMemoryChunk() {
        const size_t totalSizeInPages = ChunkSize_ / PageSize;

        TMemory memory;
        const auto res = GetAlignedMemory(totalSizeInPages, memory.Raw, memory.Aligned);
        Y_ABORT_UNLESS(res, "Failed to allocate memory for coro stack pool");

        NextToAlloc_ = memory.Aligned + Guard_.GetPageAlignedSize(); // skip first guard page
        Guard_.Protect(memory.Aligned, Guard_.GetPageAlignedSize(), false); // protect first guard page

        Memory_.push_back(std::move(memory));
    }

    template<typename TGuard>
    bool TPool<TGuard>::IsSmallStack() const noexcept {
        return StackSize_ / PageSize <= SmallStackMaxSizeInPages;
    }

    template<typename TGuard>
    bool TPool<TGuard>::IsStackFromThisPool(const NDetails::TStack& stack) const noexcept {
        for (const auto& chunk : Memory_) {
            const char* endOfStacksMemory = chunk.Aligned + ChunkSize_;
            if (chunk.Raw <= stack.GetRawMemory() && stack.GetRawMemory() < endOfStacksMemory) {
                return true;
            }
        }
        return false;
    }

    template<typename TGuard>
    NDetails::TStack TPool<TGuard>::AllocNewStack(const char* name) {
        if (NextToAlloc_ + StackSize_ > Memory_.rbegin()->Aligned + ChunkSize_) {
            AllocNewMemoryChunk(); // also sets NextToAlloc_ to first stack position in new allocated chunk of memory
        }
        Y_ASSERT(NextToAlloc_ + StackSize_ <= Memory_.rbegin()->Aligned + ChunkSize_);

        char* newStack = NextToAlloc_;
        NextToAlloc_ += StackSize_;

        Guard_.Protect(newStack, StackSize_, true);
        return NDetails::TStack{newStack, newStack, StackSize_, name};
    }

}
