#pragma once

#include "stack.h"
#include "stack_common.h"

#include <util/generic/noncopyable.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>


namespace NCoro::NStack {

    class IGuard;
    class TStorage;
    struct TPoolAllocatorSettings;

    template<typename TGuard>
    class TPool final : private TMoveOnly {
        struct TMemory {
            char* Raw = nullptr;
            char* Aligned = nullptr; // points to aligned memory, which includes space for first page guard
        };
    public:
        TPool(size_t stackSize, const TPoolAllocatorSettings& settings, const TGuard& guard);
        TPool(TPool&& other) noexcept;
        ~TPool();

        NDetails::TStack AllocStack(const char* name);
        void FreeStack(NDetails::TStack& stack);

        size_t GetReleasedSize() const noexcept;
        size_t GetFullSize() const noexcept;
        size_t GetNumOfAllocated() const noexcept { return NumOfAllocated_; }

    private:
        void AllocNewMemoryChunk();
        bool IsSmallStack() const noexcept;
        bool IsStackFromThisPool(const NDetails::TStack& stack) const noexcept;
        NDetails::TStack AllocNewStack(const char* name);

    private:
        const size_t StackSize_ = 0;
        size_t RssPagesToKeep_ = 0;
        const TGuard& Guard_;
        TVector<TMemory> Memory_; // memory chunks
        THolder<TStorage> Storage_;
        char* NextToAlloc_ = nullptr; // points to next available stack in the last memory chunk
        const size_t ChunkSize_ = 0;
        size_t NumOfAllocated_ = 0;
    };

}

#include "stack_pool.inl"
