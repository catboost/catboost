#pragma once

#include "stack.h"

#include <util/datetime/base.h>
#include <util/generic/deque.h>


class TCont;

namespace NCoro::NStack {

    class IGuard;

    class TStorage final : private TMoveOnly {
    public:
        TStorage(uint64_t stackSize, uint64_t rssPagesToKeep, uint64_t releaseRate);

        bool IsEmpty() const noexcept;
        uint64_t Size() const noexcept;

        uint64_t GetReleasedSize() const noexcept { return Released_.size(); }
        uint64_t GetFullSize() const noexcept { return Full_.size(); }

        template<typename TGuard>
        NDetails::TStack GetStack(const TGuard& guard, const char* name);
        void ReturnStack(NDetails::TStack& stack);

    private:
        void ReleaseMemory(char* alignedStackMemory, uint64_t pagesToKeep) noexcept;

    private:
        TDeque<void*> Released_; //!< stacks memory with released RSS memory
        TDeque<void*> Full_;     //!< stacks memory with RSS memory
        uint64_t StackSize_ = 0;
        uint64_t RssPagesToKeep_ = 0;
        const uint64_t ReleaseRate_ = 1;
    };


    template<typename TGuard>
    NDetails::TStack TStorage::GetStack(const TGuard& guard, const char* name) {
        Y_VERIFY(!IsEmpty()); // check before call

        void* newStack = nullptr;
        if (!Full_.empty()) {
            newStack = Full_.back();
            Full_.pop_back();
        } else {
            Y_ASSERT(!Released_.empty());
            newStack = Released_.back();
            Released_.pop_back();
        }

        Y_VERIFY(guard.CheckOverflow(newStack), "corrupted stack in pool");
        Y_VERIFY(guard.CheckOverride(newStack, StackSize_), "corrupted stack in pool");

        return NDetails::TStack{newStack, newStack, StackSize_, name};
    }
}
