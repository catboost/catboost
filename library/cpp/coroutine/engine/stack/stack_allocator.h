#pragma once

#include "stack.h"
#include "stack_common.h"

#include <util/generic/maybe.h>
#include <util/generic/noncopyable.h>
#include <util/generic/ptr.h>

#include <cstdint>


namespace NCoro::NStack {

    class IAllocator : private TNonCopyable {
    public:
        virtual ~IAllocator() = default;

        //! Size should be page-aligned. Stack would be protected by guard, thus, actual
        //! workspace for stack = size - size of guard.
        NDetails::TStack AllocStack(size_t size, const char* name) {
            size_t alignedSize = (size + PageSize - 1) & ~PageSizeMask;
            Y_ASSERT(alignedSize < 10 * 1024 * PageSize); // more than 10K pages for stack - do you really need it?
#if defined(_san_enabled_) || !defined(NDEBUG)
            alignedSize *= DebugOrSanStackMultiplier;
#endif
            return DoAllocStack(alignedSize, name);
        }

        void FreeStack(NDetails::TStack& stack) noexcept {
            if (stack.GetAlignedMemory()) {
                DoFreeStack(stack);
            }
        }

        virtual TAllocatorStats GetStackStats() const noexcept = 0;

        // Stack helpers
        virtual TArrayRef<char> GetStackWorkspace(void* stack, size_t size) noexcept = 0;
        virtual bool CheckStackOverflow(void* stack) const noexcept = 0;
        virtual bool CheckStackOverride(void* stack, size_t size) const noexcept = 0;

    private:
        virtual NDetails::TStack DoAllocStack(size_t size, const char* name) = 0;
        virtual void DoFreeStack(NDetails::TStack& stack) noexcept = 0;
    };

    THolder<IAllocator> GetAllocator(TMaybe<TPoolAllocatorSettings> poolSettings, EGuard guardType);

}

#include "stack_allocator.inl"
