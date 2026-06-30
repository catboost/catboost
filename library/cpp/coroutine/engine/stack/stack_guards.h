#pragma once

#include "stack_common.h"

#include <util/generic/array_ref.h>
#include <util/generic/strbuf.h>
#include <util/system/protect.h>


namespace NCoro::NStack {

    /*! Guard detect stack overflow/override, by setting memory before and after stack with predefined values/properties.
     * Actually, it sets memory only after the end of stack workspace memory - previous guard section should be set
     * already (for previous stack in case of pool allocator) and can be checked on demand.
     * Stack pointer should be page-aligned.
     */


    //! Checks integrity by writing a predefined sequence and comparing it with original
    class TCanaryGuard final {
    public:
        //! Size of guard section in bytes
        static constexpr size_t GetSize() { return Canary.size(); }
        //! Size of page-aligned guard section in bytes
        static constexpr size_t GetPageAlignedSize() { return AlignedSize_; }

        //! Get stack memory between guard sections
        static TArrayRef<char> GetWorkspace(void* stack, size_t size) noexcept {
            Y_ASSERT( !((size_t)stack & PageSizeMask) );
            Y_ASSERT( !(size & PageSizeMask) );
            Y_ASSERT(size > Canary.size());

            return {(char*)stack, size - Canary.size()};
        }

        /*! Set guard section before the end of stack memory (at stack + size - guard size position)
         *  checkPrevious: check guard before stack memory for integrity
         */
        static void Protect(void* stack, size_t size, bool checkPrevious) noexcept {
            Y_ASSERT( !((size_t)stack & PageSizeMask) ); // stack pointer should be page aligned
            Y_ASSERT( !(size & PageSizeMask) ); // stack size should be page aligned
            Y_ASSERT(size >= Canary.size()); // stack should have enough space to place guard

            if (checkPrevious) {
                Y_ABORT_UNLESS(CheckOverflow(stack), "Previous stack was corrupted");
            }
            auto guardPos = (char*) stack + size - Canary.size();
            memcpy(guardPos, Canary.data(), Canary.size());
        }

        //! This guard doesn't change memory flags
        static constexpr void RemoveProtection(void*, size_t) {}
        //! Should remove protection before returning memory to system
        static constexpr bool ShouldRemoveProtectionBeforeFree() { return false; }

        static bool CheckOverflow(void* stack) noexcept {
            Y_ASSERT(stack);

            char* guardPos = (char*) ((size_t)stack - Canary.size());
            return TStringBuf(guardPos, Canary.size()) == Canary;
        }

        static bool CheckOverride(void* stack, size_t size) noexcept {
            Y_ASSERT(stack);
            Y_ASSERT(size > Canary.size());

            char* guardPos = (char*) ((size_t)stack + size - Canary.size());
            return TStringBuf(guardPos, Canary.size()) == Canary;
        }

    private:
        static constexpr TStringBuf Canary = "[ThisIsACanaryCoroutineStackGuardIfYouReadThisTheStackIsStillOK]";
        static_assert(Canary.size() == 64);
        static constexpr size_t AlignedSize_ = (Canary.size() + PageSize - 1) & ~PageSizeMask;
    };


    // ------------------------------------------------------------------------
    //
    //! Ensures integrity by removing access rights for border pages
    class TPageGuard final {
    public:
        //! Size of guard section in bytes
        static constexpr size_t GetSize() { return PageSize; }
        //! Size of page-aligned guard section in bytes
        static constexpr size_t GetPageAlignedSize() { return PageSize; }

        static TArrayRef<char> GetWorkspace(void* stack, size_t size) noexcept {
            Y_ASSERT( !((size_t)stack & PageSizeMask) );
            Y_ASSERT( !(size & PageSizeMask) );
            Y_ASSERT(size > PageSize);

            return {(char*)stack, size - PageSize};
        }

        static void Protect(void* stack, size_t size, bool /*checkPrevious*/) noexcept {
            Y_ASSERT( !((size_t)stack & PageSizeMask) ); // stack pointer should be page aligned
            Y_ASSERT( !(size & PageSizeMask) ); // stack size should be page aligned
            Y_ASSERT(size >= PageSize); // stack should have enough space to place guard

            ProtectMemory((char*)stack + size - PageSize, PageSize, PM_NONE);
        }

        //! Remove protection, to allow stack memory be freed
        static void RemoveProtection(void* stack, size_t size) noexcept {
            Y_ASSERT( !((size_t)stack & PageSizeMask) );
            Y_ASSERT( !(size & PageSizeMask) );
            Y_ASSERT(size >= PageSize);

            ProtectMemory((char*)stack + size - PageSize, PageSize, PM_WRITE | PM_READ);
        }
        //! Should remove protection before returning memory to system
        static constexpr bool ShouldRemoveProtectionBeforeFree() { return true; }

        //! For page guard is not used - it crashes process at once in this case.
        static constexpr bool CheckOverflow(void*) { return true; }
        static constexpr bool CheckOverride(void*, size_t) { return true; }
    };


    template<typename TGuard>
    const TGuard& GetGuard() noexcept;
}
