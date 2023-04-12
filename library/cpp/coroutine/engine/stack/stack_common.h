#pragma once

#include <cstddef>

class TContExecutor;

namespace NCoro::NStack {

    static constexpr size_t PageSize = 4096;
    static constexpr size_t PageSizeMask = PageSize - 1; // for checks
    static constexpr size_t DebugOrSanStackMultiplier = 4; // for debug or sanitizer builds
    static constexpr size_t SmallStackMaxSizeInPages = 6;

    enum class EGuard {
        Canary, //!< writes some data to check it for corruption
        Page,   //!< prohibits access to page memory
    };

    struct TPoolAllocatorSettings {
        size_t RssPagesToKeep = 1;
        size_t SmallStackRssPagesToKeep = 1; // for stack less than SmallStackMaxSizeInPages
        size_t ReleaseRate = 8;
#if !defined(_san_enabled_) && defined(NDEBUG)
        size_t StacksPerChunk = 1024;
#else
        size_t StacksPerChunk = 2;
#endif
    };

    struct TAllocatorStats {
        size_t ReleasedSize = 0;
        size_t NotReleasedSize = 0;
        size_t NumOfAllocated = 0;
    };
}
