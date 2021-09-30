#pragma once

#include <cstdint>

class TContExecutor;

namespace NCoro::NStack {

    static constexpr uint64_t PageSize = 4096;
    static constexpr uint64_t PageSizeMask = PageSize - 1; // for checks
    static constexpr uint64_t DebugOrSanStackMultiplier = 4; // for debug or sanitizer builds
    static constexpr uint64_t SmallStackMaxSizeInPages = 6;

    enum class EGuard {
        Canary, //!< writes some data to check it for corruption
        Page,   //!< prohibits access to page memory
    };

    struct TPoolAllocatorSettings {
        uint64_t RssPagesToKeep = 1;
        uint64_t SmallStackRssPagesToKeep = 1; // for stack less than SmallStackMaxSizeInPages
        uint64_t ReleaseRate = 2;
#if !defined(_san_enabled_) && defined(NDEBUG)
        uint64_t StacksPerChunk = 256;
#else
        uint64_t StacksPerChunk = 2;
#endif
    };

    struct TAllocatorStats {
        uint64_t ReleasedSize = 0;
        uint64_t NotReleasedSize = 0;
        uint64_t NumOfAllocated = 0;
    };
}
