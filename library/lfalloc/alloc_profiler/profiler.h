#pragma once

#include "stackcollect.h"

#if defined(PROFILE_MEMORY_ALLOCATIONS)
#include <library/lfalloc/dbg_info/dbg_info.h>
#endif

#include <util/generic/noncopyable.h>
#include <util/stream/output.h>

namespace NAllocProfiler {

////////////////////////////////////////////////////////////////////////////////

inline int SetCurrentScopeTag(int value)
{
#if defined(PROFILE_MEMORY_ALLOCATIONS)
    return NAllocDbg::SetThreadAllocTag(value);
#else
    Y_UNUSED(value);
    return 0;
#endif
}

inline bool SetProfileCurrentThread(bool value)
{
#if defined(PROFILE_MEMORY_ALLOCATIONS)
    return NAllocDbg::SetProfileCurrentThread(value);
#else
    Y_UNUSED(value);
    return false;
#endif
}

bool StartAllocationSampling(bool profileAllThreads = false);
bool StopAllocationSampling(IAllocationStatsDumper& out, int count = 100);
bool StopAllocationSampling(IOutputStream& out, int count = 100);

////////////////////////////////////////////////////////////////////////////////

class TProfilingScope: private TNonCopyable {
private:
    const int Prev;

public:
    explicit TProfilingScope(int value)
        : Prev(SetCurrentScopeTag(value))
    {}

    ~TProfilingScope()
    {
        SetCurrentScopeTag(Prev);
    }
};

}   // namespace NAllocProfiler
