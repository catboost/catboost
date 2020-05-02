#pragma once

#include "stackcollect.h"

#include <library/cpp/lfalloc/dbg_info/dbg_info.h>

#include <util/generic/noncopyable.h>
#include <util/stream/output.h>

namespace NAllocProfiler {

////////////////////////////////////////////////////////////////////////////////

inline int SetCurrentScopeTag(int value)
{
    return NAllocDbg::SetThreadAllocTag(value);
}

inline bool SetProfileCurrentThread(bool value)
{
    return NAllocDbg::SetProfileCurrentThread(value);
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
