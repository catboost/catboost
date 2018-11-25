#include "profiler.h"

#include "stackcollect.h"

#include <util/generic/algorithm.h>
#include <util/generic/singleton.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/str.h>

namespace NAllocProfiler {

namespace {

static TAllocationStackCollector& AllocationStackCollector()
{
    return *Singleton<TAllocationStackCollector>();
}

int AllocationCallback(int tag, size_t size, int sizeIdx)
{
    Y_UNUSED(sizeIdx);

    static const size_t STACK_FRAMES_COUNT = 32;
    static const size_t STACK_FRAMES_SKIP = 1;

    void* frames[STACK_FRAMES_COUNT];
    size_t frameCount = BackTrace(frames, Y_ARRAY_SIZE(frames));
    if (frameCount <= STACK_FRAMES_SKIP) {
        return -1;
    }

    void** stack = &frames[STACK_FRAMES_SKIP];
    frameCount -= STACK_FRAMES_SKIP;

    auto& collector = AllocationStackCollector();
    return collector.Alloc(stack, frameCount, tag, size);
}

void DeallocationCallback(int stackId, int tag, size_t size, int sizeIdx)
{
    Y_UNUSED(tag);
    Y_UNUSED(sizeIdx);

    auto& collector = AllocationStackCollector();
    collector.Free(stackId, size);
}

}   // namespace

////////////////////////////////////////////////////////////////////////////////

bool StartAllocationSampling(bool profileAllThreads)
{
    auto& collector = AllocationStackCollector();
    collector.Clear();

    NAllocDbg::SetProfileAllThreads(profileAllThreads);
    NAllocDbg::SetAllocationCallback(AllocationCallback);
    NAllocDbg::SetDeallocationCallback(DeallocationCallback);
    NAllocDbg::SetAllocationSamplingEnabled(true);
    return true;
}

bool StopAllocationSampling(IAllocationStatsDumper &out, int count)
{
    NAllocDbg::SetAllocationCallback(nullptr);
    NAllocDbg::SetDeallocationCallback(nullptr);
    NAllocDbg::SetAllocationSamplingEnabled(false);

    auto& collector = AllocationStackCollector();
    collector.Dump(count, out);
    return true;
}

bool StopAllocationSampling(IOutputStream& out, int count) {
    TAllocationStatsDumper dumper(out);
    return StopAllocationSampling(dumper, count);
}

}   // namespace NProfiler
