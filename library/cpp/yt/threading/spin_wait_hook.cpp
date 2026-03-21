#include "spin_wait_hook.h"

#include <library/cpp/yt/assert/assert.h>

#include <array>
#include <atomic>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

static constexpr int MaxSpinWaitSlowPathHooks = 8;
static std::array<std::atomic<TSpinWaitSlowPathHook>, MaxSpinWaitSlowPathHooks> SpinWaitSlowPathHooks;
static std::atomic<int> SpinWaitSlowPathHookCount;

void RegisterSpinWaitSlowPathHook(TSpinWaitSlowPathHook hook)
{
    int index = SpinWaitSlowPathHookCount++;
    YT_VERIFY(index < MaxSpinWaitSlowPathHooks);
    SpinWaitSlowPathHooks[index].store(hook);
}

void InvokeSpinWaitSlowPathHooks(
    TCpuDuration cpuDelay,
    const ::TSourceLocation& location,
    ESpinLockActivityKind activityKind)
{
    for (const auto& atomicHook : SpinWaitSlowPathHooks) {
        if (auto hook = atomicHook.load()) {
            hook(cpuDelay, location, activityKind);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
