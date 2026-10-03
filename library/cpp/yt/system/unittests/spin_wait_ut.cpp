#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/system/spin_wait.h>
#include <library/cpp/yt/system/spin_wait_hook.h>

#include <mutex>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

bool SpinWaitSlowPathHookInvoked;

void SpinWaitSlowPathHook(
    TCpuDuration cpuDelay,
    const TSourceLocation& /*location*/,
    ESpinLockActivityKind /*activityKind*/)
{
    SpinWaitSlowPathHookInvoked = true;
    auto delay = CpuDurationToDuration(cpuDelay);
    EXPECT_GE(delay, TDuration::Seconds(1));
    EXPECT_LE(delay, TDuration::Seconds(5));
}

TEST(TSpinWaitTest, SlowPathHook)
{
    static std::once_flag registerFlag;
    std::call_once(
        registerFlag,
        [] {
            RegisterSpinWaitSlowPathHook(SpinWaitSlowPathHook);
        });
    SpinWaitSlowPathHookInvoked = false;
    {
        TSpinWait spinWait(__LOCATION__, ESpinLockActivityKind::ReadWrite);
        for (int i = 0; i < 1'000'000; ++i) {
            spinWait.Wait();
        }
    }
    EXPECT_TRUE(SpinWaitSlowPathHookInvoked);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
