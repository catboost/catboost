#include <gtest/gtest.h>

#include <library/cpp/yt/cpu_clock/clock.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

template <class T>
i64 DiffMS(T a, T b)
{
    return a >= b
        ? static_cast<i64>(a.MilliSeconds()) - static_cast<i64>(b.MilliSeconds())
        : DiffMS(b, a);
}

TEST(TTimingTest, GetInstant)
{
    GetInstant();

    EXPECT_LE(DiffMS(GetInstant(), TInstant::Now()), 10);
}

TEST(TTimingTest, InstantVSCpuInstant)
{
    auto instant1 = TInstant::Now();
    auto cpuInstant = InstantToCpuInstant(instant1);
    auto instant2 = CpuInstantToInstant(cpuInstant);
    EXPECT_LE(DiffMS(instant1, instant2), 10);
}

TEST(TTimingTest, DurationVSCpuDuration)
{
    auto cpuInstant1 = GetCpuInstant();
    constexpr auto duration1 = TDuration::MilliSeconds(100);
    Sleep(duration1);
    auto cpuInstant2 = GetCpuInstant();
    auto duration2 = CpuDurationToDuration(cpuInstant2 - cpuInstant1);
    EXPECT_LE(DiffMS(duration1, duration2), 10);
}

#if defined(_arm64_) && defined(_linux_)
TEST(TTimingTest, TicksPerSecondFromCntfrq)
{
    ui64 frequency;
    __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(frequency));
    ASSERT_NE(frequency, 0u);
    EXPECT_NEAR(DurationToCpuDuration(TDuration::Seconds(1)), static_cast<TCpuDuration>(frequency), 1);
}
#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
