#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/system/cpu_id.h>

#include <sched.h>

#include <vector>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

std::vector<int> GetAllowedCpus()
{
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return {};
    }
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &set)) {
            cpus.push_back(cpu);
        }
    }
    return cpus;
}

bool TryPinToCpu(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    // Pinning the current thread to a single allowed CPU: the kernel migrates us
    // onto it before sched_setaffinity returns, and keeps us there (it is the only
    // permitted CPU), so there is no migration race afterwards.
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

////////////////////////////////////////////////////////////////////////////////

// When pinned to a CPU, GetCurrentCpuId must report exactly that CPU -- this checks
// the actual contract (the real running CPU), not just a range, and is independent
// of how many CPUs are online or how they are numbered.
TEST(TGetCurrentCpuIdTest, MatchesPinnedCpu)
{
    auto allowedCpus = GetAllowedCpus();
    if (allowedCpus.empty() || !TryPinToCpu(allowedCpus.front())) {
        GTEST_SKIP() << "Cannot control CPU affinity in this environment";
    }

    for (int cpu : allowedCpus) {
        ASSERT_TRUE(TryPinToCpu(cpu));
        EXPECT_EQ(GetCurrentCpuId(), cpu);
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
