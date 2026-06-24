#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/rseq/per_cpu.h>
#include <library/cpp/yt/rseq/rseq.h>

#include <library/cpp/yt/memory/public.h>

#include <util/system/types.h>

#include <atomic>
#include <iterator>
#include <thread>
#include <vector>

namespace NYT::NRseq {
namespace {

////////////////////////////////////////////////////////////////////////////////

// A per-CPU i64 accumulator built on the rseq primitives, mirroring how a profiling
// counter would use them.
class TPerCpuI64
{
public:
    TPerCpuI64()
        : Slots_(GetCpuCount())
    { }

    void Add(i64 value)
    {
        AddPerCpu(Slots_.data(), &TSlot::Value, value);
    }

    i64 GetValue() const
    {
        i64 total = 0;
        for (int index = 0; index < std::ssize(Slots_); ++index) {
            total += LoadPerCpu(Slots_.data(), &TSlot::Value, index);
        }
        return total;
    }

private:
    struct alignas(CacheLineSize) TSlot
    {
        i64 Value = 0;
    };

    static_assert(sizeof(TSlot) == CacheLineSize);

    std::vector<TSlot> Slots_;
};

////////////////////////////////////////////////////////////////////////////////

TEST(TPerCpuRseqTest, CpuCountIsSane)
{
    EXPECT_GE(GetCpuCount(), 1);
    EXPECT_LE(GetCpuCount(), 1 << 20);
}

TEST(TPerCpuRseqTest, FastPathSupportIsStable)
{
    // The probe spawns a thread on first use and caches its verdict, so repeated calls must
    // agree. We avoid asserting a specific value: it depends on kernel rseq support.
    bool supported = IsPerCpuFastPathSupported();
    EXPECT_EQ(supported, IsPerCpuFastPathSupported());
}

TEST(TPerCpuRseqTest, ParsePossibleCpuCount)
{
    using NDetail::ParsePossibleCpuCount;
    EXPECT_EQ(ParsePossibleCpuCount("0"), 1);
    EXPECT_EQ(ParsePossibleCpuCount("0-3"), 4);
    EXPECT_EQ(ParsePossibleCpuCount("0-63"), 64);
    // Sparse mask: the bound is the highest id + 1 (12), not the CPU popcount (8).
    EXPECT_EQ(ParsePossibleCpuCount("0-3,8-11"), 12);
    EXPECT_EQ(ParsePossibleCpuCount("0-3\n"), 4);
    EXPECT_EQ(ParsePossibleCpuCount(""), -1);
    EXPECT_EQ(ParsePossibleCpuCount("\n"), -1);
}

TEST(TPerCpuRseqTest, SingleThreadAccumulates)
{
    TPerCpuI64 counter;
    constexpr i64 Iterations = 1'000'000;
    for (i64 i = 0; i < Iterations; ++i) {
        counter.Add(1);
    }
    EXPECT_EQ(counter.GetValue(), Iterations);
}

TEST(TPerCpuRseqTest, SingleThreadHandlesNegativeAndLargeDeltas)
{
    TPerCpuI64 counter;
    counter.Add(1'000'000'000'000LL);
    counter.Add(-7);
    counter.Add(-1'000'000'000'000LL);
    EXPECT_EQ(counter.GetValue(), -7);
}

// The core correctness guarantee: across many threads (which the scheduler migrates
// between CPUs, exercising rseq aborts/restarts), not a single increment is lost.
TEST(TPerCpuRseqTest, ConcurrentNoLostUpdates)
{
    TPerCpuI64 counter;

    const int threadCount = std::max<int>(4, std::thread::hardware_concurrency());
    constexpr i64 PerThread = 2'000'000;

    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < threadCount; ++t) {
        threads.emplace_back([&] {
            while (!start.load(std::memory_order::acquire)) {
            }
            for (i64 i = 0; i < PerThread; ++i) {
                counter.Add(1);
            }
        });
    }
    start.store(true, std::memory_order::release);
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(counter.GetValue(), static_cast<i64>(threadCount) * PerThread);
}

// Independent counters updated concurrently must not interfere with each other.
TEST(TPerCpuRseqTest, IndependentCountersDoNotInterfere)
{
    TPerCpuI64 a;
    TPerCpuI64 b;

    const int threadCount = std::max<int>(4, std::thread::hardware_concurrency());
    constexpr i64 PerThread = 1'000'000;

    std::vector<std::thread> threads;
    for (int t = 0; t < threadCount; ++t) {
        threads.emplace_back([&, t] {
            for (i64 i = 0; i < PerThread; ++i) {
                a.Add(1);
                if (t % 2 == 0) {
                    b.Add(2);
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(a.GetValue(), static_cast<i64>(threadCount) * PerThread);
    EXPECT_EQ(b.GetValue(), static_cast<i64>((threadCount + 1) / 2) * PerThread * 2);
}

////////////////////////////////////////////////////////////////////////////////

struct TPair
{
    ui64 A;
    ui64 B;
};

struct alignas(CacheLineSize) TPairSlot
{
    TPair Value{};
};

static_assert(sizeof(TPairSlot) == CacheLineSize);

TEST(TPerCpuRseqTest, StorePerCpuPublishesValue)
{
    std::vector<TPairSlot> slots(GetCpuCount());
    constexpr ui64 Last = 100'000;
    for (ui64 i = 1; i <= Last; ++i) {
        StorePerCpu(slots.data(), &TPairSlot::Value, TPair{i, i});
    }

    bool foundLast = false;
    for (const auto& slot : slots) {
        // No store ever writes mismatched halves, so any populated slot must be consistent.
        EXPECT_EQ(slot.Value.A, slot.Value.B);
        if (slot.Value.A == Last) {
            foundLast = true;
        }
    }
    EXPECT_TRUE(foundLast);
}

////////////////////////////////////////////////////////////////////////////////

struct alignas(CacheLineSize) TWordSlot
{
    ui64 Value = 0;
};

static_assert(sizeof(TWordSlot) == CacheLineSize);

TEST(TPerCpuRseqTest, StorePerCpu8PublishesValue)
{
    std::vector<TWordSlot> slots(GetCpuCount());
    constexpr ui64 Last = 100'000;
    for (ui64 i = 1; i <= Last; ++i) {
        StorePerCpu(slots.data(), &TWordSlot::Value, i);
    }

    bool foundLast = false;
    for (const auto& slot : slots) {
        if (slot.Value == Last) {
            foundLast = true;
        }
    }
    EXPECT_TRUE(foundLast);
}

////////////////////////////////////////////////////////////////////////////////

// LoadPerCpu must read exactly the requested slot (verifies the base + index * stride
// addressing), independent of the calling CPU.
TEST(TPerCpuRseqTest, LoadPerCpuReadsRequestedSlot)
{
    std::vector<TWordSlot> slots(GetCpuCount());
    for (int index = 0; index < std::ssize(slots); ++index) {
        slots[index].Value = static_cast<ui64>(index) * 100 + 1;
    }
    for (int index = 0; index < std::ssize(slots); ++index) {
        EXPECT_EQ(
            LoadPerCpu(slots.data(), &TWordSlot::Value, index),
            static_cast<ui64>(index) * 100 + 1);
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT::NRseq
