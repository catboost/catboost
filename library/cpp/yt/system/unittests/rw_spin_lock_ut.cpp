#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/system/rw_spin_lock.h>

#include <util/thread/pool.h>

#include <latch>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TReaderWriterSpinLockTest, WriterPriority)
{
    int readerThreads = 10;
    std::latch latch(readerThreads + 1);
    std::atomic<size_t> finishedCount = {0};

    YT_DECLARE_SPIN_LOCK(TReaderWriterSpinLock, lock);

    volatile std::atomic<ui32> x = {0};

    auto readerTask = [&latch, &lock, &finishedCount, &x] () {
        latch.arrive_and_wait();
        while (true) {
            {
                auto guard = ReaderGuard(lock);
                // do some stuff
                for (ui32 i = 0; i < 10'000u; ++i) {
                    x.fetch_add(i);
                }
            }
            if (finishedCount.fetch_add(1) > 20'000) {
                break;
            }
        }
    };

    auto readerPool = CreateThreadPool(readerThreads);
    for (int i = 0; i < readerThreads; ++i) {
        readerPool->SafeAddFunc(readerTask);
    }

    latch.arrive_and_wait();
    while (finishedCount.load() == 0);
    auto guard = WriterGuard(lock);
    EXPECT_LE(finishedCount.load(), 1'000u);
    DoNotOptimizeAway(x);
}

TEST(TReaderWriterSpinLockDeathTest, ReaderReentrance)
{
    YT_DECLARE_SPIN_LOCK(TReaderWriterSpinLock, lock);

    EXPECT_DEBUG_DEATH({
        auto guard1 = ReaderGuard(lock);
        auto guard2 = ReaderGuard(lock);
    }, "two acquisitions in one thread");
}

TEST(TReaderWriterSpinLockDeathTest, MixedReentrance)
{
    YT_DECLARE_SPIN_LOCK(NDetail::TCheckedReaderWriterSpinLock, lock);

    EXPECT_DEATH({
        auto guard1 = ReaderGuard(lock);
        auto guard2 = WriterGuard(lock);
    }, "two acquisitions in one thread");
}

TEST(TReaderWriterSpinLockDeathTest, TryReaderReentrance)
{
    YT_DECLARE_SPIN_LOCK(TReaderWriterSpinLock, lock);

    EXPECT_DEBUG_DEATH({
        auto guard = ReaderGuard(lock);
        lock.TryAcquireReader();
    }, "two acquisitions in one thread");
}

TEST(TReaderWriterSpinLockDeathTest, TryWriterReentrance)
{
    YT_DECLARE_SPIN_LOCK(TReaderWriterSpinLock, lock);

    EXPECT_DEBUG_DEATH({
        auto guard = WriterGuard(lock);
        lock.TryAcquireWriter();
    }, "two acquisitions in one thread");
}

TEST(TReaderWriterSpinLockDeathTest, ReleaseUnacquiredLock)
{
    YT_DECLARE_SPIN_LOCK(TReaderWriterSpinLock, lock);

    EXPECT_DEBUG_DEATH({
        lock.ReleaseReader();
    }, "has never been acquired");
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
