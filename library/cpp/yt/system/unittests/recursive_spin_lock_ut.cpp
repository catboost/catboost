#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/system/recursive_spin_lock.h>
#include <library/cpp/yt/system/event_count.h>

#include <thread>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TRecursiveSpinLockTest, SingleThread)
{
    YT_DECLARE_SPIN_LOCK(TRecursiveSpinLock, lock);
    EXPECT_FALSE(lock.IsLocked());
    EXPECT_TRUE(lock.TryAcquire());
    EXPECT_TRUE(lock.IsLocked());
    EXPECT_TRUE(lock.TryAcquire());
    EXPECT_TRUE(lock.IsLocked());
    lock.Release();
    EXPECT_TRUE(lock.IsLocked());
    lock.Release();
    EXPECT_FALSE(lock.IsLocked());
    EXPECT_TRUE(lock.TryAcquire());
    EXPECT_TRUE(lock.IsLocked());
    lock.Release();
    lock.Acquire();
    lock.Release();
}

TEST(TRecursiveSpinLockTest, TwoThreads)
{
    YT_DECLARE_SPIN_LOCK(TRecursiveSpinLock, lock);
    TEvent e1, e2, e3, e4, e5, e6, e7;

    std::jthread t1([&] {
        e1.Wait();
        EXPECT_TRUE(lock.IsLocked());
        EXPECT_FALSE(lock.IsLockedByCurrentThread());
        EXPECT_FALSE(lock.TryAcquire());
        e2.NotifyOne();
        e3.Wait();
        EXPECT_TRUE(lock.IsLocked());
        EXPECT_FALSE(lock.IsLockedByCurrentThread());
        EXPECT_FALSE(lock.TryAcquire());
        e4.NotifyOne();
        e5.Wait();
        EXPECT_FALSE(lock.IsLocked());
        EXPECT_FALSE(lock.IsLockedByCurrentThread());
        EXPECT_TRUE(lock.TryAcquire());
        e6.NotifyOne();
        e7.Wait();
        lock.Release();
    });

    std::jthread t2([&] {
        EXPECT_FALSE(lock.IsLocked());
        EXPECT_TRUE(lock.TryAcquire());
        EXPECT_TRUE(lock.IsLockedByCurrentThread());
        e1.NotifyOne();
        e2.Wait();
        EXPECT_TRUE(lock.TryAcquire());
        EXPECT_TRUE(lock.IsLockedByCurrentThread());
        e3.NotifyOne();
        e4.Wait();
        lock.Release();
        lock.Release();
        EXPECT_FALSE(lock.IsLocked());
        e5.NotifyOne();
        e6.Wait();
        EXPECT_TRUE(lock.IsLocked());
        EXPECT_FALSE(lock.IsLockedByCurrentThread());
        e7.NotifyOne();
        lock.Acquire();
        lock.Acquire();
        lock.Release();
        lock.Release();
    });
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
