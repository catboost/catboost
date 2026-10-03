#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/system/spin_lock.h>
#include <library/cpp/yt/system/recursive_spin_lock.h>
#include <library/cpp/yt/system/rw_spin_lock.h>
#include <library/cpp/yt/system/fork_aware_spin_lock.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

#ifndef NDEBUG

TEST(TSpinLockCountTest, SpinLock)
{
    EXPECT_EQ(GetActiveSpinLockCount(), 0);
    YT_DECLARE_SPIN_LOCK(TSpinLock, lock);
    auto guard = Guard(lock);
    EXPECT_EQ(GetActiveSpinLockCount(), 1);
    guard.Release();
    EXPECT_EQ(GetActiveSpinLockCount(), 0);
}

TEST(TSpinLockCountTest, RecursiveSpinLock)
{
    EXPECT_EQ(GetActiveSpinLockCount(), 0);
    YT_DECLARE_SPIN_LOCK(TRecursiveSpinLock, lock);
    auto guard1 = Guard(lock);
    EXPECT_EQ(GetActiveSpinLockCount(), 1);
    auto guard2 = Guard(lock);
    EXPECT_EQ(GetActiveSpinLockCount(), 2);
    guard2.Release();
    EXPECT_EQ(GetActiveSpinLockCount(), 1);
    guard1.Release();
    EXPECT_EQ(GetActiveSpinLockCount(), 0);
}

TEST(TSpinLockCountTest, ForkAwareSpinLock)
{
    EXPECT_EQ(GetActiveSpinLockCount(), 0);
    YT_DECLARE_SPIN_LOCK(TForkAwareSpinLock, lock);
    auto guard = Guard(lock);
    EXPECT_EQ(GetActiveSpinLockCount(), 1);
    guard.Release();
    EXPECT_EQ(GetActiveSpinLockCount(), 0);
}

TEST(TSpinLockCountTest, RWSpinLock)
{
    EXPECT_EQ(GetActiveSpinLockCount(), 0);
    YT_DECLARE_SPIN_LOCK(TReaderWriterSpinLock, lock);
    auto readerGuard = ReaderGuard(lock);
    EXPECT_EQ(GetActiveSpinLockCount(), 1);
    readerGuard.Release();
    EXPECT_EQ(GetActiveSpinLockCount(), 0);
    auto writerGuard = WriterGuard(lock);
    EXPECT_EQ(GetActiveSpinLockCount(), 1);
    writerGuard.Release();
    EXPECT_EQ(GetActiveSpinLockCount(), 0);
}

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
