#include "rw_spin_lock.h"
#include "util/system/backtrace.h"

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/misc/tls.h>

#include <util/generic/hash_set.h>

#include <thread>

namespace NYT::NThreading::NDetail {

////////////////////////////////////////////////////////////////////////////////

void TUncheckedReaderWriterSpinLock::AcquireReaderSlow() noexcept
{
    TSpinWait spinWait(Location_, ESpinLockActivityKind::Read);
    while (!TryAndTryAcquireReader()) {
        spinWait.Wait();
    }
}

void TUncheckedReaderWriterSpinLock::AcquireReaderForkFriendlySlow() noexcept
{
    TSpinWait spinWait(Location_, ESpinLockActivityKind::Read);
    while (!TryAcquireReaderForkFriendly()) {
        spinWait.Wait();
    }
}

void TUncheckedReaderWriterSpinLock::AcquireWriterSlow() noexcept
{
    TSpinWait spinWait(Location_, ESpinLockActivityKind::Write);
    while (!TryAndTryAcquireWriter()) {
        spinWait.Wait();
    }
}

////////////////////////////////////////////////////////////////////////////////

namespace {

// NB: Reading this after destruction is UB, but we can't deal with Static Initialization Order Fiasco in any other feasible way :(
YT_DEFINE_THREAD_LOCAL(bool, ThreadLockTrackerDestroyed, false);

class TThreadLockTracker
{
public:
    void RecordThreadLockAcquisition(TCheckedReaderWriterSpinLock* lock) noexcept
    {
        auto [_, inserted] = ThreadLocksAcquired_.insert(lock);
        YT_VERIFY(inserted, "Non-reentrant RWSpinLock detected two acquisitions in one thread");
    }

    void RecordThreadLockRelease(TCheckedReaderWriterSpinLock* lock) noexcept
    {
        YT_VERIFY(ThreadLocksAcquired_.erase(lock) == 1, "Released RWSpinLock that has never been acquired");
    }

    ~TThreadLockTracker()
    {
        ThreadLockTrackerDestroyed() = true;
    };

private:
    THashSet<TCheckedReaderWriterSpinLock*> ThreadLocksAcquired_;
};

YT_DEFINE_THREAD_LOCAL(TThreadLockTracker, ThreadLockTracker);

} // namespace

void TCheckedReaderWriterSpinLock::RecordThreadAcquisition(bool acquired)
{
    if (acquired) {
        // We might be called from destructor of another static variable after the destruction of actual TThreadLockTracker.
        if (ThreadLockTrackerDestroyed()) {
            return;
        }
        ThreadLockTracker().RecordThreadLockAcquisition(this);
    }
}

void TCheckedReaderWriterSpinLock::RecordThreadRelease()
{
    // We might be called from destructor of another static variable after the destruction of actual TThreadLockTracker.
    if (ThreadLockTrackerDestroyed()) {
        return;
    }
    ThreadLockTracker().RecordThreadLockRelease(this);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading::NDetail
