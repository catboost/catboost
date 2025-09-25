#include "rw_spin_lock.h"

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

void TReaderWriterSpinLock::AcquireReaderSlow() noexcept
{
    TSpinWait spinWait(Location_, ESpinLockActivityKind::Read);
    while (!TryAndTryAcquireReader()) {
        spinWait.Wait();
    }
}

void TReaderWriterSpinLock::AcquireReaderForkFriendlySlow() noexcept
{
    TSpinWait spinWait(Location_, ESpinLockActivityKind::Read);
    while (!TryAcquireReaderForkFriendly()) {
        spinWait.Wait();
    }
}

void TReaderWriterSpinLock::AcquireWriterSlow() noexcept
{
    TSpinWait spinWait(Location_, ESpinLockActivityKind::Write);
    while (!TryAndTryAcquireWriter()) {
        spinWait.Wait();
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
