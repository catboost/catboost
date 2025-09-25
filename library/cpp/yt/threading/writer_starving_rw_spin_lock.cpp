#include "writer_starving_rw_spin_lock.h"

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

void TWriterStarvingRWSpinLock::AcquireReaderSlow() noexcept
{
    TSpinWait spinWait(Location_, ESpinLockActivityKind::Read);
    while (!TryAndTryAcquireReader()) {
        spinWait.Wait();
    }
}

void TWriterStarvingRWSpinLock::AcquireWriterSlow() noexcept
{
    TSpinWait spinWait(Location_, ESpinLockActivityKind::Write);
    while (!TryAndTryAcquireWriter()) {
        spinWait.Wait();
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
