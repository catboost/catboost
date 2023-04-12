#include "spin_lock.h"

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

void TSpinLock::AcquireSlow() noexcept
{
    TSpinWait spinWait(Location_, ESpinLockActivityKind::ReadWrite);
    while (!TryAndTryAcquire()) {
        spinWait.Wait();
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
