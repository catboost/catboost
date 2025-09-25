#include "recursive_spin_lock.h"

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

void TRecursiveSpinLock::AcquireSlow() noexcept
{
    TSpinWait spinWait(Location_, ESpinLockActivityKind::ReadWrite);
    while (!TryAndTryAcquire()) {
        spinWait.Wait();
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
