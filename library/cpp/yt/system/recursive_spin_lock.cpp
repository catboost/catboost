#include "recursive_spin_lock.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

void TRecursiveSpinLock::AcquireSlow() noexcept
{
    TSpinWait spinWait(Location_, ESpinLockActivityKind::ReadWrite);
    while (!TryAndTryAcquire()) {
        spinWait.Wait();
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
