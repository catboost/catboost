#include "fork_aware_spin_lock.h"

#include "at_fork.h"

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

void TForkAwareSpinLock::Acquire() noexcept
{
    GetForkLock()->AcquireReaderForkFriendly();
    SpinLock_.Acquire();
}

bool TForkAwareSpinLock::TryAcquire() noexcept
{
    if (!GetForkLock()->TryAcquireReaderForkFriendly()) {
        return false;
    }
    if (!SpinLock_.TryAcquire()) {
        GetForkLock()->ReleaseReader();
        return false;
    }
    return true;
}

void TForkAwareSpinLock::Release() noexcept
{
    SpinLock_.Release();
    GetForkLock()->ReleaseReader();
}

bool TForkAwareSpinLock::IsLocked() const noexcept
{
    return SpinLock_.IsLocked();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

