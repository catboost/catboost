#include "fork_aware_rw_spin_lock.h"

#include "at_fork.h"

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

void TForkAwareReaderWriterSpinLock::AcquireReader() noexcept
{
    GetForkLock()->AcquireReaderForkFriendly();
    SpinLock_.AcquireReader();
}

void TForkAwareReaderWriterSpinLock::ReleaseReader() noexcept
{
    SpinLock_.ReleaseReader();
    GetForkLock()->ReleaseReader();
}

void TForkAwareReaderWriterSpinLock::AcquireWriter() noexcept
{
    GetForkLock()->AcquireReaderForkFriendly();
    SpinLock_.AcquireWriter();
}

void TForkAwareReaderWriterSpinLock::ReleaseWriter() noexcept
{
    SpinLock_.ReleaseWriter();
    GetForkLock()->ReleaseReader();
}

bool TForkAwareReaderWriterSpinLock::IsLocked() const noexcept
{
    return SpinLock_.IsLocked();
}

bool TForkAwareReaderWriterSpinLock::IsLockedByReader() const noexcept
{
    return SpinLock_.IsLockedByReader();
}

bool TForkAwareReaderWriterSpinLock::IsLockedByWriter() const noexcept
{
    return SpinLock_.IsLockedByWriter();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

