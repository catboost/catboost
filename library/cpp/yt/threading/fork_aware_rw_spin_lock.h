#pragma once

#include "rw_spin_lock.h"

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

//! Wraps TReaderWriterSpinLock and additionally acquires a global fork lock (in read mode)
//! preventing concurrent forks from happening.
class TForkAwareReaderWriterSpinLock
{
public:
    TForkAwareReaderWriterSpinLock() = default;
    TForkAwareReaderWriterSpinLock(const TForkAwareReaderWriterSpinLock&) = delete;
    TForkAwareReaderWriterSpinLock& operator =(const TForkAwareReaderWriterSpinLock&) = delete;

    // TODO(babenko): make use of location.
    explicit constexpr TForkAwareReaderWriterSpinLock(const ::TSourceLocation& /*location*/)
    { }

    void AcquireReader() noexcept;
    void ReleaseReader() noexcept;

    void AcquireWriter() noexcept;
    void ReleaseWriter() noexcept;

    bool IsLocked() const noexcept;
    bool IsLockedByReader() const noexcept;
    bool IsLockedByWriter() const noexcept;

private:
    TReaderWriterSpinLock SpinLock_;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
