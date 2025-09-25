#pragma once
#ifndef WRITER_STARVING_RW_SPIN_LOCK_INL_H_
#error "Direct inclusion of this file is not allowed, include rw_spin_lock.h"
// For the sake of sane code completion.
#include "writer_starving_rw_spin_lock.h"
#endif
#undef WRITER_STARVING_RW_SPIN_LOCK_INL_H_

#include "spin_wait.h"

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

inline void TWriterStarvingRWSpinLock::AcquireReader() noexcept
{
    if (TryAcquireReader()) {
        return;
    }
    AcquireReaderSlow();
}

inline void TWriterStarvingRWSpinLock::ReleaseReader() noexcept
{
    auto prevValue = Value_.fetch_sub(ReaderDelta, std::memory_order::release);
    Y_ASSERT((prevValue & ~WriterMask) != 0);
    NDetail::RecordSpinLockReleased();
}

inline void TWriterStarvingRWSpinLock::AcquireWriter() noexcept
{
    if (TryAcquireWriter()) {
        return;
    }
    AcquireWriterSlow();
}

inline void TWriterStarvingRWSpinLock::ReleaseWriter() noexcept
{
    auto prevValue = Value_.fetch_and(~WriterMask, std::memory_order::release);
    Y_ASSERT(prevValue & WriterMask);
    NDetail::RecordSpinLockReleased();
}

inline bool TWriterStarvingRWSpinLock::IsLocked() const noexcept
{
    return Value_.load() != UnlockedValue;
}

inline bool TWriterStarvingRWSpinLock::IsLockedByReader() const noexcept
{
    return Value_.load() >= ReaderDelta;
}

inline bool TWriterStarvingRWSpinLock::IsLockedByWriter() const noexcept
{
    return (Value_.load() & WriterMask) != 0;
}

inline bool TWriterStarvingRWSpinLock::TryAcquireReader() noexcept
{
    auto oldValue = Value_.fetch_add(ReaderDelta, std::memory_order::acquire);
    if ((oldValue & WriterMask) != 0) {
        Value_.fetch_sub(ReaderDelta, std::memory_order::relaxed);
        return false;
    }
    NDetail::RecordSpinLockAcquired();
    return true;
}

inline bool TWriterStarvingRWSpinLock::TryAndTryAcquireReader() noexcept
{
    auto oldValue = Value_.load(std::memory_order::relaxed);
    if ((oldValue & WriterMask) != 0) {
        return false;
    }
    return TryAcquireReader();
}

inline bool TWriterStarvingRWSpinLock::TryAcquireWriter() noexcept
{
    auto expected = UnlockedValue;
    bool acquired =  Value_.compare_exchange_weak(expected, WriterMask, std::memory_order::acquire);
    NDetail::MaybeRecordSpinLockAcquired(acquired);
    return acquired;
}

inline bool TWriterStarvingRWSpinLock::TryAndTryAcquireWriter() noexcept
{
    auto oldValue = Value_.load(std::memory_order::relaxed);
    if (oldValue != UnlockedValue) {
        return false;
    }
    return TryAcquireWriter();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

