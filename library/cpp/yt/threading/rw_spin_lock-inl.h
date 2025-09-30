#pragma once
#ifndef RW_SPIN_LOCK_INL_H_
#error "Direct inclusion of this file is not allowed, include rw_spin_lock.h"
// For the sake of sane code completion.
#include "rw_spin_lock.h"
#endif
#undef RW_SPIN_LOCK_INL_H_

#include "spin_wait.h"

namespace NYT::NThreading {
namespace NDetail {

////////////////////////////////////////////////////////////////////////////////

inline void TUncheckedReaderWriterSpinLock::AcquireReader() noexcept
{
    if (TryAcquireReader()) {
        return;
    }
    AcquireReaderSlow();
}

inline void TUncheckedReaderWriterSpinLock::AcquireReaderForkFriendly() noexcept
{
    if (TryAcquireReaderForkFriendly()) {
        return;
    }
    AcquireReaderForkFriendlySlow();
}

inline void TUncheckedReaderWriterSpinLock::ReleaseReader() noexcept
{
    auto prevValue = Value_.fetch_sub(ReaderDelta, std::memory_order::release);
    Y_ASSERT((prevValue & ~(WriterMask | WriterReadyMask)) != 0);
    NDetail::RecordSpinLockReleased();
}

inline void TUncheckedReaderWriterSpinLock::AcquireWriter() noexcept
{
    if (TryAcquireWriter()) {
        return;
    }
    AcquireWriterSlow();
}

inline void TUncheckedReaderWriterSpinLock::ReleaseWriter() noexcept
{
    auto prevValue = Value_.fetch_and(~(WriterMask | WriterReadyMask), std::memory_order::release);
    Y_ASSERT(prevValue & WriterMask);
    NDetail::RecordSpinLockReleased();
}

inline bool TUncheckedReaderWriterSpinLock::IsLocked() const noexcept
{
    return (Value_.load() & ~WriterReadyMask) != 0;
}

inline bool TUncheckedReaderWriterSpinLock::IsLockedByReader() const noexcept
{
    return Value_.load() >= ReaderDelta;
}

inline bool TUncheckedReaderWriterSpinLock::IsLockedByWriter() const noexcept
{
    return (Value_.load() & WriterMask) != 0;
}

inline bool TUncheckedReaderWriterSpinLock::TryAcquireReader() noexcept
{
    auto oldValue = Value_.fetch_add(ReaderDelta, std::memory_order::acquire);
    if ((oldValue & (WriterMask | WriterReadyMask)) != 0) {
        Value_.fetch_sub(ReaderDelta, std::memory_order::relaxed);
        return false;
    }
    NDetail::RecordSpinLockAcquired();
    return true;
}

inline bool TUncheckedReaderWriterSpinLock::TryAndTryAcquireReader() noexcept
{
    auto oldValue = Value_.load(std::memory_order::relaxed);
    if ((oldValue & (WriterMask | WriterReadyMask)) != 0) {
        return false;
    }
    return TryAcquireReader();
}

inline bool TUncheckedReaderWriterSpinLock::TryAcquireReaderForkFriendly() noexcept
{
    auto oldValue = Value_.load(std::memory_order::relaxed);
    if ((oldValue & (WriterMask | WriterReadyMask)) != 0) {
        return false;
    }
    auto newValue = oldValue + ReaderDelta;

    bool acquired = Value_.compare_exchange_weak(oldValue, newValue, std::memory_order::acquire);
    NDetail::MaybeRecordSpinLockAcquired(acquired);
    return acquired;
}

inline bool TUncheckedReaderWriterSpinLock::TryAcquireWriterWithExpectedValue(TValue expected) noexcept
{
    bool acquired = Value_.compare_exchange_weak(expected, WriterMask, std::memory_order::acquire);
    NDetail::MaybeRecordSpinLockAcquired(acquired);
    return acquired;
}

inline bool TUncheckedReaderWriterSpinLock::TryAcquireWriter() noexcept
{
    // NB(pavook): we cannot expect writer ready to be set, as this method
    // might be called without indicating writer readiness and we cannot
    // indicate readiness on the hot path. This means that code calling
    // TryAcquireWriter will spin against code calling AcquireWriter.
    return TryAcquireWriterWithExpectedValue(UnlockedValue);
}

inline bool TUncheckedReaderWriterSpinLock::TryAndTryAcquireWriter() noexcept
{
    auto oldValue = Value_.load(std::memory_order::relaxed);

    if ((oldValue & WriterReadyMask) == 0) {
        oldValue = Value_.fetch_or(WriterReadyMask, std::memory_order::relaxed);
    }

    if ((oldValue & (~WriterReadyMask)) != 0) {
        return false;
    }

    return TryAcquireWriterWithExpectedValue(WriterReadyMask);
}

////////////////////////////////////////////////////////////////////////////////

inline void TCheckedReaderWriterSpinLock::AcquireReader() noexcept
{
    RecordThreadAcquisition(true);
    TUncheckedReaderWriterSpinLock::AcquireReader();
}

inline void TCheckedReaderWriterSpinLock::AcquireReaderForkFriendly() noexcept
{
    RecordThreadAcquisition(true);
    TUncheckedReaderWriterSpinLock::AcquireReaderForkFriendly();
}

inline void TCheckedReaderWriterSpinLock::AcquireWriter() noexcept
{
    RecordThreadAcquisition(true);
    TUncheckedReaderWriterSpinLock::AcquireWriter();
}

inline bool TCheckedReaderWriterSpinLock::TryAcquireReader() noexcept
{
    RecordThreadAcquisition(true);
    bool acquired = TUncheckedReaderWriterSpinLock::TryAcquireReader();
    if (!acquired) {
        RecordThreadRelease();
    }
    return acquired;
}

inline bool TCheckedReaderWriterSpinLock::TryAcquireReaderForkFriendly() noexcept
{
    RecordThreadAcquisition(true);
    bool acquired = TUncheckedReaderWriterSpinLock::TryAcquireReaderForkFriendly();
    if (!acquired) {
        RecordThreadRelease();
    }
    return acquired;
}

inline bool TCheckedReaderWriterSpinLock::TryAcquireWriter() noexcept
{
    RecordThreadAcquisition(true);
    bool acquired = TUncheckedReaderWriterSpinLock::TryAcquireWriter();
    if (!acquired) {
        RecordThreadRelease();
    }
    return acquired;
}

inline void TCheckedReaderWriterSpinLock::ReleaseReader() noexcept
{
    RecordThreadRelease();
    TUncheckedReaderWriterSpinLock::ReleaseReader();
}

inline void TCheckedReaderWriterSpinLock::ReleaseWriter() noexcept
{
    RecordThreadRelease();
    TUncheckedReaderWriterSpinLock::ReleaseWriter();
}

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

template <class T>
void TReaderSpinlockTraits<T>::Acquire(T* spinlock)
{
   spinlock->AcquireReader();
}

template <class T>
void TReaderSpinlockTraits<T>::Release(T* spinlock)
{
    spinlock->ReleaseReader();
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
void TForkFriendlyReaderSpinlockTraits<T>::Acquire(T* spinlock)
{
    spinlock->AcquireReaderForkFriendly();
}

template <class T>
void TForkFriendlyReaderSpinlockTraits<T>::Release(T* spinlock)
{
    spinlock->ReleaseReader();
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
void TWriterSpinlockTraits<T>::Acquire(T* spinlock)
{
    spinlock->AcquireWriter();
}

template <class T>
void TWriterSpinlockTraits<T>::Release(T* spinlock)
{
    spinlock->ReleaseWriter();
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
auto ReaderGuard(const T& lock)
{
    return TReaderGuard<T>(lock);
}

template <class T>
auto ReaderGuard(const T* lock)
{
    return TReaderGuard<T>(lock);
}

template <class T>
auto ForkFriendlyReaderGuard(const T& lock)
{
    return TGuard<T, TForkFriendlyReaderSpinlockTraits<T>>(lock);
}

template <class T>
auto ForkFriendlyReaderGuard(const T* lock)
{
    return TGuard<T, TForkFriendlyReaderSpinlockTraits<T>>(lock);
}

template <class T>
auto WriterGuard(const T& lock)
{
    return TWriterGuard<T>(lock);
}

template <class T>
auto WriterGuard(const T* lock)
{
    return TWriterGuard<T>(lock);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
