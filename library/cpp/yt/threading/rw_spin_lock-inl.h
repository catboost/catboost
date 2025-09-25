#pragma once
#ifndef RW_SPIN_LOCK_INL_H_
#error "Direct inclusion of this file is not allowed, include rw_spin_lock.h"
// For the sake of sane code completion.
#include "rw_spin_lock.h"
#endif
#undef RW_SPIN_LOCK_INL_H_

#include "spin_wait.h"

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

inline void TReaderWriterSpinLock::AcquireReader() noexcept
{
    if (TryAcquireReader()) {
        return;
    }
    AcquireReaderSlow();
}

inline void TReaderWriterSpinLock::AcquireReaderForkFriendly() noexcept
{
    if (TryAcquireReaderForkFriendly()) {
        return;
    }
    AcquireReaderForkFriendlySlow();
}

inline void TReaderWriterSpinLock::ReleaseReader() noexcept
{
    auto prevValue = Value_.fetch_sub(ReaderDelta, std::memory_order::release);
    Y_ASSERT((prevValue & ~(WriterMask | WriterReadyMask)) != 0);
    NDetail::RecordSpinLockReleased();
}

inline void TReaderWriterSpinLock::AcquireWriter() noexcept
{
    if (TryAcquireWriter()) {
        return;
    }
    AcquireWriterSlow();
}

inline void TReaderWriterSpinLock::ReleaseWriter() noexcept
{
    auto prevValue = Value_.fetch_and(~(WriterMask | WriterReadyMask), std::memory_order::release);
    Y_ASSERT(prevValue & WriterMask);
    NDetail::RecordSpinLockReleased();
}

inline bool TReaderWriterSpinLock::IsLocked() const noexcept
{
    return (Value_.load() & ~WriterReadyMask) != 0;
}

inline bool TReaderWriterSpinLock::IsLockedByReader() const noexcept
{
    return Value_.load() >= ReaderDelta;
}

inline bool TReaderWriterSpinLock::IsLockedByWriter() const noexcept
{
    return (Value_.load() & WriterMask) != 0;
}

inline bool TReaderWriterSpinLock::TryAcquireReader() noexcept
{
    auto oldValue = Value_.fetch_add(ReaderDelta, std::memory_order::acquire);
    if ((oldValue & (WriterMask | WriterReadyMask)) != 0) {
        Value_.fetch_sub(ReaderDelta, std::memory_order::relaxed);
        return false;
    }
    NDetail::RecordSpinLockAcquired();
    return true;
}

inline bool TReaderWriterSpinLock::TryAndTryAcquireReader() noexcept
{
    auto oldValue = Value_.load(std::memory_order::relaxed);
    if ((oldValue & (WriterMask | WriterReadyMask)) != 0) {
        return false;
    }
    return TryAcquireReader();
}

inline bool TReaderWriterSpinLock::TryAcquireReaderForkFriendly() noexcept
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

inline bool TReaderWriterSpinLock::TryAcquireWriterWithExpectedValue(TValue expected) noexcept
{
    bool acquired = Value_.compare_exchange_weak(expected, WriterMask, std::memory_order::acquire);
    NDetail::MaybeRecordSpinLockAcquired(acquired);
    return acquired;
}

inline bool TReaderWriterSpinLock::TryAcquireWriter() noexcept
{
    // NB(pavook): we cannot expect writer ready to be set, as this method
    // might be called without indicating writer readiness and we cannot
    // indicate readiness on the hot path. This means that code calling
    // TryAcquireWriter will spin against code calling AcquireWriter.
    return TryAcquireWriterWithExpectedValue(UnlockedValue);
}

inline bool TReaderWriterSpinLock::TryAndTryAcquireWriter() noexcept
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

