#pragma once
#ifndef RW_SPINLOCK_INL_H_
#error "Direct inclusion of this file is not allowed, include rw_spinlock.h"
// For the sake of sane code completion.
#include "rw_spinlock.h"
#endif
#undef RW_SPINLOCK_INL_H_

#include <util/system/spin_wait.h>

namespace NYT::NConcurrency {

////////////////////////////////////////////////////////////////////////////////

inline void TReaderWriterSpinLock::AcquireReader() noexcept
{
    if (TryAcquireReader()) {
        return;
    }
    TSpinWait spinWait;
    while (!TryAndTryAcquireReader()) {
        spinWait.Sleep();
    }
}

inline void TReaderWriterSpinLock::AcquireReaderForkFriendly() noexcept
{
    TSpinWait spinWait;
    while (!TryAcquireReaderForkFriendly()) {
        spinWait.Sleep();
    }
}

inline void TReaderWriterSpinLock::ReleaseReader() noexcept
{
    auto prevValue = Value_.fetch_sub(ReaderDelta, std::memory_order_release);
    Y_ASSERT((prevValue & ~WriterMask) != 0);
}

inline void TReaderWriterSpinLock::AcquireWriter() noexcept
{
    if (TryAcquireWriter()) {
        return;
    }
    TSpinWait spinWait;
    while (!TryAndTryAcquireWriter()) {
        spinWait.Sleep();
    }
}

inline void TReaderWriterSpinLock::ReleaseWriter() noexcept
{
    auto prevValue = Value_.fetch_and(~WriterMask, std::memory_order_release);
    Y_ASSERT(prevValue & WriterMask);
}

inline bool TReaderWriterSpinLock::IsLocked() const noexcept
{
    return Value_.load() != 0;
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
    auto oldValue = Value_.fetch_add(ReaderDelta, std::memory_order_acquire);
    if ((oldValue & WriterMask) != 0) {
        Value_.fetch_sub(ReaderDelta, std::memory_order_relaxed);
        return false;
    }
    return true;
}

inline bool TReaderWriterSpinLock::TryAndTryAcquireReader() noexcept
{
    auto oldValue = Value_.load(std::memory_order_relaxed);
    if ((oldValue & WriterMask) != 0) {
        return false;
    }
    return TryAcquireReader();
}

inline bool TReaderWriterSpinLock::TryAcquireReaderForkFriendly() noexcept
{
    auto oldValue = Value_.load(std::memory_order_relaxed);
    if ((oldValue & WriterMask) != 0) {
        return false;
    }
    auto newValue = oldValue + ReaderDelta;
    return Value_.compare_exchange_weak(oldValue, newValue, std::memory_order_acquire);
}

inline bool TReaderWriterSpinLock::TryAcquireWriter() noexcept
{
    TValue expected = 0;
    return Value_.compare_exchange_weak(expected, WriterMask, std::memory_order_acquire);
}

inline bool TReaderWriterSpinLock::TryAndTryAcquireWriter() noexcept
{
    auto oldValue = Value_.load(std::memory_order_relaxed);
    if (oldValue != 0) {
        return false;
    }
    return TryAcquireWriter();
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

} // namespace NYT::NConcurrency

