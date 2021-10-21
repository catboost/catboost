#pragma once

#include <util/system/rwlock.h>

#include <atomic>

namespace NYT::NConcurrency {

////////////////////////////////////////////////////////////////////////////////

//! Single-writer multiple-readers spin lock.
/*!
 *  Reader-side calls are pretty cheap.
 *  The lock is unfair.
 */
class TReaderWriterSpinLock
{
public:
    //! Acquires the reader lock.
    /*!
     *  Optimized for the case of read-intensive workloads.
     *  Cheap (just one atomic increment and no spinning if no writers are present).
     *  Don't use this call if forks are possible: forking at some
     *  intermediate point inside #AcquireReader may leave the lock
     *  forever stuck for the child process.
     */
    void AcquireReader() noexcept;
    //! Acquires the reader lock.
    /*!
     *  A more expensive version of #AcquireReader (includes at least
     *  one atomic load and CAS; also may spin even if just readers are present).
     *  In contrast to #AcquireReader, this call is safe to use in presence of forks.
     */
    void AcquireReaderForkFriendly() noexcept;
    //! Tries acquiring the reader lock; see #AcquireReader.
    //! Returns |true| on success.
    bool TryAcquireReader() noexcept;
    //! Tries acquiring the reader lock (and does this in a fork-friendly manner); see #AcquireReaderForkFriendly.
    //! returns |true| on success.
    bool TryAcquireReaderForkFriendly() noexcept;
    //! Releases the reader lock.
    /*!
     *  Cheap (just one atomic decrement).
     */
    void ReleaseReader() noexcept;

    //! Acquires the writer lock.
    /*!
     *  Rather cheap (just one CAS).
     */
    void AcquireWriter() noexcept;
    //! Tries acquiring the writer lock; see #AcquireWriter.
    //! Returns |true| on success.
    bool TryAcquireWriter() noexcept;
    //! Releases the writer lock.
    /*!
     *  Cheap (just one atomic store).
     */
    void ReleaseWriter() noexcept;

    //! Returns true if the lock is taken (either by a reader or writer).
    /*!
     *  This is inherently racy.
     *  Only use for debugging and diagnostic purposes.
     */
    bool IsLocked() const noexcept;

    //! Returns true if the lock is taken by reader.
    /*!
     *  This is inherently racy.
     *  Only use for debugging and diagnostic purposes.
     */
    bool IsLockedByReader() const noexcept;

    //! Returns true if the lock is taken by writer.
    /*!
     *  This is inherently racy.
     *  Only use for debugging and diagnostic purposes.
     */
    bool IsLockedByWriter() const noexcept;

private:
    using TValue = ui64;
    std::atomic<TValue> Value_ = 0;

    static constexpr TValue WriterMask = 1;
    static constexpr TValue ReaderDelta = 2;


    bool TryAndTryAcquireReader() noexcept;
    bool TryAndTryAcquireWriter() noexcept;
};

////////////////////////////////////////////////////////////////////////////////

//! A variant of TReaderWriterSpinLock occupyig the whole cache line.
class TPaddedReaderWriterSpinLock
    : public TReaderWriterSpinLock
{
private:
    [[maybe_unused]]
    char Padding_[64 - sizeof(TReaderWriterSpinLock)];
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TReaderSpinlockTraits
{
    static void Acquire(T* spinlock);
    static void Release(T* spinlock);
};

template <class T>
struct TForkFriendlyReaderSpinlockTraits
{
    static void Acquire(T* spinlock);
    static void Release(T* spinlock);
};


template <class T>
struct TWriterSpinlockTraits
{
    static void Acquire(T* spinlock);
    static void Release(T* spinlock);
};

template <class T>
using TReaderGuard = TGuard<T, TReaderSpinlockTraits<T>>;
template <class T>
using TWriterGuard = TGuard<T, TWriterSpinlockTraits<T>>;

template <class T>
auto ReaderGuard(const T& lock);
template <class T>
auto ReaderGuard(const T* lock);
template <class T>
auto ForkFriendlyReaderGuard(const T& lock);
template <class T>
auto ForkFriendlyReaderGuard(const T* lock);
template <class T>
auto WriterGuard(const T& lock);
template <class T>
auto WriterGuard(const T* lock);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NConcurrency

#define RW_SPINLOCK_INL_H_
#include "rw_spinlock-inl.h"
#undef RW_SPINLOCK_INL_H_

