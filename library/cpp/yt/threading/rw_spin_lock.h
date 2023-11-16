#pragma once

#include "public.h"
#include "spin_lock_base.h"
#include "spin_lock_count.h"

#include <library/cpp/yt/memory/public.h>

#include <util/system/rwlock.h>

#include <atomic>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

//! Single-writer multiple-readers spin lock.
/*!
 *  Reader-side calls are pretty cheap.
 *  The lock is unfair.
 */
class TReaderWriterSpinLock
    : public TSpinLockBase
{
public:
    using TSpinLockBase::TSpinLockBase;

    //! Acquires the reader lock.
    /*!
     *  Optimized for the case of read-intensive workloads.
     *  Cheap (just one atomic increment and no spinning if no writers are present).
     *  Don't use this call if forks are possible: forking at some
     *  intermediate point inside #AcquireReader may corrupt the lock state and
     *  leave lock forever stuck for the child process.
     */
    void AcquireReader() noexcept;
    //! Acquires the reader lock.
    /*!
     *  A more expensive version of #AcquireReader (includes at least
     *  one atomic load and CAS; also may spin even if just readers are present).
     *  In contrast to #AcquireReader, this method can be used in the presence of forks.
     *  Note that fork-friendliness alone does not provide fork-safety: additional
     *  actions must be performed to release the lock after a fork.
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
    using TValue = ui32;
    static constexpr TValue UnlockedValue = 0;
    static constexpr TValue WriterMask = 1;
    static constexpr TValue ReaderDelta = 2;

    std::atomic<TValue> Value_ = UnlockedValue;


    bool TryAndTryAcquireReader() noexcept;
    bool TryAndTryAcquireWriter() noexcept;

    void AcquireReaderSlow() noexcept;
    void AcquireReaderForkFriendlySlow() noexcept;
    void AcquireWriterSlow() noexcept;
};

REGISTER_TRACKED_SPIN_LOCK_CLASS(TReaderWriterSpinLock)

////////////////////////////////////////////////////////////////////////////////

//! A variant of TReaderWriterSpinLock occupying the whole cache line.
class alignas(CacheLineSize) TPaddedReaderWriterSpinLock
    : public TReaderWriterSpinLock
{ };

REGISTER_TRACKED_SPIN_LOCK_CLASS(TPaddedReaderWriterSpinLock)

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

} // namespace NYT::NThreading

#define RW_SPIN_LOCK_INL_H_
#include "rw_spin_lock-inl.h"
#undef RW_SPIN_LOCK_INL_H_

