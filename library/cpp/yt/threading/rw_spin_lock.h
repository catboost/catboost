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
 *  Reader-side acquires are pretty cheap, and readers don't spin unless writers
 *  are present.
 *
 *  The lock is unfair, but writers are prioritized over readers, that is,
 *  if AcquireWriter() is called at some time, then some writer
 *  (not necessarily the same one that called AcquireWriter) will succeed
 *  in the next time. This is implemented by an additional flag "WriterReady",
 *  that writers set on arrival. No readers can proceed until this flag is reset.
 *
 *  WARNING: You probably should not use this lock if forks are possible: see
 *  fork_aware_rw_spin_lock.h for a proper fork-safe lock which does the housekeeping for you.
 *
 *  WARNING: This lock is not recursive: you can't call AcquireReader() twice in the same
 *  thread, as that may lead to a deadlock. For the same reason you shouldn't do WaitFor or any
 *  other context switch under lock.
 *
 *  See tla+/spinlock.tla for the formally verified lock's properties.
 */
class TReaderWriterSpinLock
    : public TSpinLockBase
{
public:
    static constexpr bool Traced = true;

    using TSpinLockBase::TSpinLockBase;

    //! Acquires the reader lock.
    /*!
     *  Optimized for the case of read-intensive workloads.
     *  Cheap (just one atomic increment and no spinning if no writers are present).
     *
     *  WARNING: Don't use this call if forks are possible: forking at some
     *  intermediate point inside #AcquireReader may corrupt the lock state and
     *  leave the lock stuck forever for the child process.
     *
     *  WARNING: The lock is not recursive/reentrant, i.e. it assumes that no thread calls
     *  AcquireReader() if the reader is already acquired for it.
     */
    void AcquireReader() noexcept;
    //! Acquires the reader lock.
    /*!
     *  A more expensive version of #AcquireReader (includes at least
     *  one atomic load and CAS; also may spin even if just readers are present).
     *
     *  In contrast to #AcquireReader, this method can be used in the presence of forks.
     *
     *  WARNING: fork-friendliness alone does not provide fork-safety: additional
     *  actions must be performed to release the lock after a fork. This means you
     *  probably should NOT use this lock in the presence of forks, consider
     *  fork_aware_rw_spin_lock.h instead as a proper fork-safe lock.
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
    static constexpr TValue WriterReadyMask = 2;
    static constexpr TValue ReaderDelta = 4;

    std::atomic<TValue> Value_ = UnlockedValue;

    bool TryAcquireWriterWithExpectedValue(TValue expected) noexcept;

    bool TryAndTryAcquireReader() noexcept;
    bool TryAndTryAcquireWriter() noexcept;

    void AcquireReaderSlow() noexcept;
    void AcquireReaderForkFriendlySlow() noexcept;
    void AcquireWriterSlow() noexcept;
};

////////////////////////////////////////////////////////////////////////////////

//! A variant of TReaderWriterSpinLock occupying the whole cache line.
class alignas(CacheLineSize) TPaddedReaderWriterSpinLock
    : public TReaderWriterSpinLock
{ };

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

