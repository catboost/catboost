#pragma once

#include "public.h"
#include "rw_spin_lock.h"
#include "spin_lock_base.h"
#include "spin_lock_count.h"

#include <library/cpp/yt/memory/public.h>

#include <util/system/rwlock.h>

#include <atomic>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

// TODO(pavook): deprecate it.

//! Single-writer multiple-readers spin lock.
/*!
 *  Reader-side calls are pretty cheap.
 *  WARNING: The lock is unfair, and readers can starve writers. See rw_spin_lock.h for a writer-prioritized lock.
 *  WARNING: Never use the bare lock if forks are possible: see fork_aware_rw_spin_lock.h for a fork-safe lock.
 *  Unlike rw_spin_lock.h, reader-side is reentrant here: it is possible to acquire the **reader** lock multiple times
 *  even in the single thread.
 *  This doesn't mean you should do it: in fact, you shouldn't: use separate locks for separate entities.
 *  If you see this class in your code, try migrating to the proper rw_spin_lock.h after ensuring you don't rely on
 *  reentrant locking.
 */
class TWriterStarvingRWSpinLock
    : public TSpinLockBase
{
public:
    static constexpr bool Traced = true;

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
    //! Tries acquiring the reader lock; see #AcquireReader.
    //! Returns |true| on success.
    bool TryAcquireReader() noexcept;
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
    void AcquireWriterSlow() noexcept;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

#define WRITER_STARVING_RW_SPIN_LOCK_INL_H_
#include "writer_starving_rw_spin_lock-inl.h"
#undef WRITER_STARVING_RW_SPIN_LOCK_INL_H_

