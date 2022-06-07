#pragma once

#include <util/generic/cast.h>
#include <util/generic/ptr.h>
#include <util/generic/utility.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/guard.h>
#include <util/system/spinlock.h>
#include <util/system/yassert.h>

namespace NHotSwapPrivate {
    // Special guard object for THotSwap
    class TWriterLock {
    public:
        // Implements multi-lock wait-free interface for readers
        void Acquire() noexcept;
        void Release() noexcept;

        void WaitAllReaders() const noexcept;

    private:
        TAtomic ReadersCount = 0;
    };

}

/// Object container that can be switched to another such object concurrently.
/// T must support atomic reference counting
///
/// Typical usage is when we have rarely changed, but frequently used data.
/// If we want to use reference counting, we can't concurrently change and read
/// intrusive pointer without extra synchronization.
/// This class provides such synchronization mechanism with minimal read time.
///
///
/// Usage sample
///
/// THotSwap<T> Obj;
///
/// thread 1:
/// ...
/// TIntrusivePtr<T> obj = Obj.AtomicLoad(); // get current object
/// ... use of obj
///
/// thread 2:
/// ...
/// Obj.AtomicStore(new T()); // set new object
///
template <class T, class Ops = TDefaultIntrusivePtrOps<T>>
class THotSwap {
public:
    using TPtr = TIntrusivePtr<T, Ops>;

public:
    THotSwap() noexcept {
    }

    explicit THotSwap(T* p) noexcept {
        AtomicStore(p);
    }

    explicit THotSwap(const TPtr& p) noexcept
        : THotSwap(p.Get())
    {
    }

    THotSwap(const THotSwap& p) noexcept
        : THotSwap(p.AtomicLoad())
    {
    }

    THotSwap(THotSwap&& other) noexcept {
        DoSwap(RawPtr, other.RawPtr); // we don't need thread safety, because both objects are local
    }

    ~THotSwap() noexcept {
        AtomicStore(nullptr);
    }

    THotSwap& operator=(const THotSwap& p) noexcept {
        AtomicStore(p.AtomicLoad());
        return *this;
    }

    /// Wait-free read pointer to object
    ///
    /// @returns          Current value of stored object
    TPtr AtomicLoad() const noexcept {
        const TAtomicBase lockIndex = GetLockIndex();
        auto guard = Guard(WriterLocks[lockIndex]); // non-blocking (for other AtomicLoad()'s) guard
        return GetRawPtr();
    }

    /// Update to new object
    ///
    /// @param[in] p      New value to store
    void AtomicStore(T* p) noexcept;

    /// Update to new object
    ///
    /// @param[in] p      New value to store
    void AtomicStore(const TPtr& p) noexcept {
        AtomicStore(p.Get());
    }

private:
    T* GetRawPtr() const noexcept {
        return reinterpret_cast<T*>(AtomicGet(RawPtr));
    }

    TAtomicBase GetLockIndex() const noexcept {
        return AtomicGet(LockIndex);
    }

    TAtomicBase SwitchLockIndex() noexcept; // returns previous index value
    void SwitchRawPtr(T* from, T* to) noexcept;
    void WaitReaders() noexcept;

private:
    TAtomic RawPtr = 0; // T* // Pointer to current value
    static_assert(sizeof(TAtomic) == sizeof(T*), "TAtomic can't represent a pointer value");

    TAdaptiveLock UpdateMutex;                           // Guarantee that AtomicStore() will be one at a time
    mutable NHotSwapPrivate::TWriterLock WriterLocks[2]; // Guarantee that AtomicStore() will wait for all concurrent AtomicLoad()'s completion
    TAtomic LockIndex = 0;
};

// Atomic operations of AtomicLoad:
// r:1 index = LockIndex
// r:2 WriterLocks[index].ReadersCount++
// r:3 p = RawPtr
// r:4 p->RefCount++
// r:5 WriterLocks[index].ReadersCount--

// Important atomic operations of AtomicStore(newRawPtr):
// w:1 RawPtr = newRawPtr
// w:2 LockIndex = 1
// w:3 WriterLocks[0].Wait()
// w:4 LockIndex = 0
// w:5 WriterLocks[1].Wait()

// w:3 (first wait) is needed for sequences:
// r:1-3, w:1-2, r:4-5, w:3-5 // the most frequent case
// w1:1, r:1, w1:2-5, r:2-3, w2:1-2, r:4-5, w2:3-5

// w:5 (second wait) is needed for sequences:
// w1:1-2, r:1, w1:3-5, r:2-3, w2:1-4, r:4-5, w2:5
// If there was only one wait,
// in this case writer wouldn't wait appropriate reader

// w1, w2 - two different writers

template <class T, class Ops>
void THotSwap<T, Ops>::AtomicStore(T* p) noexcept {
    TPtr oldPtr;
    with_lock (UpdateMutex) {
        oldPtr = GetRawPtr();

        SwitchRawPtr(oldPtr.Get(), p);
        Y_ASSERT(!oldPtr || oldPtr.RefCount() > 0);

        // Wait all AtomicLoad()'s to properly take old pointer value concurrently
        WaitReaders();

        // Release lock and then kill (maybe) old object
    }
}

template <class T, class Ops>
TAtomicBase THotSwap<T, Ops>::SwitchLockIndex() noexcept {
    const TAtomicBase prevIndex = AtomicGet(LockIndex);
    Y_ASSERT(prevIndex == 0 || prevIndex == 1);
    AtomicSet(LockIndex, prevIndex ^ 1);
    return prevIndex;
}

template <class T, class Ops>
void THotSwap<T, Ops>::WaitReaders() noexcept {
    WriterLocks[SwitchLockIndex()].WaitAllReaders();
    WriterLocks[SwitchLockIndex()].WaitAllReaders();
}

template <class T, class Ops>
void THotSwap<T, Ops>::SwitchRawPtr(T* from, T* to) noexcept {
    if (to)
        Ops::Ref(to); // Ref() for new value

    AtomicSet(RawPtr, reinterpret_cast<TAtomicBase>(to));

    if (from)
        Ops::UnRef(from); // Unref() for old value
}
