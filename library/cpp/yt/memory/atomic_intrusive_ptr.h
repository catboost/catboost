#pragma once

#include "intrusive_ptr.h"

#include <util/system/compiler.h>

#if defined(_lsan_enabled_) || defined(_asan_enabled_)
#include <util/system/spinlock.h>
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Atomic pointer with split reference counting.
/*
 *  \see https://github.com/facebook/folly/blob/main/folly/concurrency/AtomicSharedPtr.h
*/
template <class T>
class TAtomicIntrusivePtr
{
public:
    using TUnderlying = T;
    using element_type = T;

    TAtomicIntrusivePtr() = default;
    TAtomicIntrusivePtr(std::nullptr_t);

    explicit TAtomicIntrusivePtr(TIntrusivePtr<T> other);
    TAtomicIntrusivePtr(TAtomicIntrusivePtr&& other);

    ~TAtomicIntrusivePtr();

    TAtomicIntrusivePtr& operator=(TIntrusivePtr<T> other);
    TAtomicIntrusivePtr& operator=(std::nullptr_t);

    TIntrusivePtr<T> Acquire() const;

    TIntrusivePtr<T> Exchange(TIntrusivePtr<T> other);
    void Store(TIntrusivePtr<T> other);

    void Reset();

    using TRawPtr = std::conditional_t<std::is_const_v<T>, const void*, void*>;
    bool CompareAndSwap(TRawPtr& comparePtr, T* target);
    bool CompareAndSwap(TRawPtr& comparePtr, TIntrusivePtr<T> target);

    //! Result is only suitable for comparison, not dereference.
    TRawPtr Get() const;

    //! Result is only suitable for comparison, not dereference.
    TRawPtr get() const;

    explicit operator bool() const;

private:
    template <class U>
    friend bool operator==(const TAtomicIntrusivePtr<U>& lhs, const TIntrusivePtr<U>& rhs);

    template <class U>
    friend bool operator==(const TIntrusivePtr<U>& lhs, const TAtomicIntrusivePtr<U>& rhs);

    template <class U>
    friend bool operator!=(const TAtomicIntrusivePtr<U>& lhs, const TIntrusivePtr<U>& rhs);

    template <class U>
    friend bool operator!=(const TIntrusivePtr<U>& lhs, const TAtomicIntrusivePtr<U>& rhs);

#if defined(_lsan_enabled_) || defined(_asan_enabled_)
    ::TSpinLock Lock_;
    TIntrusivePtr<T> Ptr_;
#else
    // Keeps packed pointer (localRefCount, objectPtr).
    // Atomic ptr holds N references, where N = ReservedRefCount - localRefCount.
    // LocalRefCount is incremented in Acquire method.
    // When localRefCount exceeds ReservedRefCount / 2 a new portion of refs are required globally.
    // This field is marked mutable in order to make Acquire const-qualified in accordance to its semantics.
    mutable std::atomic<TPackedPtr> Ptr_ = 0;

    constexpr static int CounterBits = PackedPtrTagBits;
    constexpr static int ReservedRefCount = (1 << CounterBits) - 1;

    // Consume ref if ownership is transferred.
    static TPackedPtr AcquireObject(T* obj, bool consumeRef = false);
    static void ReleaseObject(TPackedPtr packedPtr);
    static void DoRelease(T* obj, int refs);
#endif
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ATOMIC_INTRUSIVE_PTR_INL_H_
#include "atomic_intrusive_ptr-inl.h"
#undef ATOMIC_INTRUSIVE_PTR_INL_H_
