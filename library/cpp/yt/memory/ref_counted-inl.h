#ifndef REF_COUNTED_INL_H_
#error "Direct inclusion of this file is not allowed, include ref_counted.h"
// For the sake of sane code completion.
#include "ref_counted.h"
#endif

#include "tagged_ptr.h"

#include <util/system/sanitizers.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// TODO(babenko): move to hazard pointers
void RetireHazardPointer(TPackedPtr packedPtr, void (*reclaimer)(TPackedPtr));

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

////////////////////////////////////////////////////////////////////////////////

template <class T, class = void>
struct TFreeMemory
{
    static void Do(void* ptr)
    {
#ifdef _win_
        ::_aligned_free(ptr);
#else
        ::free(ptr);
#endif
    }
};

template <class T>
struct TFreeMemory<T, std::void_t<typename T::TAllocator>>
{
    static void Do(void* ptr)
    {
        using TAllocator = typename T::TAllocator;
        TAllocator::Free(ptr);
    }
};

////////////////////////////////////////////////////////////////////////////////

template <class T, class = void>
struct TMemoryReleaser
{
    static void Do(void* ptr, ui16 /*offset*/)
    {
        TFreeMemory<T>::Do(ptr);
    }
};

template <class T>
struct TMemoryReleaser<T, std::enable_if_t<T::EnableHazard>>
{
    static void Do(void* ptr, ui16 offset)
    {
        // Base pointer is used in HazardPtr as the identity of object.
        auto packedPtr = TTaggedPtr<char>{static_cast<char*>(ptr) + offset, offset}.Pack();
        RetireHazardPointer(packedPtr, [] (TPackedPtr packedPtr) {
            // Base ptr and the beginning of allocated memory region may differ.
            auto [ptr, offset] = TTaggedPtr<char>::Unpack(packedPtr);
            TFreeMemory<T>::Do(ptr - offset);
        });
    }
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
Y_FORCE_INLINE void DestroyRefCountedImpl(T* obj)
{
    // No standard way to statically calculate the base offset even if T is final.
    // static_cast<TFinalDerived*>(virtualBasePtr) does not work.
    auto* basePtr = static_cast<TRefCountedBase*>(obj);
    auto offset = reinterpret_cast<uintptr_t>(basePtr) - reinterpret_cast<uintptr_t>(obj);
    auto* refCounter = GetRefCounter(obj);

    // No virtual call when T is final.
    obj->~T();

    // Fast path. Weak refs cannot appear if there are neither strong nor weak refs.
    if (refCounter->GetWeakRefCount() == 1) {
        NYT::NDetail::TMemoryReleaser<T>::Do(obj, offset);
        return;
    }

    YT_ASSERT(offset < std::numeric_limits<ui16>::max());

    auto* vTablePtr = reinterpret_cast<TPackedPtr*>(basePtr);
    *vTablePtr = TTaggedPtr<void(void*, ui16)>(&NYT::NDetail::TMemoryReleaser<T>::Do, offset).Pack();

    if (refCounter->WeakUnref()) {
        NYT::NDetail::TMemoryReleaser<T>::Do(obj, offset);
    }
}

////////////////////////////////////////////////////////////////////////////////

// Specialization for final classes.
template <class T, bool = std::derived_from<T, TRefCountedBase>>
struct TRefCountedTraits
{
    static_assert(
        std::is_final_v<T>,
        "Ref-counted objects must be derived from TRefCountedBase or to be final");

    static constexpr size_t RefCounterSpace = (sizeof(TRefCounter) + alignof(T) - 1) & ~(alignof(T) - 1);
    static constexpr size_t RefCounterOffset = RefCounterSpace - sizeof(TRefCounter);

    Y_FORCE_INLINE static const TRefCounter* GetRefCounter(const T* obj)
    {
        return reinterpret_cast<const TRefCounter*>(obj) - 1;
    }

    Y_FORCE_INLINE static void Destroy(const T* obj)
    {
        auto* refCounter = GetRefCounter(obj);

        // No virtual call when T is final.
        obj->~T();

        char* ptr = reinterpret_cast<char*>(const_cast<TRefCounter*>(refCounter));

        // Fast path. Weak refs cannot appear if there are neither strong nor weak refs.
        if (refCounter->GetWeakRefCount() == 1) {
            NYT::NDetail::TMemoryReleaser<T>::Do(ptr - RefCounterOffset, RefCounterSpace);
            return;
        }

        if (refCounter->WeakUnref()) {
            NYT::NDetail::TMemoryReleaser<T>::Do(ptr - RefCounterOffset, RefCounterSpace);
        }
    }

    Y_FORCE_INLINE static void Deallocate(const T* obj)
    {
        char* ptr = reinterpret_cast<char*>(const_cast<TRefCounter*>(GetRefCounter(obj)));
        NYT::NDetail::TMemoryReleaser<T>::Do(ptr - RefCounterOffset, RefCounterSpace);
    }
};

// Specialization for classes derived from TRefCountedBase.
template <class T>
struct TRefCountedTraits<T, true>
{
    Y_FORCE_INLINE static const TRefCounter* GetRefCounter(const T* obj)
    {
        return obj;
    }

    Y_FORCE_INLINE static void Destroy(const TRefCountedBase* obj)
    {
        const_cast<TRefCountedBase*>(obj)->DestroyRefCounted();
    }

    Y_FORCE_INLINE static void Deallocate(const TRefCountedBase* obj)
    {
        auto* ptr = reinterpret_cast<TPackedPtr*>(const_cast<TRefCountedBase*>(obj));
        auto [ptrToDeleter, offset] = TTaggedPtr<void(void*, ui16)>::Unpack(*ptr);

        // The most derived type is erased here. So we cannot call TMemoryReleaser with derived type.
        ptrToDeleter(reinterpret_cast<char*>(ptr) - offset, offset);
    }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE int TRefCounter::GetRefCount() const noexcept
{
    return StrongCount_.load(std::memory_order::acquire);
}

Y_FORCE_INLINE void TRefCounter::Ref(int n) const noexcept
{
    // It is safe to use relaxed here, since new reference is always created from another live reference.
    StrongCount_.fetch_add(n, std::memory_order::relaxed);

    YT_ASSERT(WeakCount_.load(std::memory_order::relaxed) > 0);
}

Y_FORCE_INLINE bool TRefCounter::TryRef() const noexcept
{
    auto value = StrongCount_.load(std::memory_order::relaxed);
    YT_ASSERT(WeakCount_.load(std::memory_order::relaxed) > 0);

    while (value != 0 && !StrongCount_.compare_exchange_weak(value, value + 1));
    return value != 0;
}

Y_FORCE_INLINE bool TRefCounter::Unref(int n) const
{
    // We must properly synchronize last access to object with it destruction.
    // Otherwise compiler might reorder access to object past this decrement.
    //
    // See http://www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html#boost_atomic.usage_examples.example_reference_counters
    //
    auto oldStrongCount = StrongCount_.fetch_sub(n, std::memory_order::release);
    YT_ASSERT(oldStrongCount >= n);
    if (oldStrongCount == n) {
        std::atomic_thread_fence(std::memory_order::acquire);
        NSan::Acquire(&StrongCount_);
        return true;
    } else {
        return false;
    }
}

Y_FORCE_INLINE int TRefCounter::GetWeakRefCount() const noexcept
{
    return WeakCount_.load(std::memory_order::acquire);
}

Y_FORCE_INLINE void TRefCounter::WeakRef() const noexcept
{
    auto oldWeakCount = WeakCount_.fetch_add(1, std::memory_order::relaxed);
    YT_ASSERT(oldWeakCount > 0);
}

Y_FORCE_INLINE bool TRefCounter::WeakUnref() const
{
    auto oldWeakCount = WeakCount_.fetch_sub(1, std::memory_order::release);
    YT_ASSERT(oldWeakCount > 0);
    if (oldWeakCount == 1) {
        std::atomic_thread_fence(std::memory_order::acquire);
        NSan::Acquire(&WeakCount_);
        return true;
    } else {
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
Y_FORCE_INLINE const TRefCounter* GetRefCounter(const T* obj)
{
    return NYT::NDetail::TRefCountedTraits<T>::GetRefCounter(obj);
}

template <class T>
Y_FORCE_INLINE void DestroyRefCounted(const T* obj)
{
    NYT::NDetail::TRefCountedTraits<T>::Destroy(obj);
}

template <class T>
Y_FORCE_INLINE void DeallocateRefCounted(const T* obj)
{
    NYT::NDetail::TRefCountedTraits<T>::Deallocate(obj);
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
Y_FORCE_INLINE void Ref(T* obj, int n)
{
    GetRefCounter(obj)->Ref(n);
}

template <class T>
Y_FORCE_INLINE void Unref(T* obj, int n)
{
    if (GetRefCounter(obj)->Unref(n)) {
        DestroyRefCounted(obj);
    }
}

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE void TRefCounted::Unref() const
{
    ::NYT::Unref(this);
}

Y_FORCE_INLINE void TRefCounted::WeakUnref() const
{
    if (TRefCounter::WeakUnref()) {
        DeallocateRefCounted(this);
    }
}

template <class T>
void TRefCounted::DestroyRefCountedImpl(T* obj)
{
    NYT::NDetail::DestroyRefCountedImpl<T>(obj);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
