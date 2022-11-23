#ifndef REF_COUNTED_INL_H_
#error "Direct inclusion of this file is not allowed, include ref_counted.h"
// For the sake of sane code completion.
#include "ref_counted.h"
#endif

#include <util/system/sanitizers.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

constexpr uint16_t PtrBits = 48;
constexpr uintptr_t PtrMask = (1ULL << PtrBits) - 1;

template <class T>
Y_FORCE_INLINE char* PackPointer(T* ptr, uint16_t data)
{
    return reinterpret_cast<char*>((static_cast<uintptr_t>(data) << PtrBits) | reinterpret_cast<uintptr_t>(ptr));
}

template <class T>
struct TPackedPointer
{
    uint16_t Data;
    T* Ptr;
};

template <class T>
Y_FORCE_INLINE TPackedPointer<T> UnpackPointer(void* packedPtr)
{
    auto castedPtr = reinterpret_cast<uintptr_t>(packedPtr);
    return {static_cast<uint16_t>(castedPtr >> PtrBits), reinterpret_cast<T*>(castedPtr & PtrMask)};
}

////////////////////////////////////////////////////////////////////////////////

template <class T, class = void>
struct TMemoryReleaser
{
    static void Do(void* ptr, uint16_t /*offset*/)
    {
        TFreeMemory<T>::Do(ptr);
    }
};

using TDeleter = void (*)(void*);

void ScheduleObjectDeletion(void* ptr, TDeleter deleter);

template <class T>
struct TMemoryReleaser<T, std::enable_if_t<T::EnableHazard>>
{
    static void Do(void* ptr, uint16_t offset)
    {
        // Base pointer is used in HazardPtr as the identity of object.
        auto* basePtr = PackPointer(static_cast<char*>(ptr) + offset, offset);

        ScheduleObjectDeletion(basePtr, [] (void* ptr) {
            // Base ptr and the beginning of allocated memory region may differ.
            auto [offset, basePtr] = UnpackPointer<char>(ptr);
            TFreeMemory<T>::Do(basePtr - offset);
        });
    }
};

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE int TRefCounter::GetRefCount() const noexcept
{
    return StrongCount_.load(std::memory_order_acquire);
}

Y_FORCE_INLINE void TRefCounter::Ref(int n) const noexcept
{
    // It is safe to use relaxed here, since new reference is always created from another live reference.
    StrongCount_.fetch_add(n, std::memory_order_relaxed);

    YT_ASSERT(WeakCount_.load(std::memory_order_relaxed) > 0);
}

Y_FORCE_INLINE bool TRefCounter::TryRef() const noexcept
{
    auto value = StrongCount_.load(std::memory_order_relaxed);
    YT_ASSERT(WeakCount_.load(std::memory_order_relaxed) > 0);

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
    auto oldStrongCount = StrongCount_.fetch_sub(n, std::memory_order_release);
    YT_ASSERT(oldStrongCount >= n);
    if (oldStrongCount == n) {
        std::atomic_thread_fence(std::memory_order_acquire);
        NSan::Acquire(&StrongCount_);
        return true;
    } else {
        return false;
    }
}

Y_FORCE_INLINE int TRefCounter::GetWeakRefCount() const noexcept
{
    return WeakCount_.load(std::memory_order_acquire);
}

Y_FORCE_INLINE void TRefCounter::WeakRef() const noexcept
{
    auto oldWeakCount = WeakCount_.fetch_add(1, std::memory_order_relaxed);
    YT_ASSERT(oldWeakCount > 0);
}

Y_FORCE_INLINE bool TRefCounter::WeakUnref() const
{
    auto oldWeakCount = WeakCount_.fetch_sub(1, std::memory_order_release);
    YT_ASSERT(oldWeakCount > 0);
    if (oldWeakCount == 1) {
        std::atomic_thread_fence(std::memory_order_acquire);
        NSan::Acquire(&WeakCount_);
        return true;
    } else {
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////

template <class T, bool = std::is_base_of_v<TRefCountedBase, T>>
struct TRefCountedHelper
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
            TMemoryReleaser<T>::Do(ptr - RefCounterOffset, RefCounterSpace);
            return;
        }

        if (refCounter->WeakUnref()) {
            TMemoryReleaser<T>::Do(ptr - RefCounterOffset, RefCounterSpace);
        }
    }

    Y_FORCE_INLINE static void Deallocate(const T* obj)
    {
        char* ptr = reinterpret_cast<char*>(const_cast<TRefCounter*>(GetRefCounter(obj)));
        TMemoryReleaser<T>::Do(ptr - RefCounterOffset, RefCounterSpace);
    }
};

template <class T>
struct TRefCountedHelper<T, true>
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
        auto* ptr = reinterpret_cast<void**>(const_cast<TRefCountedBase*>(obj));
        auto [offset, ptrToDeleter] = UnpackPointer<void(void*, uint16_t)>(*ptr);

        // The most derived type is erased here. So we cannot call TMemoryReleaser with derived type.
        ptrToDeleter(reinterpret_cast<char*>(ptr) - offset, offset);
    }
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
Y_FORCE_INLINE const TRefCounter* GetRefCounter(const T* obj)
{
    return TRefCountedHelper<T>::GetRefCounter(obj);
}

template <class T>
Y_FORCE_INLINE void DestroyRefCounted(const T* obj)
{
    TRefCountedHelper<T>::Destroy(obj);
}

template <class T>
Y_FORCE_INLINE void DeallocateRefCounted(const T* obj)
{
    TRefCountedHelper<T>::Deallocate(obj);
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
void TRefCounted::DestroyRefCountedImpl(T* ptr)
{
    // No standard way to statically calculate the base offset even if T is final.
    // static_cast<TFinalDerived*>(virtualBasePtr) does not work.

    auto* basePtr = static_cast<TRefCountedBase*>(ptr);
    auto offset = reinterpret_cast<uintptr_t>(basePtr) - reinterpret_cast<uintptr_t>(ptr);
    auto* refCounter = GetRefCounter(ptr);

    // No virtual call when T is final.
    ptr->~T();

    // Fast path. Weak refs cannot appear if there are neither strong nor weak refs.
    if (refCounter->GetWeakRefCount() == 1) {
        TMemoryReleaser<T>::Do(ptr, offset);
        return;
    }

    YT_ASSERT(offset < std::numeric_limits<uint16_t>::max());

    auto* vTablePtr = reinterpret_cast<char**>(basePtr);
    *vTablePtr = PackPointer(&TMemoryReleaser<T>::Do, offset);

    if (refCounter->WeakUnref()) {
        TMemoryReleaser<T>::Do(ptr, offset);
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
