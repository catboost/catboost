#ifndef ATOMIC_INTRUSIVE_PTR_INL_H_
#error "Direct inclusion of this file is not allowed, include atomic_intrusive_ptr.h"
// For the sake of sane code completion.
#include "atomic_intrusive_ptr.h"
#endif
#undef ATOMIC_INTRUSIVE_PTR_INL_H_

#include <util/system/spinlock.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
TAtomicIntrusivePtr<T>::TAtomicIntrusivePtr(std::nullptr_t)
{ }

template <class T>
TAtomicIntrusivePtr<T>::TAtomicIntrusivePtr(TIntrusivePtr<T> other)
    : Ptr_(AcquireObject(other.Release(), true))
{ }

template <class T>
TAtomicIntrusivePtr<T>::TAtomicIntrusivePtr(TAtomicIntrusivePtr&& other)
    : Ptr_(other.Ptr_.load(std::memory_order::relaxed))
{
    other.Ptr_.store(nullptr, std::memory_order::relaxed);
}

template <class T>
TAtomicIntrusivePtr<T>::~TAtomicIntrusivePtr()
{
    ReleaseObject(Ptr_.load());
}

template <class T>
TAtomicIntrusivePtr<T>& TAtomicIntrusivePtr<T>::operator=(TIntrusivePtr<T> other)
{
    Store(std::move(other));
    return *this;
}

template <class T>
TAtomicIntrusivePtr<T>& TAtomicIntrusivePtr<T>::operator=(std::nullptr_t)
{
    Reset();
    return *this;
}

template <class T>
TIntrusivePtr<T> TAtomicIntrusivePtr<T>::Acquire() const
{
    auto ptr = Ptr_.load();
    while (true) {
        auto [obj, localRefs] = TTaggedPtr<T>::Unpack(ptr);

        if (!obj) {
            return {};
        }

        YT_VERIFY(localRefs < ReservedRefCount);

        auto newLocalRefs = localRefs + 1;

        if (newLocalRefs == ReservedRefCount) {
            SpinLockPause();

            ptr = Ptr_.load();
            continue;
        }

        // Can not Ref(obj) here because it can be destroyed.
        if (Ptr_.compare_exchange_weak(ptr, TTaggedPtr(obj, newLocalRefs).Pack())) {
            if (Y_UNLIKELY(newLocalRefs > ReservedRefCount / 2)) {
                Ref(obj, ReservedRefCount / 2);

                // Decrease local ref count.
                while (true) {
                    auto [currentObj, localRefs] = TTaggedPtr<T>::Unpack(ptr);

                    if (currentObj != obj || localRefs <= ReservedRefCount / 2) {
                        Unref(obj, ReservedRefCount / 2);
                        break;
                    }

                    if (Ptr_.compare_exchange_weak(ptr, TTaggedPtr(obj, localRefs - ReservedRefCount / 2).Pack())) {
                        break;
                    }
                }
            }

            return TIntrusivePtr<T>(obj, false);
        }
    }
}

template <class T>
TIntrusivePtr<T> TAtomicIntrusivePtr<T>::Exchange(TIntrusivePtr<T> other)
{
    auto [obj, localRefs] = TTaggedPtr<T>::Unpack(Ptr_.exchange(AcquireObject(other.Release(), true)));
    DoRelease(obj, localRefs + 1);
    return TIntrusivePtr<T>(obj, false);
}

template <class T>
void TAtomicIntrusivePtr<T>::Store(TIntrusivePtr<T> other)
{
    ReleaseObject(Ptr_.exchange(AcquireObject(other.Release(), true)));
}

template <class T>
void TAtomicIntrusivePtr<T>::Reset()
{
    ReleaseObject(Ptr_.exchange(0));
}

template <class T>
bool TAtomicIntrusivePtr<T>::CompareAndSwap(TRawPtr& comparePtr, T* target)
{
    auto* targetPtr = AcquireObject(target, false);

    auto currentPtr = Ptr_.load();
    if (UnpackPointer<T>(currentPtr).Ptr == comparePtr && Ptr_.compare_exchange_strong(currentPtr, targetPtr)) {
        ReleaseObject(currentPtr);
        return true;
    }

    comparePtr = UnpackPointer<T>(currentPtr).Ptr;

    ReleaseObject(targetPtr);
    return false;
}

template <class T>
bool TAtomicIntrusivePtr<T>::CompareAndSwap(TRawPtr& comparePtr, TIntrusivePtr<T> target)
{
    // TODO(lukyan): Make helper for packed owning ptr?
    auto targetPtr = AcquireObject(target.Release(), true);

    auto currentPtr = Ptr_.load();
    if (TTaggedPtr<T>::Unpack(currentPtr).Ptr == comparePtr && Ptr_.compare_exchange_strong(currentPtr, targetPtr)) {
        ReleaseObject(currentPtr);
        return true;
    }

    comparePtr = TTaggedPtr<T>::Unpack(currentPtr).Ptr;

    ReleaseObject(targetPtr);
    return false;
}

template <class T>
typename TAtomicIntrusivePtr<T>::TRawPtr TAtomicIntrusivePtr<T>::Get() const
{
    return TTaggedPtr<void>::Unpack(Ptr_.load()).Ptr;
}

template <class T>
TAtomicIntrusivePtr<T>::operator bool() const
{
    return Get();
}

template <class T>
TPackedPtr TAtomicIntrusivePtr<T>::AcquireObject(T* obj, bool consumeRef)
{
    if (obj) {
        Ref(obj, static_cast<int>(ReservedRefCount - consumeRef));
    }

    return TTaggedPtr(obj).Pack();
}

template <class T>
void TAtomicIntrusivePtr<T>::ReleaseObject(TPackedPtr packedPtr)
{
    auto [obj, localRefs] = TTaggedPtr<T>::Unpack(packedPtr);
    DoRelease(obj, localRefs);
}

template <class T>
void TAtomicIntrusivePtr<T>::DoRelease(T* obj, int refs)
{
    if (obj) {
        Unref(obj, static_cast<int>(ReservedRefCount - refs));
    }
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
bool operator==(const TAtomicIntrusivePtr<T>& lhs, const TIntrusivePtr<T>& rhs)
{
    return lhs.Get() == rhs.Get();
}

template <class T>
bool operator==(const TIntrusivePtr<T>& lhs, const TAtomicIntrusivePtr<T>& rhs)
{
    return lhs.Get() == rhs.Get();
}

template <class T>
bool operator!=(const TAtomicIntrusivePtr<T>& lhs, const TIntrusivePtr<T>& rhs)
{
    return lhs.Get() != rhs.Get();
}

template <class T>
bool operator!=(const TIntrusivePtr<T>& lhs, const TAtomicIntrusivePtr<T>& rhs)
{
    return lhs.Get() != rhs.Get();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
