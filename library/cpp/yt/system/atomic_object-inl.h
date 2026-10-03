#ifndef ATOMIC_OBJECT_INL_H_
#error "Direct inclusion of this file is not allowed, include atomic_object.h"
// For the sake of sane code completion.
#include "atomic_object.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
template <class U>
TAtomicObject<T>::TAtomicObject(U&& u)
    : Object_(std::forward<U>(u))
{ }

template <class T>
template <class U>
void TAtomicObject<T>::Store(U&& u)
{
    // NB: Using exchange to avoid destructing the old object while holding the lock.
    std::ignore = Exchange(std::forward<U>(u));
}

template <class T>
template <class U>
T TAtomicObject<T>::Exchange(U&& u)
{
    T tmpObject = std::forward<U>(u);
    {
        auto guard = WriterGuard(Spinlock_);
        std::swap(Object_, tmpObject);
    }
    return tmpObject;
}

template <class T>
bool TAtomicObject<T>::CompareExchange(T& expected, const T& desired)
{
    auto guard = WriterGuard(Spinlock_);
    if (Object_ == expected) {
        auto oldObject = std::move(Object_);
        Y_UNUSED(oldObject);
        Object_ = desired;
        guard.Release();
        return true;
    } else {
        auto oldExpected = std::move(expected);
        Y_UNUSED(oldExpected);
        expected = Object_;
        guard.Release();
        return false;
    }
}

template <class T>
template <std::invocable<T&> F>
std::invoke_result_t<F, T&> TAtomicObject<T>::Transform(F&& func)
{
    auto guard = WriterGuard(Spinlock_);
    return std::invoke(std::forward<F>(func), Object_);
}

template <class T>
template <std::invocable<const T&> F>
std::invoke_result_t<F, const T&> TAtomicObject<T>::Read(F&& func) const
{
    auto guard = ReaderGuard(Spinlock_);
    return std::invoke(std::forward<F>(func), Object_);
}

template <class T>
T TAtomicObject<T>::Load() const
{
    auto guard = ReaderGuard(Spinlock_);
    return Object_;
}

////////////////////////////////////////////////////////////////////////////////

template <class TOriginal, class TSerialized>
void ToProto(TSerialized* serialized, const TAtomicObject<TOriginal>& original)
{
    ToProto(serialized, original.Load());
}

template <class TOriginal, class TSerialized>
void FromProto(TAtomicObject<TOriginal>* original, const TSerialized& serialized)
{
    TOriginal data;
    FromProto(&data, serialized);
    original->Store(std::move(data));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
