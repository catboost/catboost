#ifndef IMMORTAL_INL_H_
#error "Direct inclusion of this file is not allowed, include immortal.h"
// For the sake of sane code completion.
#include "immortal.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
template <class... TArgs>
    requires std::constructible_from<T, TArgs...>
constexpr TImmortal<T>::TImmortal(TArgs&&... args)
    : Value_(std::forward<TArgs>(args)...)
{ }

template <class T>
constexpr TImmortal<T>::~TImmortal()
{ }

template <class T>
constexpr T* TImmortal<T>::Get()
{
    return std::addressof(Value_);
}

template <class T>
constexpr const T* TImmortal<T>::Get() const
{
    return std::addressof(Value_);
}

template <class T>
constexpr T& TImmortal<T>::operator*()
{
    return Value_;
}

template <class T>
constexpr const T& TImmortal<T>::operator*() const
{
    return Value_;
}

template <class T>
constexpr T* TImmortal<T>::operator->()
{
    return std::addressof(Value_);
}

template <class T>
constexpr const T* TImmortal<T>::operator->() const
{
    return std::addressof(Value_);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
