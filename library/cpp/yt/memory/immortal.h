#pragma once

#include <concepts>
#include <memory>
#include <utility>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Holds an instance of T whose destructor is never invoked.
//! Constexpr-constructible whenever T is, hence may be declared constinit.
template <class T>
class TImmortal
{
public:
    template <class... TArgs>
        requires std::constructible_from<T, TArgs...>
    constexpr explicit TImmortal(TArgs&&... args);

    constexpr ~TImmortal();

    TImmortal(const TImmortal&) = delete;
    TImmortal& operator=(const TImmortal&) = delete;

    constexpr T* Get();
    constexpr const T* Get() const;

    constexpr T& operator*();
    constexpr const T& operator*() const;

    constexpr T* operator->();
    constexpr const T* operator->() const;

private:
    // NB: The destructor of a union member is never invoked implicitly.
    union
    {
        T Value_;
    };
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define IMMORTAL_INL_H_
#include "immortal-inl.h"
#undef IMMORTAL_INL_H_
