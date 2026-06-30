#pragma once
#ifndef FUNCTION_VIEW_INL_H_
#error "Direct inclusion of this file is not allowed, include function_view.h"
// For the sake of sane code completion.
#include "function_view.h"
#endif

#include <library/cpp/yt/assert/assert.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class TResult, bool NoExcept, class... TArgs>
template <CTypeErasable<TResult(TArgs...) noexcept(NoExcept)> TConcrete>
TFunctionView<TResult(TArgs...) noexcept(NoExcept)>::TFunctionView(TConcrete& concreteRef) noexcept
    : TFunctionView(&concreteRef)
{ }

template <class TResult, bool NoExcept, class... TArgs>
template <CTypeErasable<TResult(TArgs...) noexcept(NoExcept)> TConcrete>
TFunctionView<TResult(TArgs...) noexcept(NoExcept)>::TFunctionView(TConcrete* concretePtr) noexcept
{
    Ptr_ = reinterpret_cast<void*>(concretePtr);
    Invoke_ = &TFunctionView::ConcreteInvoke<TConcrete>;
}

template <class TResult, bool NoExcept, class... TArgs>
TFunctionView<TResult(TArgs...) noexcept(NoExcept)>
TFunctionView<TResult(TArgs...) noexcept(NoExcept)>::Release() noexcept
{
    auto copy = *this;
    Reset();
    return copy;
}

template <class TResult, bool NoExcept, class... TArgs>
TResult TFunctionView<TResult(TArgs...) noexcept(NoExcept)>::operator()(TArgs... args) noexcept(NoExcept)
{
    YT_VERIFY(Ptr_);
    return Invoke_(std::forward<TArgs>(args)..., Ptr_);
}

template <class TResult, bool NoExcept, class... TArgs>
template <class TConcrete>
TResult TFunctionView<TResult(TArgs...) noexcept(NoExcept)>::ConcreteInvoke(TArgs... args, TErasedPtr ptr) noexcept(NoExcept)
{
    return (*reinterpret_cast<TConcrete*>(ptr))(std::forward<TArgs>(args)...);
}

template <class TResult, bool NoExcept, class... TArgs>
TFunctionView<TResult(TArgs...) noexcept(NoExcept)>::operator bool() const noexcept
{
    return IsValid();
}

template <class TResult, bool NoExcept, class... TArgs>
bool TFunctionView<TResult(TArgs...) noexcept(NoExcept)>::IsValid() const noexcept
{
    return Ptr_ != nullptr;
}

template <class TResult, bool NoExcept, class... TArgs>
void TFunctionView<TResult(TArgs...) noexcept(NoExcept)>::Reset() noexcept
{
    Ptr_ = nullptr;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
