#pragma once

#include "tag_invoke.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// CRTP-base which defines a baseline CPO behavior e.g.
// "If there is TagInvoke overload with the correct tag, use it".
template <class TThis>
struct TTagInvokeCpoBase
{
    template <class... TArgs>
        requires CTagInvocable<const TThis&, TArgs...>
    constexpr decltype(auto) operator()(TArgs&&... args) const
        noexcept(CNothrowTagInvocable<const TThis&, TArgs...>)
    {
        return NYT::TagInvoke(static_cast<const TThis&>(*this), std::forward<TArgs>(args)...);
    }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
