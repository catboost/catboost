#pragma once

#include "concepts.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NTagInvokeDetail {

// This name shadows possible overloads of TagInvoke from parent namespaces
// which could be found by normal unqualified name lookup.
void TagInvoke() = delete;

////////////////////////////////////////////////////////////////////////////////

// NB(arkady-e1ppa): We do not use trailing return type there in order to
// allow incomplete tag types to be safely used here.
struct TFn
{
    template <class TTag, class... TArgs>
        requires requires (TTag&& tag, TArgs&&... args) {
            // "Adl finds TagInvoke overload".
            TagInvoke(std::forward<TTag>(tag), std::forward<TArgs>(args)...);
        }
    constexpr decltype(auto) operator()(TTag&& tag, TArgs&&... args) const
        noexcept(noexcept(TagInvoke(std::forward<TTag>(tag), std::forward<TArgs>(args)...)))
    {
        return TagInvoke(std::forward<TTag>(tag), std::forward<TArgs>(args)...);
    }
};

} // namespace NTagInvokeDetail

////////////////////////////////////////////////////////////////////////////////

// Inline namespace is required so that there is no conflict with
// customizations of TagInvoke defined as friend function definition
// inside a class from namespace NYT.
inline namespace NTagInvokeCPO {

inline constexpr NTagInvokeDetail::TFn TagInvoke = {};

} // inline namespace NTagInvokeCPO

////////////////////////////////////////////////////////////////////////////////

// Some helpful concepts and aliases.
template <class TTag, class... TArgs>
concept CTagInvocable = requires (TTag&& tag, TArgs&&... args) {
    NYT::TagInvoke(std::forward<TTag>(tag), std::forward<TArgs>(args)...);
};

////////////////////////////////////////////////////////////////////////////////

template <class TTag, class... TArgs>
concept CNothrowTagInvocable =
    CTagInvocable<TTag, TArgs...> &&
    requires (TTag&& tag, TArgs&&... args) {
        { NYT::TagInvoke(std::forward<TTag>(tag), std::forward<TArgs>(args)...) } noexcept;
    };

////////////////////////////////////////////////////////////////////////////////

template <class TTag, class... TArgs>
using TTagInvokeResult = std::invoke_result_t<decltype(NYT::TagInvoke), TTag, TArgs...>;

////////////////////////////////////////////////////////////////////////////////

template <auto V>
using TTagInvokeTag = std::decay_t<decltype(V)>;

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class TTag, class TSignature>
struct TTagInvokeTraitsHelper;

template <class TTag, class TReturn, bool NoExcept, class... TArgs>
struct TTagInvokeTraitsHelper<TTag, TReturn(TArgs...) noexcept(NoExcept)>
{
    static constexpr bool IsInvocable = NYT::CInvocable<decltype(NYT::TagInvoke), TReturn(TTag, TArgs...) noexcept(NoExcept)>;
};

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

template <class TTag, class TSignature>
concept CTagInvocableS = NDetail::TTagInvokeTraitsHelper<TTag, TSignature>::IsInvocable;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
