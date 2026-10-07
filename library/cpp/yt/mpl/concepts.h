#pragma once

#include <concepts>
#include <type_traits>
#include <utility>
#include <vector>

namespace NYT::NMpl {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class T, class TSignature>
struct TIsInvocable;

template <class T, class TResult, bool NoExcept, class... TArgs>
struct TIsInvocable<T, TResult(TArgs...) noexcept(NoExcept)>
{
private:
    static constexpr bool IsInvocable_ = requires (T&& t, TArgs&&... args) {
        { std::forward<T>(t)(std::forward<TArgs>(args)...) } -> std::same_as<TResult>;
    };

    static constexpr bool IsNoThrowInvocable_ = requires (T&& t, TArgs&&... args) {
        { std::forward<T>(t)(std::forward<TArgs>(args)...) } noexcept -> std::same_as<TResult>;
    };

public:
    static constexpr bool Value =
        IsInvocable_ &&
        (!NoExcept || IsNoThrowInvocable_);
};

template <template <class...> class TTemplate, class... TArgs>
void DerivedFromSpecializationImpl(const TTemplate<TArgs...>&);

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

template <class TObject, class TScalar>
concept CScalable = requires (TObject object, TScalar scalar)
{
    { object * scalar } -> std::same_as<TObject>;
};

////////////////////////////////////////////////////////////////////////////////

template <class T, class TSignature>
concept CInvocable = NDetail::TIsInvocable<T, TSignature>::Value;

////////////////////////////////////////////////////////////////////////////////

template <class TNeedle, class... THayStack>
concept COneOf = (std::same_as<TNeedle, THayStack> || ...);

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class... Ts>
inline constexpr bool DistinctImpl = true;

template <class T, class... Ts>
inline constexpr bool DistinctImpl<T, Ts...> = DistinctImpl<Ts...> && !COneOf<T, Ts...>;

} // namespace NDetail

template <class... Ts>
concept CDistinct = NDetail::DistinctImpl<Ts...>;

////////////////////////////////////////////////////////////////////////////////

template <class TDerived, template <class...> class TTemplatedBase>
concept CDerivedFromSpecializationOf = requires (const TDerived& instance)
{
    NDetail::DerivedFromSpecializationImpl<TTemplatedBase>(instance);
};

////////////////////////////////////////////////////////////////////////////////

template <class V>
concept CStdVector = requires (V& vec) {
    [] <class... T> (std::vector<T...>&) { } (vec);
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
concept CAssociative = requires {
    typename T::key_type;
};

template <class T>
concept CMapping = CAssociative<T> && requires {
    typename T::mapped_type;
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
concept CConst = std::is_const_v<T>;

template <class T>
concept CNonConst = !CConst<T>;

////////////////////////////////////////////////////////////////////////////////

template <class T>
concept CRawPtr = std::is_pointer_v<T>;

template <class T>
concept CConstRawPtr = CRawPtr<T> && CConst<std::remove_reference_t<decltype(*std::declval<T>())>>;

template <class T>
concept CMutableRawPtr = CRawPtr<T> && !CConstRawPtr<T>;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NMpl
