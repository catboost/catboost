#pragma once

#include <concepts>
#include <vector>

namespace NYT {

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

template <class T>
struct TIsEmpty
    : public T
{
    int Dummy;

    static constexpr bool Value = (sizeof(TIsEmpty) == sizeof(int));
};

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

template <class V>
concept CStdVector = requires (V& vec) {
    [] <class... T> (std::vector<T...>&) { } (vec);
};

////////////////////////////////////////////////////////////////////////////////

template <class M>
concept CAnyMap = requires {
    typename M::mapped_type;
    typename M::key_type;
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
concept CConstRawPtr = CRawPtr<T> && CConst<decltype(*std::declval<T>())>;

template <class T>
concept CMutableRawPtr = CRawPtr<T> && !CConstRawPtr<T>;

////////////////////////////////////////////////////////////////////////////////

template <class T>
constexpr bool IsEmptyClass()
{
    return NDetail::TIsEmpty<T>::Value;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
