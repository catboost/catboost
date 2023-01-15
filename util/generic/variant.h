#pragma once

#include "variant_traits.h"

#include <util/generic/hash.h>

template <class T>
struct TVariantTypeTag {}; // aka std::in_place_type_t

template <size_t I>
struct TVariantIndexTag {}; // aka std::in_place_index_t

template <size_t I, class V>
using TVariantAlternative = ::NVariant::TAlternative<I, V>;

template <size_t I, class V>
using TVariantAlternativeType = ::NVariant::TAlternativeType<I, V>;

template <class T, class V>
struct TVariantIndex;

template <class T, class... Ts>
struct TVariantIndex<T, TVariant<Ts...>> : ::NVariant::TIndexOf<T, Ts...> {};

// Since there is now standard metafunction for std::variant,
// we need template specialization for it
#if _LIBCPP_STD_VER >= 17
#include <variant>

template <class T, class... Ts>
struct TVariantIndex<T, std::variant<Ts...>> : ::NVariant::TIndexOf<T, Ts...> {};
#endif

template <class T, class V>
constexpr size_t TVariantIndexV = TVariantIndex<T, V>::value;

template <class V>
using TVariantSize = ::NVariant::TSize<V>;

constexpr size_t TVARIANT_NPOS = ::NVariant::T_NPOS;


template <class F, class... Ts>
decltype(auto) Visit(F&& f, TVariant<Ts...>& v);

template <class F, class... Ts>
decltype(auto) Visit(F&& f, const TVariant<Ts...>& v);

template <class F, class... Ts>
decltype(auto) Visit(F&& f, TVariant<Ts...>&& v);

template <class F, class... Ts>
decltype(auto) Visit(F&& f, const TVariant<Ts...>&& v);


template <class T, class... Ts>
constexpr bool HoldsAlternative(const TVariant<Ts...>& v) noexcept;


template <size_t I, class... Ts>
decltype(auto) Get(TVariant<Ts...>& v);

template <size_t I, class... Ts>
decltype(auto) Get(const TVariant<Ts...>& v);

template <size_t I, class... Ts>
decltype(auto) Get(TVariant<Ts...>&& v);

template <size_t I, class... Ts>
decltype(auto) Get(const TVariant<Ts...>&& v);


template <class T, class... Ts>
decltype(auto) Get(TVariant<Ts...>& v);

template <class T, class... Ts>
decltype(auto) Get(const TVariant<Ts...>& v);

template <class T, class... Ts>
decltype(auto) Get(TVariant<Ts...>&& v);

template <class T, class... Ts>
decltype(auto) Get(const TVariant<Ts...>&& v);


template <size_t I, class... Ts>
auto* GetIf(TVariant<Ts...>* v) noexcept;

template <size_t I, class... Ts>
const auto* GetIf(const TVariant<Ts...>* v) noexcept;

template <class T, class... Ts>
T* GetIf(TVariant<Ts...>* v) noexcept;

template <class T, class... Ts>
const T* GetIf(const TVariant<Ts...>* v) noexcept;

//! |std::variant (c++17)| poor substitute of discriminated union.
template <class... Ts>
class TVariant {
    template <class T>
    using TIndex = ::NVariant::TIndexOf<std::decay_t<T>, Ts...>;

    using T_0 = TVariantAlternativeType<0, TVariant>;

    static_assert(::NVariant::TTypeTraits<Ts...>::TNoRefs::value,
                  "TVariant type arguments cannot be references.");
    static_assert(::NVariant::TTypeTraits<Ts...>::TNoVoids::value,
                  "TVariant type arguments cannot be void.");
    static_assert(::NVariant::TTypeTraits<Ts...>::TNoArrays::value,
                  "TVariant type arguments cannot be arrays.");
    static_assert(::NVariant::TTypeTraits<Ts...>::TNotEmpty::value,
                  "TVariant type list cannot be empty.");

public:
    TVariant() noexcept(std::is_nothrow_default_constructible<T_0>::value) {
        static_assert(std::is_default_constructible<T_0>::value,
                      "First alternative must be default constructible");
        EmplaceImpl<T_0>();
    }

    TVariant(const TVariant& rhs) {
        if (!rhs.valueless_by_exception()) {
            ForwardVariant(rhs);
        }
    }

    TVariant(TVariant&& rhs) noexcept(
        TConjunction<std::is_nothrow_move_constructible<Ts>...>::value) {
        if (!rhs.valueless_by_exception()) {
            ForwardVariant(std::move(rhs));
        }
    }

    template <class T, class = std::enable_if_t<!std::is_same<std::decay_t<T>, TVariant>::value>>
    TVariant(T&& value) {
        EmplaceImpl<T>(std::forward<T>(value));
    }

    //! Constructs an instance in-place.
    template <class T, class... TArgs>
    explicit TVariant(TVariantTypeTag<T>, TArgs&&... args) {
        EmplaceImpl<T>(std::forward<TArgs>(args)...);
    }

    //! Constructs an instance in-place by index.
    template <size_t I, class... TArgs>
    explicit TVariant(TVariantIndexTag<I>, TArgs&&... args) {
        EmplaceImpl<TVariantAlternativeType<I, TVariant>>(std::forward<TArgs>(args)...);
    }

    ~TVariant() {
        Destroy();
    }

    TVariant& operator=(const TVariant& rhs) {
        if (Y_UNLIKELY(rhs.valueless_by_exception())) {
            if (Y_LIKELY(!valueless_by_exception())) {
                DestroyImpl();
                Index_ = ::TVariantSize<TVariant>::value;
            }
        } else if (index() == rhs.index()) {
            ::Visit([&](auto& dst) -> void {
                ::Visit([&](const auto& src) -> void {
                    ::NVariant::CallIfSame<void>([](auto& x, const auto& y) {
                        x = y;
                    }, dst, src);
                }, rhs);
            }, *this);
        } else {
            // Strong exception guarantee.
            *this = TVariant{rhs};
        }
        return *this;
    }

    TVariant& operator=(TVariant&& rhs) {
        if (Y_UNLIKELY(rhs.valueless_by_exception())) {
            if (Y_LIKELY(!valueless_by_exception())) {
                DestroyImpl();
                Index_ = ::TVariantSize<TVariant>::value;
            }
        } else if (index() == rhs.index()) {
            ::Visit([&](auto& dst) -> void {
                ::Visit([&](auto& src) -> void {
                    ::NVariant::CallIfSame<void>([](auto& x, auto& y) {
                        x = std::move(y);
                    }, dst, src);
                }, rhs);
            }, *this);
        } else {
            Destroy();
            try {
                ForwardVariant(std::move(rhs));
            } catch (...) {
                Index_ = ::TVariantSize<TVariant>::value;
                throw;
            }
        }
        return *this;
    }

    template <class T>
    std::enable_if_t<!std::is_same<std::decay_t<T>, TVariant>::value,
                     TVariant&>
    operator=(T&& value) {
        if (::HoldsAlternative<std::decay_t<T>>(*this)) {
            *ReinterpretAs<T>() = std::forward<T>(value);
        } else {
            emplace<T>(std::forward<T>(value));
        }
        return *this;
    }

    void swap(TVariant& rhs) {
        if (!valueless_by_exception() || !rhs.valueless_by_exception()) {
            if (index() == rhs.index()) {
                ::Visit([&](auto& aVal) -> void {
                    ::Visit([&](auto& bVal) -> void {
                        ::NVariant::CallIfSame<void>([](auto& x, auto& y) {
                            DoSwap(x, y);
                        }, aVal, bVal);
                    }, rhs);
                }, *this);
            } else {
                TVariant tmp(rhs);
                rhs.Destroy();
                rhs.ForwardVariant(std::move(*this));
                Destroy();
                ForwardVariant(std::move(tmp));
            }
        }
    }

    void Swap(TVariant& rhs) {
        swap(rhs);
    }

    template <class T, class... TArgs>
    T& emplace(TArgs&&... args) {
        Destroy();
        try {
            return EmplaceImpl<T>(std::forward<TArgs>(args)...);
        } catch (...) {
            Index_ = ::TVariantSize<TVariant>::value;
            throw;
        }
    };

    template <size_t I, class... TArgs, class T = TVariantAlternativeType<I, TVariant>>
    T& emplace(TArgs&&... args) {
        return emplace<T>(std::forward<TArgs>(args)...);
    };

    template <class T>
    std::enable_if_t<!std::is_same<std::decay_t<T>, TVariant>::value,
                     bool>
    operator==(const T& value) const {
        return ::HoldsAlternative<T>(*this) && *ReinterpretAs<T>() == value;
    }

    //! Standart integration
    constexpr size_t index() const noexcept {
        return valueless_by_exception() ? TVARIANT_NPOS : Index_;
    }

    /* A TVariant that is valueless by exception is treated as being in an invalid state:
     * Index returns TVARIANT_NPOS, Get and Visit throw TWrongVariantError.
     */
    constexpr bool valueless_by_exception() const noexcept {
        return Index_ == ::TVariantSize<TVariant>::value;
    }

private:
    void Destroy() noexcept {
        if (!valueless_by_exception()) {
            DestroyImpl();
        }
    }

    void DestroyImpl() noexcept {
        ::Visit([](auto& value) {
            using T = std::decay_t<decltype(value)>;
            value.~T();
        }, *this);
    }

    template <class T, class... TArgs>
    T& EmplaceImpl(TArgs&&... args) {
        static_assert(TIndex<T>::value != TVARIANT_NPOS, "Type not in TVariant.");
        new (&Storage_) std::decay_t<T>(std::forward<TArgs>(args)...);
        Index_ = TIndex<T>::value;
        return *ReinterpretAs<T>();
    }

    template <class Variant>
    void ForwardVariant(Variant&& rhs) {
        ::Visit([&](auto&& value) {
            new (this) TVariant(std::forward<decltype(value)>(value));
        }, std::forward<Variant>(rhs));
    }

private:
    template <class T>
    auto* ReinterpretAs() noexcept {
        return reinterpret_cast<std::decay_t<T>*>(&Storage_);
    }

    template <class T>
    const auto* ReinterpretAs() const noexcept {
        return reinterpret_cast<const std::decay_t<T>*>(&Storage_);
    }

    friend struct NVariant::TVariantAccessor;

private:
    size_t Index_ = ::TVariantSize<TVariant>::value;
    std::aligned_union_t<0, Ts...> Storage_;
};

template <class... Ts>
bool operator==(const TVariant<Ts...>& a, const TVariant<Ts...>& b) {
    if (a.index() != b.index()) {
        return false;
    }
    return a.valueless_by_exception() || ::Visit([&](const auto& aVal) -> bool {
        return ::Visit([&](const auto& bVal) -> bool {
            return ::NVariant::CallIfSame<bool>([](const auto& x, const auto& y) {
                return x == y;
            }, aVal, bVal);
        }, b);
    }, a);
}

template <class... Ts>
bool operator!=(const TVariant<Ts...>& a, const TVariant<Ts...>& b) {
    // The standard forces as to call operator!= for values stored in variants.
    // So we cannot reuse operator==.
    if (a.index() != b.index()) {
        return true;
    }
    return !a.valueless_by_exception() && ::Visit([&](const auto& aVal) -> bool {
        return ::Visit([&](const auto& bVal) -> bool {
            return ::NVariant::CallIfSame<bool>([](const auto& x, const auto& y) {
                return x != y;
            }, aVal, bVal);
        }, b);
    }, a);
}

template <class... Ts>
bool operator<(const TVariant<Ts...>& a, const TVariant<Ts...>& b) {
    if (b.valueless_by_exception()) {
        return false;
    }
    if (a.valueless_by_exception()) {
        return true;
    }
    if (a.index() == b.index()) {
        return ::Visit([&](const auto& aVal) -> bool {
            return ::Visit([&](const auto& bVal) -> bool {
                return ::NVariant::CallIfSame<bool>([](const auto& x, const auto& y) {
                    return x < y;
                }, aVal, bVal);
            }, b);
        }, a);
    }
    return a.index() < b.index();
}

template <class... Ts>
bool operator>(const TVariant<Ts...>& a, const TVariant<Ts...>& b) {
    // The standard forces as to call operator> for values stored in variants.
    // So we cannot reuse operator< and operator==.
    if (a.valueless_by_exception()) {
        return false;
    }
    if (b.valueless_by_exception()) {
        return true;
    }
    if (a.index() == b.index()) {
        return ::Visit([&](const auto& aVal) -> bool {
            return ::Visit([&](const auto& bVal) -> bool {
                return ::NVariant::CallIfSame<bool>([](const auto& x, const auto& y) {
                    return x > y;
                }, aVal, bVal);
            }, b);
        }, a);
    }
    return a.index() > b.index();
}

template <class... Ts>
bool operator<=(const TVariant<Ts...>& a, const TVariant<Ts...>& b) {
    // The standard forces as to call operator> for values stored in variants.
    // So we cannot reuse operator< and operator==.
    if (a.valueless_by_exception()) {
        return true;
    }
    if (b.valueless_by_exception()) {
        return false;
    }
    if (a.index() == b.index()) {
        return ::Visit([&](const auto& aVal) -> bool {
            return ::Visit([&](const auto& bVal) -> bool {
                return ::NVariant::CallIfSame<bool>([](const auto& x, const auto& y) {
                    return x <= y;
                }, aVal, bVal);
            }, b);
        }, a);
    }
    return a.index() < b.index();
}

template <class... Ts>
bool operator>=(const TVariant<Ts...>& a, const TVariant<Ts...>& b) {
    // The standard forces as to call operator> for values stored in variants.
    // So we cannot reuse operator< and operator==.
    if (b.valueless_by_exception()) {
        return true;
    }
    if (a.valueless_by_exception()) {
        return false;
    }
    if (a.index() == b.index()) {
        return ::Visit([&](const auto& aVal) -> bool {
            return ::Visit([&](const auto& bVal) -> bool {
                return ::NVariant::CallIfSame<bool>([](const auto& x, const auto& y) {
                    return x >= y;
                }, aVal, bVal);
            }, b);
        }, a);
    }
    return a.index() > b.index();
}


namespace NVariant {
    template <class F, class V>
    decltype(auto) VisitImpl(F&& f, V&& v) {
        using FRef = decltype(std::forward<F>(f));
        using VRef = decltype(std::forward<V>(v));
        static_assert(::NVariant::CheckReturnTypes<FRef, VRef>(
                          std::make_index_sequence<::TVariantSize<std::decay_t<V>>::value>{}),
                      "");
        using ReturnType = ::NVariant::TReturnType<FRef, VRef>;
        return ::NVariant::VisitWrapForVoid(
            std::forward<F>(f), std::forward<V>(v), std::is_void<ReturnType>{});
    }
}

template <class F, class... Ts>
decltype(auto) Visit(F&& f, TVariant<Ts...>& v) {
    return ::NVariant::VisitImpl(std::forward<F>(f), v);
}

template <class F, class... Ts>
decltype(auto) Visit(F&& f, const TVariant<Ts...>& v) {
    return ::NVariant::VisitImpl(std::forward<F>(f), v);
}

template <class F, class... Ts>
decltype(auto) Visit(F&& f, TVariant<Ts...>&& v) {
    return ::NVariant::VisitImpl(std::forward<F>(f), std::move(v));
}

template <class F, class... Ts>
decltype(auto) Visit(F&& f, const TVariant<Ts...>&& v) {
    return ::NVariant::VisitImpl(std::forward<F>(f), std::move(v));
}


template <class T, class... Ts>
constexpr bool HoldsAlternative(const TVariant<Ts...>& v) noexcept {
    static_assert(::NVariant::TIndexOf<T, Ts...>::value != TVARIANT_NPOS, "T not in types");
    return ::NVariant::TIndexOf<T, Ts...>::value == v.index();
}


namespace NVariant {
    template <size_t I, class V>
    decltype(auto) GetImpl(V&& v) {
        Y_ENSURE_EX(v.index() == I, TWrongVariantError());
        return ::NVariant::TVariantAccessor::Get<I>(std::forward<V>(v));
    }
}

template <size_t I, class... Ts>
decltype(auto) Get(TVariant<Ts...>& v) {
    return ::NVariant::GetImpl<I>(v);
}

template <size_t I, class... Ts>
decltype(auto) Get(const TVariant<Ts...>& v) {
    return ::NVariant::GetImpl<I>(v);
}

template <size_t I, class... Ts>
decltype(auto) Get(TVariant<Ts...>&& v) {
    return ::NVariant::GetImpl<I>(std::move(v));
}

template <size_t I, class... Ts>
decltype(auto) Get(const TVariant<Ts...>&& v) {
    return ::NVariant::GetImpl<I>(std::move(v));
}


namespace NVariant {
    template <class T, class V>
    decltype(auto) GetImpl(V&& v) {
        return ::Get< ::NVariant::TAlternativeIndex<T, std::decay_t<V>>::value>(std::forward<V>(v));
    }
}

template <class T, class... Ts>
decltype(auto) Get(TVariant<Ts...>& v) {
    return ::NVariant::GetImpl<T>(v);
}

template <class T, class... Ts>
decltype(auto) Get(const TVariant<Ts...>& v) {
    return ::NVariant::GetImpl<T>(v);
}

template <class T, class... Ts>
decltype(auto) Get(TVariant<Ts...>&& v) {
    return ::NVariant::GetImpl<T>(std::move(v));
}

template <class T, class... Ts>
decltype(auto) Get(const TVariant<Ts...>&& v) {
    return ::NVariant::GetImpl<T>(std::move(v));
}


template <size_t I, class... Ts>
auto* GetIf(TVariant<Ts...>* v) noexcept {
    return v != nullptr && I == v->index() ? &::NVariant::TVariantAccessor::Get<I>(*v) : nullptr;
}

template <size_t I, class... Ts>
const auto* GetIf(const TVariant<Ts...>* v) noexcept {
    return v != nullptr && I == v->index() ? &::NVariant::TVariantAccessor::Get<I>(*v) : nullptr;
}

template <class T, class... Ts>
T* GetIf(TVariant<Ts...>* v) noexcept {
    return ::GetIf< ::NVariant::TIndexOf<T, Ts...>::value>(v);
}

template <class T, class... Ts>
const T* GetIf(const TVariant<Ts...>* v) noexcept {
    return ::GetIf< ::NVariant::TIndexOf<T, Ts...>::value>(v);
}

template <class... Ts>
struct THash<TVariant<Ts...>> {
public:
    inline size_t operator()(const TVariant<Ts...>& v) const {
        const size_t tagHash = IntHash(v.index());
        const size_t valueHash = v.valueless_by_exception() ? 0 : Visit(
            [](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                return ::THash<T>{}(value);
            }, v);
        return CombineHashes(tagHash, valueHash);
    }
};

/* Unit type intended for use as a well-behaved empty alternative in TVariant.
 * In particular, a variant of non-default-constructible types may list TMonostate
 * as its first alternative: this makes the variant itself default-constructible.
 */
struct TMonostate {};

constexpr bool operator<(TMonostate, TMonostate) noexcept {
    return false;
}
constexpr bool operator>(TMonostate, TMonostate) noexcept {
    return false;
}
constexpr bool operator<=(TMonostate, TMonostate) noexcept {
    return true;
}
constexpr bool operator>=(TMonostate, TMonostate) noexcept {
    return true;
}
constexpr bool operator==(TMonostate, TMonostate) noexcept {
    return true;
}
constexpr bool operator!=(TMonostate, TMonostate) noexcept {
    return false;
}

template <>
struct THash<TMonostate> {
public:
    inline constexpr size_t operator()(TMonostate) const noexcept {
        return 1;
    }
};

namespace NVariant {
    template <size_t I, class... Ts>
    TVariantAlternativeType<I, TVariant<Ts...>>& TVariantAccessor::Get(TVariant<Ts...>& v) {
        return *v.template ReinterpretAs<TVariantAlternativeType<I, TVariant<Ts...>>>();
    }

    template <size_t I, class... Ts>
    const TVariantAlternativeType<I, TVariant<Ts...>>& TVariantAccessor::Get(
        const TVariant<Ts...>& v) {
        return *v.template ReinterpretAs<TVariantAlternativeType<I, TVariant<Ts...>>>();
    }

    template <size_t I, class... Ts>
    TVariantAlternativeType<I, TVariant<Ts...>>&& TVariantAccessor::Get(TVariant<Ts...>&& v) {
        return std::move(*v.template ReinterpretAs<TVariantAlternativeType<I, TVariant<Ts...>>>());
    }

    template <size_t I, class... Ts>
    const TVariantAlternativeType<I, TVariant<Ts...>>&& TVariantAccessor::Get(
        const TVariant<Ts...>&& v) {
        return std::move(*v.template ReinterpretAs<TVariantAlternativeType<I, TVariant<Ts...>>>());
    }

    template <class... Ts>
    constexpr size_t TVariantAccessor::Index(const TVariant<Ts...>& v) noexcept {
        return v.Index_;
    }
}
