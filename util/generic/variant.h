#pragma once

#include "variant_traits.h"
#include "variant_visitors.h"

#include <util/digest/numeric.h>
#include <util/generic/typetraits.h>
#include <util/system/yassert.h>

template <class T>
struct TVariantTypeTag {}; // aka std::in_place_type_t

template <size_t I>
struct TVariantIndexTag {}; // aka std::in_place_index_t


class TWrongVariantError : public yexception {};


template <size_t I, class T>
struct TVariantAlternative;

template <size_t I, class... Ts>
struct TVariantAlternative<I, TVariant<Ts...>> {
    using type = typename NVariant::TTypeByIndex<I, Ts...>::type;
};


template <class T>
struct TVariantSize;

template <class... Ts>
struct TVariantSize<TVariant<Ts...>> : std::integral_constant<size_t, sizeof...(Ts)> {};


template <class F, class V>
decltype(auto) Visit(F&& f, V&& v);


constexpr size_t TVARIANT_NPOS = NVariant::T_NPOS;


//! |std::variant (c++17)| poor substitute of discriminated union.
template <class... Ts>
class TVariant {
    template <class T>
    using TIndex = NVariant::TIndexOf<std::decay_t<T>, Ts...>;

    using T_0 = typename TVariantAlternative<0, TVariant>::type;

    static_assert(NVariant::TTypeTraits<Ts...>::TNoRefs::value,
        "TVariant type arguments cannot be references.");
    static_assert(NVariant::TTypeTraits<Ts...>::TNoVoids::value,
        "TVariant type arguments cannot be void.");
    static_assert(NVariant::TTypeTraits<Ts...>::TNoArrays::value,
        "TVariant type arguments cannot be arrays.");
    static_assert(NVariant::TTypeTraits<Ts...>::TNotEmpty::value,
        "TVariant type list cannot be empty.");

public:
    TVariant() noexcept(std::is_nothrow_default_constructible<T_0>::value) {
        static_assert(std::is_default_constructible<T_0>::value,
            "First alternative must be default constructible");
        EmplaceImpl<T_0>();
    }

    TVariant(const TVariant& rhs) {
        if (!rhs.ValuelessByException()) {
            CopyVariant(rhs);
        }
    }

    TVariant(TVariant&& rhs) noexcept(
        TConjunction<std::is_nothrow_move_constructible<Ts>...>::value)
    {
        if (!rhs.ValuelessByException()) {
            MoveVariant(rhs);
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
        EmplaceImpl<typename TVariantAlternative<I, TVariant>::type>(std::forward<TArgs>(args)...);
    }

    ~TVariant() {
        Destroy();
    }

    TVariant& operator=(const TVariant& rhs) {
        if (Y_UNLIKELY(rhs.ValuelessByException())) {
            if (Y_LIKELY(!ValuelessByException())) {
                DestroyImpl();
                Index_ = TVARIANT_NPOS;
            }
        } else if (Index() == rhs.Index()) {
            ::Visit(NVariant::TVisitorCopyAssign<Ts...>{ Storage_ }, rhs);
        } else {
            Destroy();
            try {
                CopyVariant(rhs);
            } catch (...) {
                Index_ = TVARIANT_NPOS;
                throw;
            }
        }
        return *this;
    }

    TVariant& operator=(TVariant&& rhs) {
        if (Y_UNLIKELY(rhs.ValuelessByException())) {
            if (Y_LIKELY(!ValuelessByException())) {
                DestroyImpl();
                Index_ = TVARIANT_NPOS;
            }
        } else if (Index() == rhs.Index()) {
            ::Visit(NVariant::TVisitorMoveAssign<Ts...>{ Storage_ }, rhs);
        } else {
            Destroy();
            try {
                MoveVariant(rhs);
            } catch (...) {
                Index_ = TVARIANT_NPOS;
                throw;
            }
        }
        return *this;
    }

    template <class T>
    std::enable_if_t<!std::is_same<std::decay_t<T>, TVariant>::value,
    TVariant&> operator=(T&& value) {
        if (HoldsAlternative<std::decay_t<T>>()) {
            *ReinterpretAs<T>() = std::forward<T>(value);
        } else {
            Emplace<T>(std::forward<T>(value));
        }
        return *this;
    }

    void Swap(TVariant& rhs) {
        if (!ValuelessByException() || !rhs.ValuelessByException()) {
            if (Index() == rhs.Index()) {
                ::Visit(NVariant::TVisitorSwap<Ts...>{ Storage_ }, rhs);
            } else {
                TVariant tmp(rhs);
                rhs.Destroy();
                rhs.MoveVariant(*this);
                Destroy();
                MoveVariant(tmp);
            }
        }
    }

    template <class T, class... TArgs>
    T& Emplace(TArgs&&... args) {
        Destroy();
        try {
            return EmplaceImpl<T>(std::forward<TArgs>(args)...);
        } catch (...) {
            Index_ = TVARIANT_NPOS;
            throw;
        }
    };

    template <size_t I, class... TArgs, class T = typename TVariantAlternative<I, TVariant>::type>
    T& Emplace(TArgs&&... args) {
        return Emplace<T>(std::forward<TArgs>(args)...);
    };

    bool operator==(const TVariant& rhs) const {
        return Index_ == rhs.Index_ &&
            (rhs.ValuelessByException() ||
            ::Visit(NVariant::TVisitorEquals<TVariant>{ *this }, rhs));
    }

    bool operator!=(const TVariant& rhs) const {
        return !(*this == rhs);
    }

    template <class T>
    std::enable_if_t<!std::is_same<std::decay_t<T>, TVariant>::value,
    bool> operator==(const T& value) const {
        return HoldsAlternative<T>() && *ReinterpretAs<T>() == value;
    }

    //! Casts the instance to a given type.
    template <class T>
    [[deprecated("use Get() instead")]] T& As() {
        return Get<T>();
    }

    //! Similar to its non-const version.
    template <class T>
    [[deprecated("use Get() instead")]] const T& As() const {
        return Get<T>();
    }

    //! Standart integration
    template <class T>
    T& Get() {
        Y_ENSURE_EX(HoldsAlternative<T>(), TWrongVariantError());
        return *ReinterpretAs<T>();
    }

    //! Standart integration
    template <class T>
    const T& Get() const {
        Y_ENSURE_EX(HoldsAlternative<T>(), TWrongVariantError());
        return *ReinterpretAs<T>();
    }

    //! Casts the instance to a type given by its index in variant type list.
    template <size_t I, class T = typename TVariantAlternative<I, TVariant>::type>
    T& Get() {
        return Get<T>();
    }

    //! Similar to its non-const version.
    template <size_t I, class T = typename TVariantAlternative<I, TVariant>::type>
    const T& Get() const {
        return Get<T>();
    }

    //! Checks if the instance holds a given of a given type.
    //! Returns the pointer to the value on success or |nullptr| on failure.
    template <class T>
    [[deprecated("use GetIf() instead")]] T* TryAs() noexcept {
        return GetIf<T>();
    }

    //! Similar to its non-const version.
    template <class T>
    [[deprecated("use GetIf() instead")]] const T* TryAs() const noexcept {
        return GetIf<T>();
    }

    //! Standart integration
    template <class T>
    T* GetIf() noexcept {
        static_assert(TIndex<T>::value != TVARIANT_NPOS, "Type not in TVariant.");
        return HoldsAlternative<T>() ? ReinterpretAs<T>() : nullptr;
    }

    //! Standart integration
    template <class T>
    const T* GetIf() const noexcept {
        static_assert(TIndex<T>::value != TVARIANT_NPOS, "Type not in TVariant.");
        return HoldsAlternative<T>() ? ReinterpretAs<T>() : nullptr;
    }

    //! Checks if the instance holds a type given by its index in variant type list.
    //! Returns the pointer to the value on success or |nullptr| on failure.
    template <size_t I, class T = typename TVariantAlternative<I, TVariant>::type>
    T* GetIf() noexcept {
        return GetIf<T>();
    }

    //! Similar to its non-const version.
    template <size_t I, class T = typename TVariantAlternative<I, TVariant>::type>
    const T* GetIf() const noexcept {
        return GetIf<T>();
    }

    //! Returns |true| iff the instance holds a value of a given type.
    template <class T>
    [[deprecated("use HoldsAlternative() instead")]] constexpr bool Is() const noexcept {
        return HoldsAlternative<T>();
    }

    //! Standart integration
    template <class T>
    constexpr bool HoldsAlternative() const noexcept {
        static_assert(TIndex<T>::value != TVARIANT_NPOS, "Type not in TVariant.");
        return Index_ == TIndex<T>::value;
    }

    //! Pass mutable internal value to visitor
    template <class Visitor>
    [[deprecated("use global Visit() instead")]] decltype(auto) Visit(Visitor&& visitor) {
        return ::Visit(std::forward<Visitor>(visitor), *this);
    }

    //! Pass const internal value to visitor
    template <class Visitor>
    [[deprecated("use global Visit() instead")]] decltype(auto) Visit(Visitor&& visitor) const {
        return ::Visit(std::forward<Visitor>(visitor), *this);
    }

    //! Returns the discriminating index of the instance.
    [[deprecated("use Index() instead")]] constexpr size_t Tag() const noexcept {
        return Index();
    }

    //! Standart integration
    constexpr size_t Index() const noexcept {
        return Index_;
    }

    //! Returns the discriminating index of the given type.
    template <class T>
    static constexpr size_t TagOf() noexcept {
        return TIndex<T>::value;
    }

    template <class T>
    static constexpr auto TypeOf() noexcept {
        return TVariantTypeTag<T>();
    }

    /* A TVariant that is valueless by exception is treated as being in an invalid state:
     * Index returns TVARIANT_NPOS, Get and Visit throw TWrongVariantError.
     */
    constexpr bool ValuelessByException() const noexcept {
        return Index_ == TVARIANT_NPOS;
    }

private:
    void Destroy() noexcept {
        if (!ValuelessByException()) {
            DestroyImpl();
        }
    }

    void DestroyImpl() noexcept {
        ::Visit(NVariant::TVisitorDestroy{}, *this);
    }

    template <class T, class... TArgs>
    T& EmplaceImpl(TArgs&&... args) noexcept(
        std::is_nothrow_constructible<T, TArgs...>::value)
    {
        static_assert(TIndex<T>::value != TVARIANT_NPOS, "Type not in TVariant.");
        Index_ = TIndex<T>::value;
        new (&Storage_) std::decay_t<T>(std::forward<TArgs>(args)...);
        return *ReinterpretAs<T>();
    };

    void CopyVariant(const TVariant& rhs) {
        ::Visit(NVariant::TVisitorCopyConstruct<TVariant>{ this }, rhs);
    }

    void MoveVariant(TVariant& rhs) {
        ::Visit(NVariant::TVisitorMoveConstruct<TVariant>{ this }, rhs);
    }

    template <class T>
    auto* ReinterpretAs() noexcept {
        return reinterpret_cast<std::decay_t<T>*>(&Storage_);
    }

    template <class T>
    const auto* ReinterpretAs() const noexcept {
        return reinterpret_cast<const std::decay_t<T>*>(&Storage_);
    }

private:
    size_t Index_ = TVARIANT_NPOS;
    std::aligned_union_t<0, Ts...> Storage_;
};


template <class F, class V>
decltype(auto) Visit(F&& f, V&& v) {
    Y_ENSURE_EX(!v.ValuelessByException(), TWrongVariantError());
    using FRef = decltype(std::forward<F>(f));
    using VRef = decltype(std::forward<V>(v));
    static_assert(NVariant::CheckReturnTypes<FRef, VRef>(
        std::make_index_sequence<TVariantSize<std::decay_t<V>>::value>{}), "");
    using ReturnType = NVariant::TReturnType<FRef, VRef>;
    return NVariant::VisitWrapForVoid(
        std::forward<F>(f), std::forward<V>(v), std::is_same<ReturnType, void>{});
}


template <class... Ts>
struct THash<TVariant<Ts...>> {
public:
    inline size_t operator()(const ::TVariant<Ts...>& v) const {
        const size_t tagHash = IntHash(v.Index());
        const size_t valueHash = v.ValuelessByException() ? v.Visit(NVariant::TVisitorHash()) : 0;
        return CombineHashes(tagHash, valueHash);
    }
};

/* Unit type intended for use as a well-behaved empty alternative in TVariant.
 * In particular, a variant of non-default-constructible types may list TMonostate
 * as its first alternative: this makes the variant itself default-constructible.
 */
struct TMonostate {};

constexpr bool operator<(TMonostate, TMonostate) noexcept { return false; }
constexpr bool operator>(TMonostate, TMonostate) noexcept { return false; }
constexpr bool operator<=(TMonostate, TMonostate) noexcept { return true; }
constexpr bool operator>=(TMonostate, TMonostate) noexcept { return true; }
constexpr bool operator==(TMonostate, TMonostate) noexcept { return true; }
constexpr bool operator!=(TMonostate, TMonostate) noexcept { return false; }

template <>
struct THash<TMonostate> {
public:
    inline constexpr size_t operator()(TMonostate) const noexcept { return 1; }
};
