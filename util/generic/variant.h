#pragma once

#include "variant_traits.h"

#include <util/generic/hash.h>

template <class T>
using TVariantTypeTag = std::in_place_type_t<T>;

template <size_t I, class V>
using TVariantAlternative = std::variant_alternative<I, V>;

template <size_t I, class V>
using TVariantAlternativeType = std::variant_alternative_t<I, V>;

template <size_t I>
using TVariantIndexTag = std::in_place_index_t<I>;

template <class T, class V>
struct TVariantIndex;

template <class T, class... Ts>
struct TVariantIndex<T, TVariant<Ts...>>: ::NVariant::TIndexOf<T, Ts...> {};

template <class T, class V>
constexpr size_t TVariantIndexV = TVariantIndex<T, V>::value;

template <class V>
using TVariantSize = std::variant_size<V>;

constexpr size_t TVARIANT_NPOS = std::variant_npos;

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

template <class F, class... Ts>
decltype(auto) Visit(F&& f, TVariant<Ts...>& v) {
    return std::visit(std::forward<F>(f), v);
}

template <class F, class... Ts>
decltype(auto) Visit(F&& f, const TVariant<Ts...>& v) {
    return std::visit(std::forward<F>(f), v);
}

template <class F, class... Ts>
decltype(auto) Visit(F&& f, TVariant<Ts...>&& v) {
    return std::visit(std::forward<F>(f), std::move(v));
}

template <class F, class... Ts>
decltype(auto) Visit(F&& f, const TVariant<Ts...>&& v) {
    return std::visit(std::forward<F>(f), std::move(v));
}

template <class T, class... Ts>
constexpr bool HoldsAlternative(const TVariant<Ts...>& v) noexcept {
    return std::holds_alternative<T>(v);
}

template <size_t I, class... Ts>
decltype(auto) Get(TVariant<Ts...>& v) {
    return std::get<I>(v);
}

template <size_t I, class... Ts>
decltype(auto) Get(const TVariant<Ts...>& v) {
    return std::get<I>(v);
}

template <size_t I, class... Ts>
decltype(auto) Get(TVariant<Ts...>&& v) {
    return std::get<I>(std::move(v));
}

template <size_t I, class... Ts>
decltype(auto) Get(const TVariant<Ts...>&& v) {
    return std::get<I>(std::move(v));
}

template <class T, class... Ts>
decltype(auto) Get(TVariant<Ts...>& v) {
    return std::get<T>(v);
}

template <class T, class... Ts>
decltype(auto) Get(const TVariant<Ts...>& v) {
    return std::get<T>(v);
}

template <class T, class... Ts>
decltype(auto) Get(TVariant<Ts...>&& v) {
    return std::get<T>(std::move(v));
}

template <class T, class... Ts>
decltype(auto) Get(const TVariant<Ts...>&& v) {
    return std::get<T>(std::move(v));
}

template <size_t I, class... Ts>
auto* GetIf(TVariant<Ts...>* v) noexcept {
    return std::get_if<I>(v);
}

template <size_t I, class... Ts>
const auto* GetIf(const TVariant<Ts...>* v) noexcept {
    return std::get_if<I>(v);
}

template <class T, class... Ts>
T* GetIf(TVariant<Ts...>* v) noexcept {
    return std::get_if<T>(v);
}

template <class T, class... Ts>
const T* GetIf(const TVariant<Ts...>* v) noexcept {
    return std::get_if<T>(v);
}

template <class... Ts>
struct THash<TVariant<Ts...>> {
public:
    inline size_t operator()(const TVariant<Ts...>& v) const {
        const size_t tagHash = IntHash(v.index());
        const size_t valueHash = v.valueless_by_exception() ? 0 : Visit([](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            return ::THash<T>{}(value);
        }, v);
        return CombineHashes(tagHash, valueHash);
    }
};

using TMonostate = std::monostate;
/* Unit type intended for use as a well-behaved empty alternative in TVariant.
 * In particular, a variant of non-default-constructible types may list TMonostate
 * as its first alternative: this makes the variant itself default-constructible.
 */

template <>
struct THash<TMonostate> {
public:
    inline constexpr size_t operator()(TMonostate) const noexcept {
        return 1;
    }
};
