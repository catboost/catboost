#pragma once

#include "typetraits.h"
#include "yexception.h"
#include "hash.h"

#include <utility>
#include <type_traits>
#include <variant>

namespace NVariant {
    template <class X, class... Ts>
    constexpr size_t IndexOfImpl() {
        bool bs[] = {std::is_same<X, Ts>::value...};
        for (size_t i = 0; i < sizeof...(Ts); ++i) {
            if (bs[i]) {
                return i;
            }
        }
        return std::variant_npos;
    }

    template <class X, class... Ts>
    struct TIndexOf: std::integral_constant<size_t, IndexOfImpl<X, Ts...>()> {};
}

using TWrongVariantError = std::bad_variant_access;

template <class T>
using TVariantTypeTag = std::in_place_type_t<T>;

template <size_t I, class V>
using TVariantAlternative = std::variant_alternative<I, V>;

template <size_t I, class V>
using TVariantAlternativeType = std::variant_alternative_t<I, V>;

template <class T, class V>
struct TVariantIndex;

template <class T, class... Ts>
struct TVariantIndex<T, std::variant<Ts...>>: ::NVariant::TIndexOf<T, Ts...> {};

template <class T, class V>
constexpr size_t TVariantIndexV = TVariantIndex<T, V>::value;

template <class V>
using TVariantSize = std::variant_size<V>;

template <class F, class... Ts>
decltype(auto) Visit(F&& f, std::variant<Ts...>& v) {
    return std::visit(std::forward<F>(f), v);
}

template <class F, class... Ts>
decltype(auto) Visit(F&& f, const std::variant<Ts...>& v) {
    return std::visit(std::forward<F>(f), v);
}

template <class F, class... Ts>
decltype(auto) Visit(F&& f, std::variant<Ts...>&& v) {
    return std::visit(std::forward<F>(f), std::move(v));
}

template <class F, class... Ts>
decltype(auto) Visit(F&& f, const std::variant<Ts...>&& v) {
    return std::visit(std::forward<F>(f), std::move(v));
}

template <size_t I, class... Ts>
decltype(auto) Get(std::variant<Ts...>& v) {
    return std::get<I>(v);
}

template <size_t I, class... Ts>
decltype(auto) Get(const std::variant<Ts...>& v) {
    return std::get<I>(v);
}

template <size_t I, class... Ts>
decltype(auto) Get(std::variant<Ts...>&& v) {
    return std::get<I>(std::move(v));
}

template <size_t I, class... Ts>
decltype(auto) Get(const std::variant<Ts...>&& v) {
    return std::get<I>(std::move(v));
}

template <class T, class... Ts>
decltype(auto) Get(std::variant<Ts...>& v) {
    return std::get<T>(v);
}

template <class T, class... Ts>
decltype(auto) Get(const std::variant<Ts...>& v) {
    return std::get<T>(v);
}

template <class T, class... Ts>
decltype(auto) Get(std::variant<Ts...>&& v) {
    return std::get<T>(std::move(v));
}

template <class T, class... Ts>
decltype(auto) Get(const std::variant<Ts...>&& v) {
    return std::get<T>(std::move(v));
}

template <class... Ts>
struct THash<std::variant<Ts...>> {
public:
    inline size_t operator()(const std::variant<Ts...>& v) const {
        const size_t tagHash = IntHash(v.index());
        const size_t valueHash = v.valueless_by_exception() ? 0 : Visit([](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            return ::THash<T>{}(value);
        }, v);
        return CombineHashes(tagHash, valueHash);
    }
};

template <>
struct THash<std::monostate> {
public:
    inline constexpr size_t operator()(std::monostate) const noexcept {
        return 1;
    }
};
