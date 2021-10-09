#pragma once

#include "hash.h"

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

template <class T, class V>
struct TVariantIndex;

template <class T, class... Ts>
struct TVariantIndex<T, std::variant<Ts...>>: ::NVariant::TIndexOf<T, Ts...> {};

template <class T, class V>
constexpr size_t TVariantIndexV = TVariantIndex<T, V>::value;

template <class... Ts>
struct THash<std::variant<Ts...>> {
public:
    inline size_t operator()(const std::variant<Ts...>& v) const {
        const size_t tagHash = IntHash(v.index());
        const size_t valueHash = v.valueless_by_exception() ? 0 : std::visit([](const auto& value) {
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
