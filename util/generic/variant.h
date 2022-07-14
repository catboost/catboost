#pragma once

#include "hash.h"

#include <variant>

template <class... Ts>
struct THash<std::variant<Ts...>> {
public:
    size_t operator()(const std::variant<Ts...>& v) const noexcept {
        return CombineHashes(
            IntHash(v.index()),
            v.valueless_by_exception() ? 0 : std::visit([](const auto& value) { return ComputeHash(value); }, v));
    }
};

template <>
struct THash<std::monostate> {
public:
    constexpr size_t operator()(std::monostate) const noexcept {
        return 1;
    }
};
