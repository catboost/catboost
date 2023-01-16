#pragma once

#include "typetraits.h"
#include "yexception.h"

#include <utility>
#include <type_traits>
#include <variant>

template <class... Ts>
using TVariant = std::variant<Ts...>;

using TWrongVariantError = std::bad_variant_access;

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
