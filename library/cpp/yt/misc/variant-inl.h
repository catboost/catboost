#ifndef VARIANT_INL_H_
#error "Direct inclusion of this file is not allowed, include variant.h"
// For the sake of sane code completion.
#include "variant.h"
#endif

#include <type_traits>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class T>
struct TIndexOf<T>
{
    static constexpr size_t Value = std::numeric_limits<size_t>::max();
};

template <class T, class T0, class... Ts>
struct TIndexOf<T, T0, Ts...>
{
    static constexpr size_t Value = std::is_same_v<T, T0>
        ? 0
        : 1 + TIndexOf<T, Ts...>::Value;
};

template <size_t Index, class... Ts>
struct TVariantFormatter;

template <size_t Index>
struct TVariantFormatter<Index>
{
    template <class TVariant>
    static void Do(TStringBuilderBase* /*builder*/, const TVariant& /*variant*/, TStringBuf /*spec*/)
    { }
};

template <size_t Index, class T, class... Ts>
struct TVariantFormatter<Index, T, Ts...>
{
    template <class TVariant>
    static void Do(TStringBuilderBase* builder, const TVariant& variant, TStringBuf spec)
    {
        if (variant.index() == Index) {
            FormatValue(builder, std::get<Index>(variant), spec);
        } else {
            TVariantFormatter<Index + 1, Ts...>::Do(builder, variant, spec);
        }
    }
};

} // namespace NDetail

template <class... Ts>
void FormatValue(TStringBuilderBase* builder, const std::variant<Ts...>& variant, TStringBuf spec)
{
    NDetail::TVariantFormatter<0, Ts...>::Do(builder, variant, spec);
}

template <class... Ts>
TString ToString(const std::variant<Ts...>& variant)
{
    return ToStringViaBuilder(variant);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
