#ifndef FORMAT_ARG_INL_H_
#error "Direct inclusion of this file is not allowed, include format_arg.h"
// For the sake of sane code completion.
#include "format_arg.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class T>
constexpr std::string_view QualidName()
{
    constexpr size_t openBracketOffset = 5;
    constexpr size_t closeBracketOffset = 1;
    constexpr std::string_view func = __PRETTY_FUNCTION__;
    constexpr auto left = std::find(std::begin(func), std::end(func), '[') + openBracketOffset;
    return std::string_view(left, std::prev(std::end(func), closeBracketOffset));
}

template <class T>
constexpr bool IsNYTName()
{
    constexpr auto qualidName = QualidName<T>();
    return qualidName.find("NYT::") == 0;
}

template <class T>
constexpr bool IsStdName()
{
    constexpr auto qualidName = QualidName<T>();
    return qualidName.find("std::") == 0;
}

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

template <bool Hot, size_t N, std::array<char, N> List, class TFrom>
consteval auto TFormatArgBase::ExtendConversion()
{
    static_assert(std::same_as<TFrom, TFormatArgBase> || CFormattable<TFrom>);
    return AppendArrays<Hot, N, List, &TFormatArg<TFrom>::ConversionSpecifiers>();
}

template <bool Hot, size_t N, std::array<char, N> List, class TFrom>
consteval auto TFormatArgBase::ExtendFlags()
{
    static_assert(std::same_as<TFrom, TFormatArgBase> || CFormattable<TFrom>);
    return AppendArrays<Hot, N, List, &TFormatArg<TFrom>::FlagSpecifiers>();
}

template <bool Hot, size_t N, std::array<char, N> List, auto* From>
consteval auto TFormatArgBase::AppendArrays()
{
    auto& from = *From;
    return [] <size_t... I, size_t... J> (
        std::index_sequence<I...>,
        std::index_sequence<J...>) {
            if constexpr (Hot) {
                return std::array{List[J]..., from[I]...};
            } else {
                return std::array{from[I]..., List[J]...};
            }
        } (
            std::make_index_sequence<std::size(from)>(),
            std::make_index_sequence<N>());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
