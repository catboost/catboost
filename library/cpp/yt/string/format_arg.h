#pragma once

#include <util/generic/strbuf.h>

#include <array>
#include <string_view>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

class TStringBuilderBase;

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class T>
constexpr std::string_view QualidName();

template <class T>
constexpr bool IsNYTName();

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

// Base used for flag checks for each type independently.
// Use it for overrides.
class TFormatArgBase
{
public:
    // TODO(arkady-e1ppa): Consider more strict formatting rules.
    static constexpr std::array ConversionSpecifiers = {
        'v', 'c', 's', 'd', 'i', 'o',
        'x', 'X', 'u', 'f', 'F', 'e', 'E',
        'a', 'A', 'g', 'G', 'n', 'p'
    };

    static constexpr std::array FlagSpecifiers = {
        '-', '+', ' ', '#', '0',
        '1', '2', '3', '4', '5',
        '6', '7', '8', '9',
        '*', '.', 'h', 'l', 'j',
        'z', 't', 'L', 'q', 'Q'
    };

    template <class T>
    static constexpr bool IsSpecifierList = requires (T t) {
        [] <size_t N> (std::array<char, N>) { } (t);
    };

    // Hot = |true| adds specifiers to the beggining
    // of a new array.
    template <bool Hot, size_t N, std::array<char, N> List, class TFrom = TFormatArgBase>
    static consteval auto ExtendConversion();

    template <bool Hot, size_t N, std::array<char, N> List, class TFrom = TFormatArgBase>
    static consteval auto ExtendFlags();

private:
    template <bool Hot, size_t N, std::array<char, N> List, auto* From>
    static consteval auto AppendArrays();
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TFormatArg
    : public TFormatArgBase
{ };

// Ultimate customization point mechanism --- define an overload
// of FormatValue in order to support formatting of your type.
// Semantic requirement:
// Said field must be constexpr.
template <class T>
concept CFormattable =
    requires {
        TFormatArg<T>::ConversionSpecifiers;
        requires TFormatArgBase::IsSpecifierList<decltype(TFormatArg<T>::ConversionSpecifiers)>;

        TFormatArg<T>::FlagSpecifiers;
        requires TFormatArgBase::IsSpecifierList<decltype(TFormatArg<T>::FlagSpecifiers)>;
    } &&
    requires (
        TStringBuilderBase* builder,
        const T& value,
        TStringBuf spec
    ) {
        { FormatValue(builder, value, spec) } -> std::same_as<void>;
    };

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define FORMAT_ARG_INL_H_
#include "format_arg-inl.h"
#undef FORMAT_ARG_INL_H_
