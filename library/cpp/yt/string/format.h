#pragma once

#include "string_builder.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

/*
 *  Format: a type-safe and fast formatting utility.
 *
 *  Basically works as a type-safe analogue of |sprintf| and is expected to
 *  be backwards-compatible with the latter.
 *
 *  Like Go's |Sprintf|, supports the ultimate format specifier |v|
 *  causing arguments to be emitted in default format.
 *  This is the default and preferred way of formatting things,
 *  which should be used in newer code.
 *
 *  |Format| may currently invoke |sprintf| internally for emitting numeric and some other
 *  types. You can always write your own optimized implementation, if you wish :)
 *
 *  In additional to the usual |sprintf|, supports a number of non-standard flags:
 *
 *  |q|   Causes the argument to be surrounded with single quotes (|'|).
 *        Applies to all types.
 *
 *  |Q|   Causes the argument to be surrounded with double quotes (|"|).
 *        Applies to all types.
 *
 *  |l|   The argument is emitted in "lowercase" style.
 *        Only applies to enums and bools.
 *
 *  The following argument types are supported:
 *
 *  Strings (including |const char*|, |TStringBuf|, and |TString|) and chars:
 *  Emitted as is. Fast.
 *
 *  Numerics and pointers:
 *  Emitted using |sprintf|. Maybe not that fast.
 *
 *  |bool|:
 *  Emitted either as |True| and |False| or |true| and |false| (if lowercase mode is ON).
 *
 *  Enums:
 *  Emitted in either camel (|SomeName|) or in lowercase-with-underscores style
 *  (|some_name|, if lowercase mode is ON).
 *
 *  Nullables:
 *  |std::nullopt| is emitted as |<null>|.
 *
 *  All others:
 *  Emitted as strings by calling |ToString|.
 *
 */

template <size_t Length, class... TArgs>
void Format(TStringBuilderBase* builder, const char (&format)[Length], TArgs&&... args);
template <class... TArgs>
void Format(TStringBuilderBase* builder, TStringBuf format, TArgs&&... args);

template <size_t Length, class... TArgs>
TString Format(const char (&format)[Length], TArgs&&... args);
template <class... TArgs>
TString Format(TStringBuf format, TArgs&&... args);

////////////////////////////////////////////////////////////////////////////////

template <class TRange, class TFormatter>
struct TFormattableView
{
    using TBegin = std::decay_t<decltype(std::declval<const TRange>().begin())>;
    using TEnd = std::decay_t<decltype(std::declval<const TRange>().end())>;

    TBegin RangeBegin;
    TEnd RangeEnd;
    TFormatter Formatter;
    size_t Limit = std::numeric_limits<size_t>::max();

    TBegin begin() const;
    TEnd end() const;
};

//! Annotates a given #range with #formatter to be applied to each item.
template <class TRange, class TFormatter>
TFormattableView<TRange, TFormatter> MakeFormattableView(
    const TRange& range,
    TFormatter&& formatter);

template <class TRange, class TFormatter>
TFormattableView<TRange, TFormatter> MakeShrunkFormattableView(
    const TRange& range,
    TFormatter&& formatter,
    size_t limit);

////////////////////////////////////////////////////////////////////////////////

template <class TFormatter>
struct TFormatterWrapper
{
    TFormatter Formatter;
};

template <class TFormatter>
TFormatterWrapper<TFormatter> MakeFormatterWrapper(
    TFormatter&& formatter);

////////////////////////////////////////////////////////////////////////////////

template <class... TArgs>
class TLazyMultiValueFormatter;

template <class... TArgs>
void FormatValue(
    TStringBuilderBase* builder,
    const TLazyMultiValueFormatter<TArgs...>& value,
    TStringBuf /*format*/);

//! A wrapper for a bunch of values that formats them lazily on demand.
/*!
 *  The intended use of this class is when you need to use the same formatted string
 *  in several places in the function (e.g. log message tags) and want both to avoid
 *  code duplication and premature formatting of the values until necessary.
 *
 *  NB: lvalues are captured by reference without lifetime extension.
 */
template <class... TArgs>
class TLazyMultiValueFormatter
    : private TNonCopyable
{
public:
    TLazyMultiValueFormatter(TStringBuf format, TArgs&&... args);

    friend void FormatValue<>(
        TStringBuilderBase* builder,
        const TLazyMultiValueFormatter& value,
        TStringBuf /*format*/);

private:
    const TStringBuf Format_;
    const std::tuple<TArgs...> Args_;
};

template <class ... Args>
auto MakeLazyMultiValueFormatter(TStringBuf format, Args&&... args);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define FORMAT_INL_H_
#include "format-inl.h"
#undef FORMAT_INL_H_
