#pragma once

#include "format_string.h"

#include <util/generic/string.h>

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

template <class... TArgs>
TString Format(TFormatString<TArgs...> format, TArgs&&... args);

////////////////////////////////////////////////////////////////////////////////

// StringBuilder(Base) definition.

//! A simple helper for constructing strings by a sequence of appends.
class TStringBuilderBase
{
public:
    virtual ~TStringBuilderBase() = default;

    char* Preallocate(size_t size);

    void Reserve(size_t size);

    size_t GetLength() const;

    TStringBuf GetBuffer() const;

    void Advance(size_t size);

    void AppendChar(char ch);
    void AppendChar(char ch, int n);

    void AppendString(TStringBuf str);
    void AppendString(const char* str);

    template <size_t Length, class... TArgs>
    void AppendFormat(const char (&format)[Length], TArgs&&... args);
    template <class... TArgs>
    void AppendFormat(TStringBuf format, TArgs&&... args);

    void Reset();

protected:
    char* Begin_ = nullptr;
    char* Current_ = nullptr;
    char* End_ = nullptr;

    virtual void DoReset() = 0;
    virtual void DoReserve(size_t newLength) = 0;

    static constexpr size_t MinBufferLength = 128;
};

////////////////////////////////////////////////////////////////////////////////

class TStringBuilder
    : public TStringBuilderBase
{
public:
    TString Flush();

protected:
    TString Buffer_;

    void DoReset() override;
    void DoReserve(size_t size) override;
};

////////////////////////////////////////////////////////////////////////////////

template <class... TArgs>
void Format(TStringBuilderBase* builder, TFormatString<TArgs...> format, TArgs&&... args);

////////////////////////////////////////////////////////////////////////////////

template <class T>
TString ToStringViaBuilder(const T& value, TStringBuf spec = TStringBuf("v"));

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

// Allows insertion of text conditionally.
// Usage:
/*
NYT::Format(
    "Value is %v%v",
    42,
    MakeFormatterWrapper([&] (auto* builder) {
        if (PossiblyMissingInfo_) {
            builder->AppendString(", PossiblyMissingInfo: ");
            FormatValue(builder, PossiblyMissingInfo_, "v");
        }
    }));
 */
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
    TStringBuf /*spec*/);

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

    // NB(arkady-e1ppa): We actually have to
    // forward declare this method as above
    // and friend-declare it as specialization
    // here because clang is stupid and would
    // treat this friend declartion as a hidden friend
    // declaration which in turn is treated as a separate symbol
    // causing linker to not find the actual definition.
    friend void FormatValue<>(
        TStringBuilderBase* builder,
        const TLazyMultiValueFormatter& value,
        TStringBuf /*spec*/);

private:
    const TStringBuf Format_;
    const std::tuple<TArgs...> Args_;
};

template <class ... Args>
auto MakeLazyMultiValueFormatter(TStringBuf format, Args&&... args);

////////////////////////////////////////////////////////////////////////////////

/*
    Example:

    FormatVector("One: %v, Two: %v, Three: %v", {1, 2, 3})
    => "One: 1, Two: 2, Three: 3"
*/
template <size_t Length, class TVector>
void FormatVector(
    TStringBuilderBase* builder,
    const char (&format)[Length],
    const TVector& vec);

template <class TVector>
void FormatVector(
    TStringBuilderBase* builder,
    TStringBuf format,
    const TVector& vec);

template <size_t Length, class TVector>
TString FormatVector(
    const char (&format)[Length],
    const TVector& vec);

template <class TVector>
TString FormatVector(
    TStringBuf format,
    const TVector& vec);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define FORMAT_INL_H_
#include "format-inl.h"
#undef FORMAT_INL_H_
