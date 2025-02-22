#pragma once

#include "format_arg.h"

#include <util/datetime/base.h>

#include <util/generic/string.h>

#include <util/string/strip.h>

#include <vector>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Formatters enable customizable way to turn an object into a string.
//! This default implementation uses |FormatValue|.
struct TDefaultFormatter
{
    template <class T>
    void operator()(TStringBuilderBase* builder, const T& obj) const
    {
        FormatValue(builder, obj, TStringBuf("v"));
    }
};

//! Bind spec to a formatter.
//! Used in ranges processing.
class TSpecBoundFormatter
{
public:
    explicit TSpecBoundFormatter(TStringBuf spec)
        : Spec_(spec)
    { }

    template <class T>
    void operator()(TStringBuilderBase* builder, const T& obj) const
    {
        FormatValue(builder, obj, Spec_);
    }

private:
    TStringBuf Spec_;
};

static constexpr TStringBuf DefaultJoinToStringDelimiter = ", ";
static constexpr TStringBuf DefaultKeyValueDelimiter = ": ";
static constexpr TStringBuf DefaultRangeEllipsisFormat = "...";

// ASCII characters from 0x20 = ' ' to 0x7e = '~' are printable.
static constexpr char PrintableASCIILow = 0x20;
static constexpr char PrintableASCIIHigh = 0x7e;
static constexpr TStringBuf IntToHexLowercase = "0123456789abcdef";
static constexpr TStringBuf IntToHexUppercase = "0123456789ABCDEF";

//! Joins a range of items into a string intermixing them with the delimiter.
/*!
 *  \param builder String builder where the output goes.
 *  \param begin Iterator pointing to the first item (inclusive).
 *  \param end Iterator pointing to the last item (not inclusive).
 *  \param formatter Formatter to apply to the items.
 *  \param delimiter A delimiter to be inserted between items: ", " by default.
 *  \return The resulting combined string.
 */
template <std::forward_iterator TIterator, class TFormatter>
void JoinToString(
    TStringBuilderBase* builder,
    const TIterator& begin,
    const TIterator& end,
    const TFormatter& formatter,
    TStringBuf delimiter = DefaultJoinToStringDelimiter);

template <std::forward_iterator TIterator, class TFormatter>
TString JoinToString(
    const TIterator& begin,
    const TIterator& end,
    const TFormatter& formatter,
    TStringBuf delimiter = DefaultJoinToStringDelimiter);

//! A handy shortcut with default formatter.
template <std::forward_iterator TIterator>
TString JoinToString(
    const TIterator& begin,
    const TIterator& end,
    TStringBuf delimiter = DefaultJoinToStringDelimiter);

//! Joins a collection of given items into a string intermixing them with the delimiter.
/*!
 *  \param collection A collection containing the items to be joined.
 *  \param formatter Formatter to apply to the items.
 *  \param delimiter A delimiter to be inserted between items; ", " by default.
 */
template <std::ranges::range TCollection, class TFormatter>
TString JoinToString(
    TCollection&& collection,
    const TFormatter& formatter,
    TStringBuf delimiter = DefaultJoinToStringDelimiter);

//! A handy shortcut with the default formatter.
template <std::ranges::range TCollection>
TString JoinToString(
    TCollection&& collection,
    TStringBuf delimiter = DefaultJoinToStringDelimiter);

//! Concatenates a bunch of TStringBuf-like instances into TString.
template <class... Ts>
TString ConcatToString(Ts... args);

//! Converts a range of items into strings.
template <std::forward_iterator TIter, class TFormatter>
std::vector<TString> ConvertToStrings(
    const TIter& begin,
    const TIter& end,
    const TFormatter& formatter,
    size_t maxSize = std::numeric_limits<size_t>::max());

//! A handy shortcut with the default formatter.
template <std::forward_iterator TIter>
std::vector<TString> ConvertToStrings(
    const TIter& begin,
    const TIter& end,
    size_t maxSize = std::numeric_limits<size_t>::max());

//! Converts a given collection of items into strings.
/*!
 *  \param collection A collection containing the items to be converted.
 *  \param formatter Formatter to apply to the items.
 *  \param maxSize Size limit for the resulting vector.
 */
template <std::ranges::range TCollection, class TFormatter>
std::vector<TString> ConvertToStrings(
    TCollection&& collection,
    const TFormatter& formatter,
    size_t maxSize = std::numeric_limits<size_t>::max());

//! A handy shortcut with default formatter.
template <std::ranges::range TCollection>
std::vector<TString> ConvertToStrings(
    TCollection&& collection,
    size_t maxSize = std::numeric_limits<size_t>::max());

////////////////////////////////////////////////////////////////////////////////

void UnderscoreCaseToCamelCase(TStringBuilderBase* builder, TStringBuf str);
TString UnderscoreCaseToCamelCase(TStringBuf str);

void CamelCaseToUnderscoreCase(TStringBuilderBase* builder, TStringBuf str);
TString CamelCaseToUnderscoreCase(TStringBuf str);

TString TrimLeadingWhitespaces(const TString& str);
TString Trim(const TString& str, const TString& whitespaces);

////////////////////////////////////////////////////////////////////////////////

//! Implemented for |[u]i(32|64)|.
template <class T>
char* WriteDecIntToBufferBackwards(char* ptr, T value);

//! Implemented for |[u]i(32|64)|.
template <class T>
char* WriteHexIntToBufferBackwards(char* ptr, T value, bool uppercase);

////////////////////////////////////////////////////////////////////////////////

struct TCaseInsensitiveStringHasher
{
    size_t operator()(TStringBuf arg) const;
};

struct TCaseInsensitiveStringEqualityComparer
{
    bool operator()(TStringBuf lhs, TStringBuf rhs) const;
};

////////////////////////////////////////////////////////////////////////////////

bool TryParseBool(TStringBuf value, bool* result);
bool ParseBool(TStringBuf value);
TStringBuf FormatBool(bool value);

////////////////////////////////////////////////////////////////////////////////

inline constexpr TStringBuf DefaultTruncatedMessage = "...<truncated>";

void TruncateStringInplace(std::string* string, int lengthLimit, TStringBuf truncatedSuffix = DefaultTruncatedMessage);
std::string TruncateString(std::string string, int lengthLimit, TStringBuf truncatedSuffix = DefaultTruncatedMessage);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define STRING_INL_H_
#include "string-inl.h"
#undef STRING_INL_H_
