#ifndef STRING_INL_H_
#error "Direct inclusion of this file is not allowed, include string.h"
// For the sake of sane code completion.
#include "string.h"
#endif

#include "string_builder.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

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
    TStringBuf delimiter)
{
    for (auto current = begin; current != end; ++current) {
        if (current != begin) {
            builder->AppendString(delimiter);
        }
        formatter(builder, *current);
    }
}

template <std::forward_iterator TIterator, class TFormatter>
TString JoinToString(
    const TIterator& begin,
    const TIterator& end,
    const TFormatter& formatter,
    TStringBuf delimiter)
{
    TStringBuilder builder;
    JoinToString(&builder, begin, end, formatter, delimiter);
    return builder.Flush();
}

//! A handy shortcut with default formatter.
template <std::forward_iterator TIterator>
TString JoinToString(
    const TIterator& begin,
    const TIterator& end,
    TStringBuf delimiter)
{
    return JoinToString(begin, end, TDefaultFormatter(), delimiter);
}

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
    TStringBuf delimiter)
{
    using std::begin;
    using std::end;
    return JoinToString(begin(collection), end(collection), formatter, delimiter);
}

//! A handy shortcut with the default formatter.
template <std::ranges::range TCollection>
TString JoinToString(
    TCollection&& collection,
    TStringBuf delimiter)
{
    return JoinToString(std::forward<TCollection>(collection), TDefaultFormatter(), delimiter);
}

//! Concatenates a bunch of TStringBuf-like instances into TString.
template <class... Ts>
TString ConcatToString(Ts... args)
{
    size_t length = 0;
    ((length += args.length()), ...);

    TString result;
    result.reserve(length);
    (result.append(args), ...);

    return result;
}

//! Converts a range of items into strings.
template <std::forward_iterator TIter, class TFormatter>
std::vector<TString> ConvertToStrings(
    const TIter& begin,
    const TIter& end,
    const TFormatter& formatter,
    size_t maxSize)
{
    std::vector<TString> result;
    for (auto it = begin; it != end; ++it) {
        TStringBuilder builder;
        formatter(&builder, *it);
        result.push_back(builder.Flush());
        if (result.size() == maxSize) {
            break;
        }
    }
    return result;
}

//! A handy shortcut with the default formatter.
template <std::forward_iterator TIter>
std::vector<TString> ConvertToStrings(
    const TIter& begin,
    const TIter& end,
    size_t maxSize)
{
    return ConvertToStrings(begin, end, TDefaultFormatter(), maxSize);
}

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
    size_t maxSize)
{
    using std::begin;
    using std::end;
    return ConvertToStrings(begin(collection), end(collection), formatter, maxSize);
}

//! A handy shortcut with default formatter.
template <std::ranges::range TCollection>
std::vector<TString> ConvertToStrings(
    TCollection&& collection,
    size_t maxSize)
{
    return ConvertToStrings(std::forward<TCollection>(collection), TDefaultFormatter(), maxSize);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
