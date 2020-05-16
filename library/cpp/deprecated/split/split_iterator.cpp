#include "split_iterator.h"

#include <util/system/yassert.h>

#include <cctype>
#include <cstring>
#include <cstdlib>

/****************** TSplitDelimiters2 ******************/

TSplitDelimiters::TSplitDelimiters(const char* s) {
    memset(Delims, 0, sizeof(Delims));
    while (*s)
        Delims[(ui8) * (s++)] = true;
}

/****************** TSplitBase ******************/
TSplitBase::TSplitBase(const char* str, size_t length)
    : Str(str)
    , Len(length)
{
}

TSplitBase::TSplitBase(const TString& s)
    : Str(s.data())
    , Len(s.size())
{
}

/****************** TDelimitersSplit ******************/

TDelimitersSplit::TDelimitersSplit(const char* str, size_t length, const TSplitDelimiters& delimiters)
    : TSplitBase(str, length)
    , Delimiters(delimiters)
{
}

TDelimitersSplit::TDelimitersSplit(const TString& s, const TSplitDelimiters& delimiters)
    : TSplitBase(s)
    , Delimiters(delimiters)
{
}

size_t TDelimitersSplit::Begin() const {
    size_t pos = 0;
    while ((pos < Len) && Delimiters.IsDelimiter(Str[pos]))
        ++pos;
    return pos;
}

TSizeTRegion TDelimitersSplit::Next(size_t& pos) const {
    size_t begin = pos;
    while ((pos < Len) && !Delimiters.IsDelimiter(Str[pos]))
        ++pos;
    TSizeTRegion result(begin, pos);

    while ((pos < Len) && Delimiters.IsDelimiter(Str[pos]))
        ++pos;

    return result;
}

TDelimitersSplit::TIterator TDelimitersSplit::Iterator() const {
    return TIterator(*this);
}

/****************** TDelimitersStrictSplit ******************/

TDelimitersStrictSplit::TDelimitersStrictSplit(const char* str, size_t length, const TSplitDelimiters& delimiters)
    : TSplitBase(str, length)
    , Delimiters(delimiters)
{
}

TDelimitersStrictSplit::TDelimitersStrictSplit(const TString& s, const TSplitDelimiters& delimiters)
    : TSplitBase(s)
    , Delimiters(delimiters)
{
}

TDelimitersStrictSplit::TIterator TDelimitersStrictSplit::Iterator() const {
    return TIterator(*this);
}

TSizeTRegion TDelimitersStrictSplit::Next(size_t& pos) const {
    size_t begin = pos;
    while ((pos < Len) && !Delimiters.IsDelimiter(Str[pos]))
        ++pos;
    TSizeTRegion result(begin, pos);

    if (pos < Len)
        ++pos;

    return result;
}

size_t TDelimitersStrictSplit::Begin() const {
    return 0;
}

/****************** TScreenedDelimitersSplit ******************/

TScreenedDelimitersSplit::TScreenedDelimitersSplit(const TString& s, const TSplitDelimiters& delimiters, const TSplitDelimiters& screens)
    : TSplitBase(s)
    , Delimiters(delimiters)
    , Screens(screens)
{
}

TScreenedDelimitersSplit::TScreenedDelimitersSplit(const char* str, size_t length, const TSplitDelimiters& delimiters, const TSplitDelimiters& screens)
    : TSplitBase(str, length)
    , Delimiters(delimiters)
    , Screens(screens)
{
}

TScreenedDelimitersSplit::TIterator TScreenedDelimitersSplit::Iterator() const {
    return TIterator(*this);
}

TSizeTRegion TScreenedDelimitersSplit::Next(size_t& pos) const {
    size_t begin = pos;
    bool screened = false;
    while (pos < Len) {
        if (Screens.IsDelimiter(Str[pos]))
            screened = !screened;
        if (Delimiters.IsDelimiter(Str[pos]) && !screened)
            break;
        ++pos;
    }
    TSizeTRegion result(begin, pos);

    if (pos < Len)
        ++pos;

    return result;
}

size_t TScreenedDelimitersSplit::Begin() const {
    return 0;
}

/****************** TDelimitersSplitWithoutTags ******************/

TDelimitersSplitWithoutTags::TDelimitersSplitWithoutTags(const char* str, size_t length, const TSplitDelimiters& delimiters)
    : TSplitBase(str, length)
    , Delimiters(delimiters)
{
}

TDelimitersSplitWithoutTags::TDelimitersSplitWithoutTags(const TString& s, const TSplitDelimiters& delimiters)
    : TSplitBase(s)
    , Delimiters(delimiters)
{
}

size_t TDelimitersSplitWithoutTags::SkipTag(size_t pos) const {
    Y_ASSERT('<' == Str[pos]);
    while ((pos < Len) && ('>' != Str[pos]))
        ++pos;
    return pos + 1;
}

size_t TDelimitersSplitWithoutTags::SkipDelimiters(size_t pos) const {
    while (true) {
        while ((pos < Len) && Delimiters.IsDelimiter(Str[pos]) && ('<' != Str[pos]))
            ++pos;
        if (pos < Len) {
            if ('<' != Str[pos])
                break;
            else
                pos = SkipTag(pos);
        } else
            break;
    }
    return pos;
}

size_t TDelimitersSplitWithoutTags::Begin() const {
    size_t pos = 0;
    pos = SkipDelimiters(pos);
    return pos;
}

TSizeTRegion TDelimitersSplitWithoutTags::Next(size_t& pos) const {
    size_t begin = pos;
    while ((pos < Len) && !Delimiters.IsDelimiter(Str[pos]) && ('<' != Str[pos]))
        ++pos;
    TSizeTRegion result(begin, pos);

    pos = SkipDelimiters(pos);

    return result;
}

TDelimitersSplitWithoutTags::TIterator TDelimitersSplitWithoutTags::Iterator() const {
    return TIterator(*this);
}

/****************** TCharSplit ******************/

TCharSplit::TCharSplit(const char* str, size_t length)
    : TSplitBase(str, length)
{
}

TCharSplit::TCharSplit(const TString& s)
    : TSplitBase(s)
{
}

TCharSplit::TIterator TCharSplit::Iterator() const {
    return TIterator(*this);
}

TSizeTRegion TCharSplit::Next(size_t& pos) const {
    TSizeTRegion result(pos, pos + 1);
    ++pos;
    return result;
}

size_t TCharSplit::Begin() const {
    return 0;
}

/****************** TCharSplitWithoutTags ******************/

TCharSplitWithoutTags::TCharSplitWithoutTags(const char* str, size_t length)
    : TSplitBase(str, length)
{
}

TCharSplitWithoutTags::TCharSplitWithoutTags(const TString& s)
    : TSplitBase(s)
{
}

size_t TCharSplitWithoutTags::SkipTag(size_t pos) const {
    Y_ASSERT('<' == Str[pos]);
    while ((pos < Len) && ('>' != Str[pos]))
        ++pos;
    return pos + 1;
}

size_t TCharSplitWithoutTags::SkipDelimiters(size_t pos) const {
    while (true) {
        if (pos < Len) {
            if ('<' != Str[pos])
                break;
            else
                pos = SkipTag(pos);
        } else
            break;
    }
    return pos;
}

size_t TCharSplitWithoutTags::Begin() const {
    size_t pos = 0;
    pos = SkipDelimiters(pos);
    return pos;
}

TSizeTRegion TCharSplitWithoutTags::Next(size_t& pos) const {
    size_t begin = pos++;
    TSizeTRegion result(begin, pos);

    pos = SkipDelimiters(pos);

    return result;
}

TCharSplitWithoutTags::TIterator TCharSplitWithoutTags::Iterator() const {
    return TIterator(*this);
}

TSubstringSplitDelimiter::TSubstringSplitDelimiter(const TString& s)
    : Matcher(s)
    , Len(s.size())
{
}

/****************** TSubstringSplit ******************/

TSubstringSplit::TSubstringSplit(const char* str, size_t length, const TSubstringSplitDelimiter& delimiter)
    : TSplitBase(str, length)
    , Delimiter(delimiter)
{
}

TSubstringSplit::TSubstringSplit(const TString& str, const TSubstringSplitDelimiter& delimiter)
    : TSplitBase(str)
    , Delimiter(delimiter)
{
}

TSubstringSplit::TIterator TSubstringSplit::Iterator() const {
    return TIterator(*this);
}

TSizeTRegion TSubstringSplit::Next(size_t& pos) const {
    const char* begin = Str + pos;
    const char* end = Str + Len;
    const char* delim;
    if (Delimiter.Matcher.SubStr(begin, end, delim)) {
        TSizeTRegion result(pos, delim - begin + pos);
        pos += delim - begin + Delimiter.Len;
        return result;
    } else {
        TSizeTRegion result(pos, end - begin + pos);
        pos += end - begin;
        return result;
    }
}

size_t TSubstringSplit::Begin() const {
    return 0;
}
