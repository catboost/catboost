#pragma once

#include <library/cpp/deprecated/kmp/kmp.h>
#include <util/string/cast.h>
#include <util/string/util.h>
#include <util/string/builder.h>

#include <util/system/yassert.h>
#include <util/system/defaults.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

#include <cstdio>

template <typename T>
struct TNumPair {
    T Begin;
    T End;

    TNumPair() = default;

    TNumPair(T begin, T end)
        : Begin(begin)
        , End(end)
    {
        Y_ASSERT(begin <= end);
    }

    T Length() const {
        return End - Begin + 1;
    }

    bool operator==(const TNumPair& r) const {
        return (Begin == r.Begin) && (End == r.End);
    }

    bool operator!=(const TNumPair& r) const {
        return (Begin != r.Begin) || (End != r.End);
    }
};

using TSizeTRegion = TNumPair<size_t>;
using TUi32Region = TNumPair<ui32>;

template <>
inline TString ToString(const TUi32Region& r) {
    return TStringBuilder() << "(" << r.Begin << ", " << r.End << ")";
}

template <>
inline TUi32Region FromString(const TString& s) {
    TUi32Region result;
    sscanf(s.data(), "(%" PRIu32 ", %" PRIu32 ")", &result.Begin, &result.End);
    return result;
}

class TSplitDelimiters {
private:
    bool Delims[256];

public:
    explicit TSplitDelimiters(const char* s);

    Y_FORCE_INLINE bool IsDelimiter(ui8 ch) const {
        return Delims[ch];
    }
};

template <class Split>
class TSplitIterator;

class TSplitBase {
protected:
    const char* Str;
    size_t Len;

public:
    TSplitBase(const char* str, size_t length);
    TSplitBase(const TString& s);

    Y_FORCE_INLINE const char* GetString() const {
        return Str;
    }

    Y_FORCE_INLINE size_t GetLength() const {
        return Len;
    }

private:
    // we don't own Str, make sure that no one calls us with temporary object
    TSplitBase(TString&&) = delete;
};

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4512)
#endif

class TDelimitersSplit: public TSplitBase {
private:
    const TSplitDelimiters& Delimiters;

public:
    using TIterator = TSplitIterator<TDelimitersSplit>;
    friend class TSplitIterator<TDelimitersSplit>;

    TDelimitersSplit(const char* str, size_t length, const TSplitDelimiters& delimiters);
    TDelimitersSplit(const TString& s, const TSplitDelimiters& delimiters);
    TIterator Iterator() const;
    TSizeTRegion Next(size_t& pos) const;
    size_t Begin() const;

private:
    // we don't own Delimiters, make sure that no one calls us with temporary object
    TDelimitersSplit(const char*, size_t, TSplitDelimiters&&) = delete;
    TDelimitersSplit(const TString&, TSplitDelimiters&&) = delete;
    TDelimitersSplit(TString&&, const TSplitDelimiters&) = delete;
};

class TDelimitersStrictSplit: public TSplitBase {
private:
    const TSplitDelimiters& Delimiters;

public:
    using TIterator = TSplitIterator<TDelimitersStrictSplit>;
    friend class TSplitIterator<TDelimitersStrictSplit>;

    TDelimitersStrictSplit(const char* str, size_t length, const TSplitDelimiters& delimiters);
    TDelimitersStrictSplit(const TString& s, const TSplitDelimiters& delimiters);
    TIterator Iterator() const;
    TSizeTRegion Next(size_t& pos) const;
    size_t Begin() const;

private:
    // we don't own Delimiters, make sure that no one calls us with temporary object
    TDelimitersStrictSplit(const char*, size_t, TSplitDelimiters&&) = delete;
    TDelimitersStrictSplit(const TString&, TSplitDelimiters&&) = delete;
    TDelimitersStrictSplit(TString&&, const TSplitDelimiters&) = delete;
};

class TScreenedDelimitersSplit: public TSplitBase {
private:
    const TSplitDelimiters& Delimiters;
    const TSplitDelimiters& Screens;

public:
    using TIterator = TSplitIterator<TScreenedDelimitersSplit>;
    friend class TSplitIterator<TScreenedDelimitersSplit>;

    TScreenedDelimitersSplit(const char*, size_t, const TSplitDelimiters& delimiters, const TSplitDelimiters& screens);
    TScreenedDelimitersSplit(const TString& s, const TSplitDelimiters& delimiters, const TSplitDelimiters& screens);
    TIterator Iterator() const;
    TSizeTRegion Next(size_t& pos) const;
    size_t Begin() const;

private:
    // we don't own Delimiters and Screens, make sure that no one calls us with temporary object
    TScreenedDelimitersSplit(TString&&, const TSplitDelimiters&, const TSplitDelimiters&) = delete;
    TScreenedDelimitersSplit(const TString&, TSplitDelimiters&&, const TSplitDelimiters&) = delete;
    TScreenedDelimitersSplit(const TString&, const TSplitDelimiters&, TSplitDelimiters&&) = delete;
};

class TDelimitersSplitWithoutTags: public TSplitBase {
private:
    const TSplitDelimiters& Delimiters;
    size_t SkipTag(size_t pos) const;
    size_t SkipDelimiters(size_t pos) const;

public:
    using TIterator = TSplitIterator<TDelimitersSplitWithoutTags>;
    friend class TSplitIterator<TDelimitersSplitWithoutTags>;

    TDelimitersSplitWithoutTags(const char* str, size_t length, const TSplitDelimiters& delimiters);
    TDelimitersSplitWithoutTags(const TString& s, const TSplitDelimiters& delimiters);
    TIterator Iterator() const;
    TSizeTRegion Next(size_t& pos) const;
    size_t Begin() const;

private:
    // we don't own Delimiters, make sure that no one calls us with temporary object
    TDelimitersSplitWithoutTags(const char*, size_t, TSplitDelimiters&&) = delete;
    TDelimitersSplitWithoutTags(const TString&, TSplitDelimiters&&) = delete;
    TDelimitersSplitWithoutTags(TString&&, const TSplitDelimiters&) = delete;
};

class TCharSplit: public TSplitBase {
public:
    using TIterator = TSplitIterator<TCharSplit>;
    friend class TSplitIterator<TCharSplit>;

    TCharSplit(const char* str, size_t length);
    TCharSplit(const TString& s);
    TIterator Iterator() const;
    TSizeTRegion Next(size_t& pos) const;
    size_t Begin() const;

private:
    // we don't own Str, make sure that no one calls us with temporary object
    TCharSplit(TString&&) = delete;
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

class TCharSplitWithoutTags: public TSplitBase {
private:
    size_t SkipTag(size_t pos) const;
    size_t SkipDelimiters(size_t pos) const;

public:
    using TIterator = TSplitIterator<TCharSplitWithoutTags>;
    friend class TSplitIterator<TCharSplitWithoutTags>;

    TCharSplitWithoutTags(const char* str, size_t length);
    TCharSplitWithoutTags(const TString& s);
    TIterator Iterator() const;
    TSizeTRegion Next(size_t& pos) const;
    size_t Begin() const;

private:
    // we don't own Str, make sure that no one calls us with temporary object
    TCharSplitWithoutTags(TString&&) = delete;
};

class TSubstringSplitDelimiter {
public:
    TKMPMatcher Matcher;
    size_t Len;

    TSubstringSplitDelimiter(const TString& s);
};

class TSubstringSplit: public TSplitBase {
private:
    const TSubstringSplitDelimiter& Delimiter;

public:
    using TIterator = TSplitIterator<TSubstringSplit>;
    friend class TSplitIterator<TSubstringSplit>;

    TSubstringSplit(const char* str, size_t length, const TSubstringSplitDelimiter& delimiter);
    TSubstringSplit(const TString& str, const TSubstringSplitDelimiter& delimiter);
    TIterator Iterator() const;
    TSizeTRegion Next(size_t& pos) const;
    size_t Begin() const;

private:
    // we don't own Delimiters, make sure that no one calls us with temporary object
    TSubstringSplit(TString&&, const TSubstringSplitDelimiter&) = delete;
    TSubstringSplit(const TString&, TSubstringSplitDelimiter&&) = delete;
};

template <class TSplit>
class TSplitIterator {
protected:
    const TSplit& Split;
    size_t Pos;
    TString* CurrentStroka;

public:
    TSplitIterator(const TSplit& split)
        : Split(split)
        , Pos(Split.Begin())
        , CurrentStroka(nullptr)
    {
    }

    virtual ~TSplitIterator() {
        delete CurrentStroka;
    }

    inline TSizeTRegion Next() {
        Y_ENSURE(!Eof(), TStringBuf("eof reached"));
        return Split.Next(Pos);
    }

    TStringBuf NextTok() {
        if (Eof())
            return TStringBuf();
        TSizeTRegion region = Next();
        return TStringBuf(Split.Str + region.Begin, region.End - region.Begin);
    }

    const TString& NextString() {
        if (!CurrentStroka)
            CurrentStroka = new TString();
        TSizeTRegion region = Next();
        CurrentStroka->assign(Split.Str, region.Begin, region.Length() - 1);
        return *CurrentStroka;
    }

    inline bool Eof() const {
        return Pos >= Split.Len;
    }

    TString GetTail() const {
        return TString(Split.Str + Pos);
    }

    void Skip(size_t count) {
        for (size_t i = 0; i < count; ++i)
            Next();
    }
};

using TSplitTokens = TVector<TString>;

template <typename TSplit>
void Split(const TSplit& split, TSplitTokens* words) {
    words->clear();
    TSplitIterator<TSplit> it(split);
    while (!it.Eof())
        words->push_back(it.NextString());
}
