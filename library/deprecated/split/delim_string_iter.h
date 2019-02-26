#pragma once

#include <util/generic/algorithm.h>
#include <util/generic/strbuf.h>
#include <util/generic/yexception.h>
#include <util/string/cast.h>
#include <util/system/yassert.h>

#include <iterator>

class TDelimStringIter {
public:
    using value_type = TStringBuf;
    using difference_type = ptrdiff_t;
    using pointer = const TStringBuf*;
    using reference = const TStringBuf&;
    using iterator_category = std::forward_iterator_tag;

    inline TDelimStringIter(const char* begin, const char* strEnd, TStringBuf delim)
        : TDelimStringIter(TStringBuf(begin, strEnd), delim)
    {
    }

    inline TDelimStringIter(TStringBuf str, TStringBuf delim)
        : IsValid(true)
        , Str(str)
        , Delim(delim)
    {
        UpdateCurrent();
    }

    inline TDelimStringIter()
        : IsValid(false)
    {
    }

    inline explicit operator bool() const {
        return IsValid;
    }

    // NOTE: this is a potentially unsafe operation (no overrun check)
    inline TDelimStringIter& operator++() {
        if (Current.end() != Str.end()) {
            Str.Skip(Current.length() + Delim.length());
            UpdateCurrent();
        } else {
            Str.Clear();
            Current.Clear();
            IsValid = false;
        }
        return *this;
    }

    inline void operator+=(size_t n) {
        for (; n > 0; --n) {
            ++(*this);
        }
    }

    inline bool operator==(const TDelimStringIter& rhs) const {
        return (IsValid == rhs.IsValid) && (!IsValid || (Current.begin() == rhs.Current.begin()));
    }

    inline bool operator!=(const TDelimStringIter& rhs) const {
        return !(*this == rhs);
    }

    inline TStringBuf operator*() const {
        return Current;
    }

    inline const TStringBuf* operator->() const {
        return &Current;
    }

    // Get & advance
    template <class T>
    inline bool TryNext(T& t) {
        if (IsValid) {
            t = FromString<T>(Current);
            operator++();
            return true;
        } else {
            return false;
        }
    }

    template <class T>
    inline TDelimStringIter& Next(T& t) // Get & advance
    {
        if (!TryNext(t))
            ythrow yexception() << "No valid field";
        return *this;
    }

    template <class T>
    inline T GetNext() {
        T res;
        Next(res);
        return res;
    }

    inline const char* GetBegin() const {
        return Current.begin();
    }

    inline const char* GetEnd() const {
        return Current.end();
    }

    inline bool Valid() const {
        return IsValid;
    }

    // contents from next token to the end of string
    inline TStringBuf Cdr() const {
        return Str.SubStr(Current.length() + Delim.length());
    }

    inline TDelimStringIter IterEnd() const {
        return TDelimStringIter();
    }

private:
    inline void UpdateCurrent() {
        // it is much faster than TStringBuf::find
        size_t pos = std::search(Str.begin(), Str.end(), Delim.begin(), Delim.end()) - Str.begin();
        Current = Str.Head(pos);
    }

private:
    bool IsValid;

    TStringBuf Str;
    TStringBuf Current;
    TStringBuf Delim;
};

//example: for (TStringBuf field: TDelimStroka(line, "@@")) { ... }
struct TDelimStroka {
    TStringBuf S;
    TStringBuf Delim;

    inline TDelimStroka(TStringBuf s, TStringBuf delim)
        : S(s)
        , Delim(delim)
    {
    }

    inline TDelimStringIter begin() const {
        return TDelimStringIter(S, Delim);
    }

    inline TDelimStringIter end() const {
        return TDelimStringIter();
    }
};

inline TDelimStringIter begin_delim(const TString& str, TStringBuf delim) {
    return TDelimStringIter(str, delim);
}

inline TDelimStringIter begin_delim(TStringBuf str, TStringBuf delim) {
    return TDelimStringIter(str.begin(), str.end(), delim);
}

inline TDelimStringIter end_delim(const TString& /*str*/, TStringBuf /*delim*/) {
    return TDelimStringIter();
}

class TKeyValueDelimStringIter {
public:
    TKeyValueDelimStringIter(const TStringBuf str, const TStringBuf delim);
    bool Valid() const;
    TKeyValueDelimStringIter& operator++();
    const TStringBuf& Key() const;
    const TStringBuf& Value() const;

private:
    TDelimStringIter DelimIter;
    TStringBuf ChunkKey, ChunkValue;

private:
    void ReadKeyAndValue();
};
