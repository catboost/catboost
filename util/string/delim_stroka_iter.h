#pragma once

#include <util/generic/algorithm.h>
#include <util/generic/strbuf.h>
#include <util/generic/yexception.h>
#include <util/string/cast.h>
#include <util/system/yassert.h>

#include <iterator>

class TDelimStrokaIter {
public:
    using value_type = TStringBuf;
    using difference_type = ptrdiff_t;
    using pointer = const TStringBuf*;
    using reference = const TStringBuf&;
    using iterator_category = std::forward_iterator_tag;

    inline TDelimStrokaIter(const char* begin, const char* strEnd, TStringBuf delim)
        : TDelimStrokaIter(TStringBuf(begin, strEnd), delim)
    {
    }

    inline TDelimStrokaIter(TStringBuf str, TStringBuf delim)
        : IsValid(true)
        , Str(str)
        , Delim(delim)
    {
        UpdateCurrent();
    }

    inline TDelimStrokaIter()
        : IsValid(false)
    {
    }

    inline explicit operator bool() const {
        return IsValid;
    }

    // NOTE: this is a potentially unsafe operation (no overrun check)
    inline TDelimStrokaIter& operator++() {
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

    inline bool operator==(const TDelimStrokaIter& rhs) const {
        return (IsValid == rhs.IsValid) && (!IsValid || (Current.begin() == rhs.Current.begin()));
    }

    inline bool operator!=(const TDelimStrokaIter& rhs) const {
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
    inline TDelimStrokaIter& Next(T& t) // Get & advance
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

    inline TDelimStrokaIter IterEnd() const {
        return TDelimStrokaIter();
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

    inline TDelimStrokaIter begin() const {
        return TDelimStrokaIter(S, Delim);
    }

    inline TDelimStrokaIter end() const {
        return TDelimStrokaIter();
    }
};

inline TDelimStrokaIter begin_delim(const TString& str, TStringBuf delim) {
    return TDelimStrokaIter(str, delim);
}

inline TDelimStrokaIter begin_delim(TStringBuf str, TStringBuf delim) {
    return TDelimStrokaIter(str.begin(), str.end(), delim);
}

inline TDelimStrokaIter end_delim(const TString& /*str*/, TStringBuf /*delim*/) {
    return TDelimStrokaIter();
}

class TKeyValueDelimStrokaIter {
public:
    TKeyValueDelimStrokaIter(const TStringBuf str, const TStringBuf delim);
    bool Valid() const;
    TKeyValueDelimStrokaIter& operator++();
    const TStringBuf& Key() const;
    const TStringBuf& Value() const;

private:
    TDelimStrokaIter DelimIter;
    TStringBuf ChunkKey, ChunkValue;

private:
    void ReadKeyAndValue();
};
