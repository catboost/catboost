#include "diff.h"

#include <util/generic/hash.h>
#include <util/digest/fnv.h>

using NArrayRef::TConstArrayRef;

template <typename T>
struct TCollectionImpl {
    yvector<TConstArrayRef<T>> Words;
    yvector<ui64> Keys;

    inline bool Consume(const T* b, const T* e, const T*) {
        if (b < e) {
            Words.push_back(TConstArrayRef<T>(b, e));
            Keys.push_back(FnvHash<ui64>((const char*)b, (e - b) * sizeof(T)));
        }
        return true;
    }

    TConstArrayRef<T> Remap(const TConstArrayRef<ui64>& keys) {
        if (keys.empty()) {
            return TConstArrayRef<T>();
        }
        Y_ASSERT(keys.begin() >= Keys.begin() && keys.begin() <= Keys.end());
        Y_ASSERT(keys.end() >= Keys.begin() && keys.end() <= Keys.end());
        return TConstArrayRef<T>(Words[keys.begin() - Keys.begin()].begin(), Words[keys.end() - Keys.begin() - 1].end());
    }

    TConstArrayRef<ui64> GetKeys() const {
        return TConstArrayRef<ui64>(Keys.begin(), Keys.end());
    }
};

template <typename T>
struct TCollection {
};

template <>
struct TCollection<char>: public TCollectionImpl<char> {
    TCollection(const TStringBuf& str, const TString& delims)
    {
        TSetDelimiter<const char> set(~delims);
        TKeepDelimiters<TCollection<char>> c(this);
        SplitString(str.begin(), str.end(), set, c);
    }
};

template <>
struct TCollection<TChar>: public TCollectionImpl<TChar> {
    TCollection(const TWtringBuf& str, const TUtf16String& delims)
    {
        TSetDelimiter<const TChar> set(~delims);
        TKeepDelimiters<TCollection<TChar>> c(this);
        SplitString(str.begin(), str.end(), set, c);
    }
};

void NDiff::InlineDiff(yvector<TChunk<char>>& chunks, const TStringBuf& left, const TStringBuf& right, const TString& delims) {
    if (delims.empty()) {
        InlineDiff<char>(chunks, TConstArrayRef<char>(~left, +left), TConstArrayRef<char>(~right, +right));
        return;
    }
    TCollection<char> c1(left, delims);
    TCollection<char> c2(right, delims);
    yvector<TChunk<ui64>> diff;
    InlineDiff<ui64>(diff, c1.GetKeys(), c2.GetKeys());
    for (const auto& it : diff) {
        chunks.push_back(TChunk<char>(c1.Remap(it.Left), c2.Remap(it.Right), c1.Remap(it.Common)));
    }
}

void NDiff::InlineDiff(yvector<TChunk<TChar>>& chunks, const TWtringBuf& left, const TWtringBuf& right, const TUtf16String& delims) {
    if (delims.empty()) {
        InlineDiff<TChar>(chunks, TConstArrayRef<TChar>(~left, +left), TConstArrayRef<TChar>(~right, +right));
        return;
    }
    TCollection<TChar> c1(left, delims);
    TCollection<TChar> c2(right, delims);
    yvector<TChunk<ui64>> diff;
    InlineDiff<ui64>(diff, c1.GetKeys(), c2.GetKeys());
    for (const auto& it : diff) {
        chunks.push_back(TChunk<TChar>(c1.Remap(it.Left), c2.Remap(it.Right), c1.Remap(it.Common)));
    }
}
