#include "diff.h"

#include <util/generic/hash.h>
#include <util/digest/fnv.h>

#include <iterator>

template <typename T>
struct TCollectionImpl {
    TVector<TConstArrayRef<T>> Words;
    TVector<ui64> Keys;

    inline bool Consume(const T* b, const T* e, const T*) {
        if (b < e) {
            Words.push_back(TConstArrayRef<T>(b, e));
            Keys.push_back(FnvHash<ui64>((const char*)b, (e - b) * sizeof(T)));
        }
        return true;
    }

    TConstArrayRef<T> Remap(const TConstArrayRef<ui64>& keys) const {
        if (keys.empty()) {
            return TConstArrayRef<T>();
        }
        auto firstWordPos = std::distance(Keys.data(), keys.begin());
        auto lastWordPos = std::distance(Keys.data(), keys.end()) - 1;
        Y_ASSERT(firstWordPos >= 0);
        Y_ASSERT(lastWordPos >= firstWordPos);
        Y_ASSERT(static_cast<size_t>(lastWordPos) < Words.size());

        return TConstArrayRef<T>(Words[firstWordPos].begin(), Words[lastWordPos].end());
    }

    TConstArrayRef<ui64> GetKeys() const {
        return TConstArrayRef<ui64>(Keys);
    }
};

template <typename T>
struct TCollection {
};

template <>
struct TCollection<char>: public TCollectionImpl<char> {
    TCollection(const TStringBuf& str, const TString& delims) {
        TSetDelimiter<const char> set(delims.data());
        TKeepDelimiters<TCollection<char>> c(this);
        SplitString(str.begin(), str.end(), set, c);
    }
};

template <>
struct TCollection<wchar16>: public TCollectionImpl<wchar16> {
    TCollection(const TWtringBuf& str, const TUtf16String& delims) {
        TSetDelimiter<const wchar16> set(delims.data());
        TKeepDelimiters<TCollection<wchar16>> c(this);
        SplitString(str.begin(), str.end(), set, c);
    }
};

size_t NDiff::InlineDiff(TVector<TChunk<char>>& chunks, const TStringBuf& left, const TStringBuf& right, const TString& delims) {
    if (delims.empty()) {
        return InlineDiff<char>(chunks, TConstArrayRef<char>(left.data(), left.size()), TConstArrayRef<char>(right.data(), right.size()));
    }
    TCollection<char> c1(left, delims);
    TCollection<char> c2(right, delims);
    TVector<TChunk<ui64>> diff;
    const size_t dist = InlineDiff<ui64>(diff, c1.GetKeys(), c2.GetKeys());
    for (const auto& it : diff) {
        chunks.push_back(TChunk<char>(c1.Remap(it.Left), c2.Remap(it.Right), c1.Remap(it.Common)));
    }
    return dist;
}

size_t NDiff::InlineDiff(TVector<TChunk<wchar16>>& chunks, const TWtringBuf& left, const TWtringBuf& right, const TUtf16String& delims) {
    if (delims.empty()) {
        return InlineDiff<wchar16>(chunks, TConstArrayRef<wchar16>(left.data(), left.size()), TConstArrayRef<wchar16>(right.data(), right.size()));
    }
    TCollection<wchar16> c1(left, delims);
    TCollection<wchar16> c2(right, delims);
    TVector<TChunk<ui64>> diff;
    const size_t dist = InlineDiff<ui64>(diff, c1.GetKeys(), c2.GetKeys());
    for (const auto& it : diff) {
        chunks.push_back(TChunk<wchar16>(c1.Remap(it.Left), c2.Remap(it.Right), c1.Remap(it.Common)));
    }
    return dist;
}
