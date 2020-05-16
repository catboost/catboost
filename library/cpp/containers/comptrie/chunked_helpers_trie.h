#pragma once

#include <library/cpp/on_disk/chunks/chunked_helpers.h>

#include "comptrie.h"

class TTrieSet {
private:
    TCompactTrie<char> Trie;

public:
    TTrieSet(const TBlob& blob)
        : Trie(blob)
    {
    }

    bool Has(const char* key) const {
        return Trie.Find(key, strlen(key));
    }

    bool FindLongestPrefix(const char* key, size_t keylen, size_t* prefixLen) {
        return Trie.FindLongestPrefix(key, keylen, prefixLen);
    }
};

template <bool sorted = false>
class TTrieSetWriter {
private:
    TCompactTrieBuilder<char> Builder;

public:
    TTrieSetWriter(bool isSorted = sorted)
        : Builder(isSorted ? CTBF_PREFIX_GROUPED : CTBF_NONE)
    {
    }

    void Add(const char* key, size_t keylen) {
        Builder.Add(key, keylen, 0);
        assert(Has(((TString)key).substr(0, keylen).data()));
    }

    void Add(const char* key) {
        Add(key, strlen(key));
    }

    bool Has(const char* key) const {
        ui64 dummy;
        return Builder.Find(key, strlen(key), &dummy);
    }

    void Save(IOutputStream& out) const {
        Builder.Save(out);
    }

    void Clear() {
        Builder.Clear();
    }
};

template <bool isWriter, bool sorted = false>
struct TTrieSetG;

template <bool sorted>
struct TTrieSetG<false, sorted> {
    typedef TTrieSet T;
};

template <bool sorted>
struct TTrieSetG<true, sorted> {
    typedef TTrieSetWriter<sorted> T;
};

template <typename T>
class TTrieMap {
private:
    TCompactTrie<char> Trie;
    static_assert(sizeof(T) <= sizeof(ui64), "expect sizeof(T) <= sizeof(ui64)");

public:
    TTrieMap(const TBlob& blob)
        : Trie(blob)
    {
    }

    bool Get(const char* key, T* value) const {
        ui64 trieValue;
        if (Trie.Find(key, strlen(key), &trieValue)) {
            *value = ReadUnaligned<T>(&trieValue);
            return true;
        } else {
            return false;
        }
    }

    T Get(const char* key, T def = T()) const {
        ui64 trieValue;
        if (Trie.Find(key, strlen(key), &trieValue)) {
            return ReadUnaligned<T>(&trieValue);
        } else {
            return def;
        }
    }

    const TCompactTrie<char>& GetTrie() const {
        return Trie;
    }
};

template <typename T, bool sorted = false>
class TTrieMapWriter {
private:
    typedef TCompactTrieBuilder<char> TBuilder;
    TBuilder Builder;
    static_assert(sizeof(T) <= sizeof(ui64), "expect sizeof(T) <= sizeof(ui64)");
#ifndef NDEBUG
    bool IsSorted;
#endif

public:
    TTrieMapWriter(bool isSorted = sorted)
        : Builder(isSorted ? CTBF_PREFIX_GROUPED : CTBF_NONE)
#ifndef NDEBUG
        , IsSorted(isSorted)
#endif
    {
    }

    void Add(const char* key, const T& value) {
        ui64 intValue = 0;
        memcpy(&intValue, &value, sizeof(T));
        Builder.Add(key, strlen(key), intValue);
#ifndef NDEBUG
        /*
        if (!IsSorted) {
            T test;
            assert(Get(key, &test) && value == test);
        }
        */
#endif
    }

    void Add(const TString& s, const T& value) {
        ui64 intValue = 0;
        memcpy(&intValue, &value, sizeof(T));
        Builder.Add(s.data(), s.size(), intValue);
    }

    bool Get(const char* key, T* value) const {
        ui64 trieValue;
        if (Builder.Find(key, strlen(key), &trieValue)) {
            *value = ReadUnaligned<T>(&trieValue);
            return true;
        } else {
            return false;
        }
    }

    T Get(const char* key, T def = (T)0) const {
        ui64 trieValue;
        if (Builder.Find(key, strlen(key), &trieValue)) {
            return ReadUnaligned<T>(&trieValue);
        } else {
            return def;
        }
    }

    void Save(IOutputStream& out, bool minimize = false) const {
        if (minimize) {
            CompactTrieMinimize<TBuilder>(out, Builder, false);
        } else {
            Builder.Save(out);
        }
    }

    void Clear() {
        Builder.Clear();
    }
};

template <typename T>
class TTrieSortedMapWriter {
private:
    typedef std::pair<TString, T> TValue;
    typedef TVector<TValue> TValues;
    TValues Values;

public:
    TTrieSortedMapWriter() = default;

    void Add(const char* key, const T& value) {
        Values.push_back(TValue(key, value));
    }

    void Save(IOutputStream& out) {
        Sort(Values.begin(), Values.end());
        TTrieMapWriter<T, true> writer;
        for (typename TValues::const_iterator toValue = Values.begin(); toValue != Values.end(); ++toValue)
            writer.Add(toValue->first.data(), toValue->second);
        writer.Save(out);
    }

    void Clear() {
        Values.clear();
    }
};

template <typename X, bool isWriter, bool sorted = false>
struct TTrieMapG;

template <typename X, bool sorted>
struct TTrieMapG<X, false, sorted> {
    typedef TTrieMap<X> T;
};

template <typename X, bool sorted>
struct TTrieMapG<X, true, sorted> {
    typedef TTrieMapWriter<X, sorted> T;
};
