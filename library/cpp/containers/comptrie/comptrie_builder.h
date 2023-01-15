#pragma once

#include "comptrie_packer.h"
#include "minimize.h"
#include "key_selector.h"

#include <util/stream/file.h>

// --------------------------------------------------------------------------------------
// Data Builder
// To build the data buffer, we first create an automaton in memory. The automaton
// is created incrementally. It actually helps a lot to have the input data prefix-grouped
// by key; otherwise, memory consumption becomes a tough issue.
// NOTE: building and serializing the automaton may be lengthy, and takes lots of memory.

// PREFIX_GROUPED means that if we, while constructing a trie, add to the builder two keys with the same prefix,
// then all the keys that we add between these two also have the same prefix.
// Actually in this mode the builder can accept even more freely ordered input,
// but for input as above it is guaranteed to work.
enum ECompactTrieBuilderFlags {
    CTBF_NONE = 0,
    CTBF_PREFIX_GROUPED = 1 << 0,
    CTBF_VERBOSE = 1 << 1,
    CTBF_UNIQUE = 1 << 2,
};

using TCompactTrieBuilderFlags = ECompactTrieBuilderFlags;

inline TCompactTrieBuilderFlags operator|(TCompactTrieBuilderFlags first, TCompactTrieBuilderFlags second) {
    return static_cast<TCompactTrieBuilderFlags>(static_cast<int>(first) | second);
}

inline TCompactTrieBuilderFlags& operator|=(TCompactTrieBuilderFlags& first, TCompactTrieBuilderFlags second) {
    return first = first | second;
}

template <typename T>
class TArrayWithSizeHolder;

template <class T = char, class D = ui64, class S = TCompactTriePacker<D>>
class TCompactTrieBuilder {
public:
    typedef T TSymbol;
    typedef D TData;
    typedef S TPacker;
    typedef typename TCompactTrieKeySelector<TSymbol>::TKey TKey;
    typedef typename TCompactTrieKeySelector<TSymbol>::TKeyBuf TKeyBuf;

    explicit TCompactTrieBuilder(TCompactTrieBuilderFlags flags = CTBF_NONE, TPacker packer = TPacker(), IAllocator* alloc = TDefaultAllocator::Instance());

    // All Add.. methods return true if it was a new key, false if the key already existed.

    bool Add(const TSymbol* key, size_t keylen, const TData& value);
    bool Add(const TKeyBuf& key, const TData& value) {
        return Add(key.data(), key.size(), value);
    }

    // add already serialized data
    bool AddPtr(const TSymbol* key, size_t keylen, const char* data);
    bool AddPtr(const TKeyBuf& key, const char* data) {
        return AddPtr(key.data(), key.size(), data);
    }

    bool AddSubtreeInFile(const TSymbol* key, size_t keylen, const TString& filename);
    bool AddSubtreeInFile(const TKeyBuf& key, const TString& filename) {
        return AddSubtreeInFile(key.data(), key.size(), filename);
    }

    bool AddSubtreeInBuffer(const TSymbol* key, size_t keylen, TArrayWithSizeHolder<char>&& buffer);
    bool AddSubtreeInBuffer(const TKeyBuf& key, TArrayWithSizeHolder<char>&& buffer) {
        return AddSubtreeInBuffer(key.data(), key.size(), std::move(buffer));
    }

    bool Find(const TSymbol* key, size_t keylen, TData* value) const;
    bool Find(const TKeyBuf& key, TData* value = nullptr) const {
        return Find(key.data(), key.size(), value);
    }

    bool FindLongestPrefix(const TSymbol* key, size_t keylen, size_t* prefixLen, TData* value = nullptr) const;
    bool FindLongestPrefix(const TKeyBuf& key, size_t* prefixLen, TData* value = nullptr) const {
        return FindLongestPrefix(key.data(), key.size(), prefixLen, value);
    }

    size_t Save(IOutputStream& os) const;
    size_t SaveAndDestroy(IOutputStream& os);
    size_t SaveToFile(const TString& fileName) const {
        TFixedBufferFileOutput out(fileName);
        return Save(out);
    }

    void Clear(); // Returns all memory to the system and resets the builder state.

    size_t GetEntryCount() const;
    size_t GetNodeCount() const;

    // Exact output file size in bytes.
    size_t MeasureByteSize() const {
        return Impl->MeasureByteSize();
    }

protected:
    class TCompactTrieBuilderImpl;
    THolder<TCompactTrieBuilderImpl> Impl;
};

//----------------------------------------------------------------------------------------------------------------------
// Minimize the trie. The result is equivalent to the original
// trie, except that it takes less space (and has marginally lower
// performance, because of eventual epsilon links).
// The algorithm is as follows: starting from the largest pieces, we find
// nodes that have identical continuations  (Daciuk's right language),
// and repack the trie. Repacking is done in-place, so memory is less
// of an issue; however, it may take considerable time.

// IMPORTANT: never try to reminimize an already minimized trie or a trie with fast layout.
// Because of non-local structure and epsilon links, it won't work
// as you expect it to, and can destroy the trie in the making.
// If you want both minimization and fast layout, do the minimization first.

template <class TPacker>
size_t CompactTrieMinimize(IOutputStream& os, const char* data, size_t datalength, bool verbose = false, const TPacker& packer = TPacker(), NCompactTrie::EMinimizeMode mode = NCompactTrie::MM_DEFAULT);

template <class TTrieBuilder>
size_t CompactTrieMinimize(IOutputStream& os, const TTrieBuilder& builder, bool verbose = false);

//----------------------------------------------------------------------------------------------------------------
// Lay the trie in memory in such a way that there are less cache misses when jumping from root to leaf.
// The trie becomes about 2% larger, but the access became about 25% faster in our experiments.
// Can be called on minimized and non-minimized tries, in the first case in requires half a trie more memory.
// Calling it the second time on the same trie does nothing.
//
// The algorithm is based on van Emde Boas layout as described in the yandex data school lectures on external memory algoritms
// by Maxim Babenko and Ivan Puzyrevsky. The difference is that when we cut the tree into levels
// two nodes connected by a forward link are put into the same level (because they usually lie near each other in the original tree).
// The original paper (describing the layout in Section 2.1) is:
// Michael A. Bender, Erik D. Demaine, Martin Farach-Colton. Cache-Oblivious B-Trees
//      SIAM Journal on Computing, volume 35, number 2, 2005, pages 341-358.
// Available on the web: http://erikdemaine.org/papers/CacheObliviousBTrees_SICOMP/
// Or: Michael A. Bender, Erik D. Demaine, and Martin Farach-Colton. Cache-Oblivious B-Trees
//      Proceedings of the 41st Annual Symposium
//      on Foundations of Computer Science (FOCS 2000), Redondo Beach, California, November 12-14, 2000, pages 399-409.
// Available on the web: http://erikdemaine.org/papers/FOCS2000b/
// (there is not much difference between these papers, actually).
//
template <class TPacker>
size_t CompactTrieMakeFastLayout(IOutputStream& os, const char* data, size_t datalength, bool verbose = false, const TPacker& packer = TPacker());

template <class TTrieBuilder>
size_t CompactTrieMakeFastLayout(IOutputStream& os, const TTrieBuilder& builder, bool verbose = false);

// Composition of minimization and fast layout
template <class TPacker>
size_t CompactTrieMinimizeAndMakeFastLayout(IOutputStream& os, const char* data, size_t datalength, bool verbose = false, const TPacker& packer = TPacker());

template <class TTrieBuilder>
size_t CompactTrieMinimizeAndMakeFastLayout(IOutputStream& os, const TTrieBuilder& builder, bool verbose = false);

// Implementation details moved here.
#include "comptrie_builder.inl"
