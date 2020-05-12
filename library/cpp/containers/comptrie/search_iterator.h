#pragma once

#include "comptrie_trie.h"
#include "first_symbol_iterator.h"

#include <util/str_stl.h>
#include <util/digest/numeric.h>
#include <util/digest/multi.h>

// Iterator for incremental searching.
// All Advance() methods shift the iterator using specifed key/char.
// The subsequent Advance() call starts searching from the previous state.
// The Advance() returns 'true' if specified key part exists in the trie and
// returns 'false' for unsuccessful search. In case of 'false' result
// all subsequent calls also will return 'false'.
// If current iterator state is final then GetValue() method returns 'true' and
// associated value.

template <class TTrie>
class TSearchIterator {
public:
    using TData = typename TTrie::TData;
    using TSymbol = typename TTrie::TSymbol;
    using TKeyBuf = typename TTrie::TKeyBuf;

    TSearchIterator() = default;

    explicit TSearchIterator(const TTrie& trie)
        : Trie(&trie)
        , DataPos(trie.DataHolder.AsCharPtr())
        , DataEnd(DataPos + trie.DataHolder.Length())
        , ValuePos(trie.EmptyValue)
    {
    }

    explicit TSearchIterator(const TTrie& trie, const TTrie& subTrie)
        : Trie(&trie)
        , DataPos(subTrie.Data().AsCharPtr())
        , DataEnd(trie.DataHolder.AsCharPtr() + trie.DataHolder.Length())
        , ValuePos(subTrie.EmptyValue)
    {
    }

    bool operator==(const TSearchIterator& other) const {
        Y_ASSERT(Trie && other.Trie);
        return Trie == other.Trie &&
            DataPos == other.DataPos &&
            DataEnd == other.DataEnd &&
            ValuePos == other.ValuePos;
    }
    bool operator!=(const TSearchIterator& other) const {
        return !(*this == other);
    }

    inline bool Advance(TSymbol label) {
        Y_ASSERT(Trie);
        if (DataPos == nullptr || DataPos >= DataEnd) {
            return false;
        }
        return NCompactTrie::Advance(DataPos, DataEnd, ValuePos, label, Trie->Packer);
    }
    inline bool Advance(const TKeyBuf& key) {
        return Advance(key.data(), key.size());
    }
    bool Advance(const TSymbol* key, size_t keylen);
    bool GetValue(TData* value = nullptr) const;
    bool HasValue() const;
    inline size_t GetHash() const;

private:
    const TTrie* Trie = nullptr;
    const char* DataPos = nullptr;
    const char* DataEnd = nullptr;
    const char* ValuePos = nullptr;
};

template <class TTrie>
inline TSearchIterator<TTrie> MakeSearchIterator(const TTrie& trie) {
    return TSearchIterator<TTrie>(trie);
}

template <class TTrie>
struct THash<TSearchIterator<TTrie>> {
    inline size_t operator()(const TSearchIterator<TTrie>& item) {
        return item.GetHash();
    }
};

//----------------------------------------------------------------------------

template <class TTrie>
bool TSearchIterator<TTrie>::Advance(const TSymbol* key, size_t keylen) {
    Y_ASSERT(Trie);
    if (!key || DataPos == nullptr || DataPos >= DataEnd) {
        return false;
    }
    if (!keylen) {
        return true;
    }

    const TSymbol* keyend = key + keylen;
    while (key != keyend && DataPos != nullptr) {
        if (!NCompactTrie::Advance(DataPos, DataEnd, ValuePos, *(key++), Trie->Packer)) {
            return false;
        }
        if (key == keyend) {
            return true;
        }
    }
    return false;
}

template <class TTrie>
bool TSearchIterator<TTrie>::GetValue(TData* value) const {
    Y_ASSERT(Trie);
    bool result = false;
    if (value) {
        if (ValuePos) {
            result = true;
            Trie->Packer.UnpackLeaf(ValuePos, *value);
        }
    }
    return result;
}

template <class TTrie>
bool TSearchIterator<TTrie>::HasValue() const {
    Y_ASSERT(Trie);
    return ValuePos;
}

template <class TTrie>
inline size_t TSearchIterator<TTrie>::GetHash() const {
    Y_ASSERT(Trie);
    return MultiHash(
        static_cast<const void*>(Trie),
        static_cast<const void*>(DataPos),
        static_cast<const void*>(DataEnd),
        static_cast<const void*>(ValuePos));
}
