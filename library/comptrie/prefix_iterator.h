#pragma once

#include "comptrie_trie.h"

// Iterates over all prefixes of the given key in the trie.
template <class TTrie>
class TPrefixIterator {
public:
    using TSymbol = typename TTrie::TSymbol;
    using TPacker = typename TTrie::TPacker;
    using TData = typename TTrie::TData;

private:
    const TTrie& Trie;
    const TSymbol* key;
    size_t keylen;
    const TSymbol* keyend;
    size_t prefixLen;
    const char* valuepos;
    const char* datapos;
    const char* dataend;
    TPacker Packer;
    const char* EmptyValue;
    bool result;

    bool Next();

public:
    TPrefixIterator(const TTrie& trie, const TSymbol* aKey, size_t aKeylen)
        : Trie(trie)
        , key(aKey)
        , keylen(aKeylen)
        , keyend(aKey + aKeylen)
        , prefixLen(0)
        , valuepos(nullptr)
        , datapos(trie.DataHolder.AsCharPtr())
        , dataend(datapos + trie.DataHolder.Length())
    {
        result = Next();
    }

    operator bool() const {
        return result;
    }

    TPrefixIterator& operator++() {
        result = Next();
        return *this;
    }

    size_t GetPrefixLen() const {
        return prefixLen;
    }

    void GetValue(TData& to) const {
        Trie.Packer.UnpackLeaf(valuepos, to);
    }
};

template <class TTrie>
bool TPrefixIterator<TTrie>::Next() {
    using namespace NCompactTrie;
    if (!key || datapos == dataend)
        return false;

    if ((key == keyend - keylen) && !valuepos && Trie.EmptyValue) {
        valuepos = Trie.EmptyValue;
        return true;
    }

    while (datapos && key != keyend) {
        TSymbol label = *(key++);
        if (!Advance(datapos, dataend, valuepos, label, Packer)) {
            return false;
        }
        if (valuepos) { // There is a value at the end of this symbol.
            prefixLen = keylen - (keyend - key);
            return true;
        }
    }

    return false;
}

template <class TTrie>
TPrefixIterator<TTrie> MakePrefixIterator(const TTrie& trie, const typename TTrie::TSymbol* key, size_t keylen) {
    return TPrefixIterator<TTrie>(trie, key, keylen);
}
