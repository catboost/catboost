#pragma once

#include "comptrie_trie.h"

template <typename T = char>
class TCompactTrieSet: public TCompactTrie<T, ui8, TNullPacker<ui8>> {
public:
    typedef TCompactTrie<T, ui8, TNullPacker<ui8>> TBase;

    using typename TBase::TBuilder;
    using typename TBase::TKey;
    using typename TBase::TKeyBuf;
    using typename TBase::TSymbol;

    TCompactTrieSet() = default;

    explicit TCompactTrieSet(const TBlob& data)
        : TBase(data)
    {
    }

    template <typename D>
    explicit TCompactTrieSet(const TCompactTrie<T, D, TNullPacker<D>>& trie)
        : TBase(trie.Data()) // should be binary compatible for any D
    {
    }

    TCompactTrieSet(const char* data, size_t len)
        : TBase(data, len)
    {
    }

    bool Has(const typename TBase::TKeyBuf& key) const {
        return TBase::Find(key.data(), key.size());
    }

    bool FindTails(const typename TBase::TKeyBuf& key, TCompactTrieSet<T>& res) const {
        return TBase::FindTails(key, res);
    }
};
