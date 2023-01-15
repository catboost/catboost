#pragma once

#include "opaque_trie_iterator.h"
#include <util/generic/ptr.h>

namespace NCompactTrie {
    // Iterates over possible first symbols in a trie.
    // Allows one to get the symbol and the subtrie starting from it.
    template <class TTrie>
    class TFirstSymbolIterator {
    public:
        using TSymbol = typename TTrie::TSymbol;
        using TData = typename TTrie::TData;

        void SetTrie(const TTrie& trie, const ILeafSkipper& skipper) {
            Trie = trie;
            Impl.Reset(new TOpaqueTrieIterator(
                TOpaqueTrie(Trie.Data().AsCharPtr(), Trie.Data().Size(), skipper),
                nullptr,
                false,
                sizeof(TSymbol)));
            if (Impl->MeasureKey<TSymbol>() == 0) {
                MakeStep();
            }
        }

        const TTrie& GetTrie() const {
            return Trie;
        }

        bool AtEnd() const {
            return *Impl == TOpaqueTrieIterator(Impl->GetTrie(), nullptr, true, sizeof(TSymbol));
        }

        TSymbol GetKey() const {
            return Impl->GetKey<TSymbol>()[0];
        }

        TTrie GetTails() const {
            const TNode& node = Impl->GetNode();
            const size_t forwardOffset = node.GetForwardOffset();
            const char* emptyValue = node.IsFinal() ? Trie.Data().AsCharPtr() + node.GetLeafOffset() : nullptr;
            if (forwardOffset) {
                const char* start = Trie.Data().AsCharPtr() + forwardOffset;
                TBlob body = TBlob::NoCopy(start, Trie.Data().Size() - forwardOffset);
                return TTrie(body, emptyValue, Trie.GetPacker());
            } else {
                return TTrie(emptyValue);
            }
        }

        void MakeStep() {
            Impl->Forward();
        }

    private:
        TTrie Trie;
        TCopyPtr<TOpaqueTrieIterator> Impl;
    };

}
