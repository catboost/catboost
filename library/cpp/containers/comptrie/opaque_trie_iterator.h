#pragma once

#include "comptrie_impl.h"
#include "node.h"
#include "key_selector.h"
#include "leaf_skipper.h"

#include <util/generic/vector.h>
#include <util/generic/yexception.h>

namespace NCompactTrie {
    class ILeafSkipper;

    class TFork { // Auxiliary class for a branching point in the iterator
    public:
        TNode Node;
        const char* Data;
        size_t Limit; // valid data is in range [Data + Node.GetOffset(), Data + Limit)
        TDirection CurrentDirection;

    public:
        TFork(const char* data, size_t offset, size_t limit, const ILeafSkipper& skipper);

        bool operator==(const TFork& rhs) const;

        bool HasLabelInKey() const {
            return CurrentDirection == D_NEXT || CurrentDirection == D_FINAL;
        }

        bool NextDirection();
        bool PrevDirection();
        void LastDirection();

        bool HasDirection(TDirection direction) const {
            return Node.GetOffsetByDirection(direction);
        }
        // If the fork doesn't have the specified direction,
        // leaves the direction intact and returns false.
        // Otherwise returns true.
        bool SetDirection(TDirection direction);
        TFork NextFork(const ILeafSkipper& skipper) const;

        char GetLabel() const;
        size_t GetValueOffset() const;
    };

    inline TFork TFork::NextFork(const ILeafSkipper& skipper) const {
        Y_ASSERT(CurrentDirection != D_FINAL);
        size_t offset = Node.GetOffsetByDirection(CurrentDirection);
        return TFork(Data, offset, Limit, skipper);
    }

    //------------------------------------------------------------------------------------------------
    class TForkStack {
    public:
        void Push(const TFork& fork) {
            if (TopHasLabelInKey()) {
                Key.push_back(Top().GetLabel());
            }
            Forks.push_back(fork);
        }

        void Pop() {
            Forks.pop_back();
            if (TopHasLabelInKey()) {
                Key.pop_back();
            }
        }

        TFork& Top() {
            return Forks.back();
        }
        const TFork& Top() const {
            return Forks.back();
        }

        bool Empty() const {
            return Forks.empty();
        }

        void Clear() {
            Forks.clear();
            Key.clear();
        }

        bool operator==(const TForkStack& other) const {
            return Forks == other.Forks;
        }
        bool operator!=(const TForkStack& other) const {
            return !(*this == other);
        }

        TString GetKey() const;
        size_t MeasureKey() const;

    private:
        TVector<TFork> Forks;
        TString Key;

    private:
        bool TopHasLabelInKey() const {
            return !Empty() && Top().HasLabelInKey();
        }
        bool HasEmptyKey() const;
    };

    //------------------------------------------------------------------------------------------------

    template <class TSymbol>
    struct TConvertRawKey {
        typedef typename TCompactTrieKeySelector<TSymbol>::TKey TKey;
        static TKey Get(const TString& rawkey) {
            TKey result;
            const size_t sz = rawkey.size();
            result.reserve(sz / sizeof(TSymbol));
            for (size_t i = 0; i < sz; i += sizeof(TSymbol)) {
                TSymbol sym = 0;
                for (size_t j = 0; j < sizeof(TSymbol); j++) {
                    if (sizeof(TSymbol) <= 1)
                        sym = 0;
                    else
                        sym <<= 8;
                    if (i + j < sz)
                        sym |= TSymbol(0x00FF & rawkey[i + j]);
                }
                result.push_back(sym);
            }
            return result;
        }

        static size_t Size(size_t rawsize) {
            return rawsize / sizeof(TSymbol);
        }
    };

    template <>
    struct TConvertRawKey<char> {
        static TString Get(const TString& rawkey) {
            return rawkey;
        }

        static size_t Size(size_t rawsize) {
            return rawsize;
        }
    };

    //------------------------------------------------------------------------------------------------
    class TOpaqueTrieIterator { // Iterator stuff. Stores a stack of visited forks.
    public:
        TOpaqueTrieIterator(const TOpaqueTrie& trie, const char* emptyValue, bool atend,
                            size_t maxKeyLength = size_t(-1));

        bool operator==(const TOpaqueTrieIterator& rhs) const;
        bool operator!=(const TOpaqueTrieIterator& rhs) const {
            return !(*this == rhs);
        }

        bool Forward();
        bool Backward();

        template <class TSymbol>
        bool UpperBound(const typename TCompactTrieKeySelector<TSymbol>::TKeyBuf& key); // True if matched exactly.

        template <class TSymbol>
        typename TCompactTrieKeySelector<TSymbol>::TKey GetKey() const {
            return TConvertRawKey<TSymbol>::Get(GetNarrowKey());
        }

        template <class TSymbol>
        size_t MeasureKey() const {
            return TConvertRawKey<TSymbol>::Size(MeasureNarrowKey());
        }

        TString GetNarrowKey() const {
            return Forks.GetKey();
        }
        size_t MeasureNarrowKey() const {
            return Forks.MeasureKey();
        }

        const char* GetValuePtr() const; // 0 if none
        const TNode& GetNode() const {   // Could be called for non-empty key and not AtEnd.
            return Forks.Top().Node;
        }
        const TOpaqueTrie& GetTrie() const {
            return Trie;
        }

    private:
        TOpaqueTrie Trie;
        TForkStack Forks;
        const char* const EmptyValue;
        bool AtEmptyValue;
        const size_t MaxKeyLength;

    private:
        bool HasMaxKeyLength() const;

        template <class TSymbol>
        int LongestPrefix(const typename TCompactTrieKeySelector<TSymbol>::TKeyBuf& key); // Used in UpperBound.
    };

    template <class TSymbol>
    int TOpaqueTrieIterator::LongestPrefix(const typename TCompactTrieKeySelector<TSymbol>::TKeyBuf& key) {
        Forks.Clear();
        TFork next(Trie.Data, 0, Trie.Length, Trie.SkipFunction);
        for (size_t i = 0; i < key.size(); i++) {
            TSymbol symbol = key[i];
            const bool isLastSymbol = (i + 1 == key.size());
            for (i64 shift = (i64)NCompactTrie::ExtraBits<TSymbol>(); shift >= 0; shift -= 8) {
                const unsigned char label = (unsigned char)(symbol >> shift);
                const bool isLastByte = (isLastSymbol && shift == 0);
                do {
                    Forks.Push(next);
                    TFork& top = Forks.Top();
                    if (label < (unsigned char)top.GetLabel()) {
                        if (!top.SetDirection(D_LEFT))
                            return 1;
                    } else if (label > (unsigned char)top.GetLabel()) {
                        if (!top.SetDirection(D_RIGHT)) {
                            Forks.Pop(); // We don't pass this fork on the way to the upper bound.
                            return -1;
                        }
                    } else if (isLastByte) { // Here and below label == top.GetLabel().
                        if (top.SetDirection(D_FINAL)) {
                            return 0; // Skip the NextFork() call at the end of the cycle.
                        } else {
                            top.SetDirection(D_NEXT);
                            return 1;
                        }
                    } else if (!top.SetDirection(D_NEXT)) {
                        top.SetDirection(D_FINAL);
                        return -1;
                    }
                    next = top.NextFork(Trie.SkipFunction);
                } while (Forks.Top().CurrentDirection != D_NEXT); // Proceed to the next byte.
            }
        }
        // We get here only if the key was empty.
        Forks.Push(next);
        return 1;
    }

    template <class TSymbol>
    bool TOpaqueTrieIterator::UpperBound(const typename TCompactTrieKeySelector<TSymbol>::TKeyBuf& key) {
        Forks.Clear();
        if (key.empty() && EmptyValue) {
            AtEmptyValue = true;
            return true;
        } else {
            AtEmptyValue = false;
        }
        const int defect = LongestPrefix<TSymbol>(key);
        if (defect > 0) {
            // Continue the constructed forks with the smallest key possible.
            while (Forks.Top().CurrentDirection != D_FINAL) {
                TFork next = Forks.Top().NextFork(Trie.SkipFunction);
                Forks.Push(next);
            }
        } else if (defect < 0) {
            Forward();
        }
        return defect == 0;
    }

}
