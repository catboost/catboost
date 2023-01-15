#include "opaque_trie_iterator.h"
#include "comptrie_impl.h"
#include "node.h"

namespace NCompactTrie {
    TOpaqueTrieIterator::TOpaqueTrieIterator(const TOpaqueTrie& trie, const char* emptyValue, bool atend,
                                             size_t maxKeyLength)
        : Trie(trie)
        , EmptyValue(emptyValue)
        , AtEmptyValue(emptyValue && !atend)
        , MaxKeyLength(maxKeyLength)
    {
        if (!AtEmptyValue && !atend)
            Forward();
    }

    bool TOpaqueTrieIterator::operator==(const TOpaqueTrieIterator& rhs) const {
        return (Trie == rhs.Trie &&
                Forks == rhs.Forks &&
                EmptyValue == rhs.EmptyValue &&
                AtEmptyValue == rhs.AtEmptyValue &&
                MaxKeyLength == rhs.MaxKeyLength);
    }

    bool TOpaqueTrieIterator::HasMaxKeyLength() const {
        return MaxKeyLength != size_t(-1) && MeasureNarrowKey() == MaxKeyLength;
    }

    bool TOpaqueTrieIterator::Forward() {
        if (AtEmptyValue) {
            AtEmptyValue = false;
            bool res = Forward(); // TODO delete this after format change
            if (res && MeasureNarrowKey() != 0) {
                return res; // there was not "\0" key
            }
            // otherwise we are skipping "\0" key
        }

        if (!Trie.Length)
            return false;

        if (Forks.Empty()) {
            TFork fork(Trie.Data, 0, Trie.Length, Trie.SkipFunction);
            Forks.Push(fork);
        } else {
            TFork* topFork = &Forks.Top();
            while (!topFork->NextDirection()) {
                if (topFork->Node.GetOffset() >= Trie.Length)
                    return false;
                Forks.Pop();
                if (Forks.Empty())
                    return false;
                topFork = &Forks.Top();
            }
        }

        Y_ASSERT(!Forks.Empty());
        while (Forks.Top().CurrentDirection != D_FINAL && !HasMaxKeyLength()) {
            TFork nextFork = Forks.Top().NextFork(Trie.SkipFunction);
            Forks.Push(nextFork);
        }
        TFork& top = Forks.Top();
        static_assert(D_FINAL < D_NEXT, "relative order of NEXT and FINAL directions has changed");
        if (HasMaxKeyLength() && top.CurrentDirection == D_FINAL && top.HasDirection(D_NEXT)) {
            top.NextDirection();
        }
        return true;
    }

    bool TOpaqueTrieIterator::Backward() {
        if (AtEmptyValue)
            return false;

        if (!Trie.Length) {
            if (EmptyValue) {
                // A trie that has only the empty value;
                // we are not at the empty value, so move to it.
                AtEmptyValue = true;
                return true;
            } else {
                // Empty trie.
                return false;
            }
        }

        if (Forks.Empty()) {
            TFork fork(Trie.Data, 0, Trie.Length, Trie.SkipFunction);
            fork.LastDirection();
            Forks.Push(fork);
        } else {
            TFork* topFork = &Forks.Top();
            while (!topFork->PrevDirection()) {
                if (topFork->Node.GetOffset() >= Trie.Length)
                    return false;
                Forks.Pop();
                if (!Forks.Empty()) {
                    topFork = &Forks.Top();
                } else {
                    // When there are no more forks,
                    // we have to iterate over the empty value.
                    if (!EmptyValue)
                        return false;
                    AtEmptyValue = true;
                    return true;
                }
            }
        }

        Y_ASSERT(!Forks.Empty());
        while (Forks.Top().CurrentDirection != D_FINAL && !HasMaxKeyLength()) {
            TFork nextFork = Forks.Top().NextFork(Trie.SkipFunction);
            nextFork.LastDirection();
            Forks.Push(nextFork);
        }
        TFork& top = Forks.Top();
        static_assert(D_FINAL < D_NEXT, "relative order of NEXT and FINAL directions has changed");
        if (HasMaxKeyLength() && top.CurrentDirection == D_NEXT && top.HasDirection(D_FINAL)) {
            top.PrevDirection();
        }
        if (MeasureNarrowKey() == 0) {
            // This is the '\0' key, skip it and get to the EmptyValue.
            AtEmptyValue = true;
            Forks.Clear();
        }
        return true;
    }

    const char* TOpaqueTrieIterator::GetValuePtr() const {
        if (!Forks.Empty()) {
            const TFork& lastFork = Forks.Top();
            Y_ASSERT(lastFork.Node.IsFinal() && lastFork.CurrentDirection == D_FINAL);
            return Trie.Data + lastFork.GetValueOffset();
        }
        Y_ASSERT(AtEmptyValue);
        return EmptyValue;
    }

    //-------------------------------------------------------------------------

    TString TForkStack::GetKey() const {
        if (HasEmptyKey()) {
            return TString();
        }

        TString result(Key);
        if (TopHasLabelInKey()) {
            result.append(Top().GetLabel());
        }
        return result;
    }

    bool TForkStack::HasEmptyKey() const {
        // Special case: if we get a single zero label, treat it as an empty key
        // TODO delete this after format change
        if (TopHasLabelInKey()) {
            return Key.size() == 0 && Top().GetLabel() == '\0';
        } else {
            return Key.size() == 1 && Key[0] == '\0';
        }
    }

    size_t TForkStack::MeasureKey() const {
        size_t result = Key.size() + (TopHasLabelInKey() ? 1 : 0);
        if (result == 1 && HasEmptyKey()) {
            return 0;
        }
        return result;
    }

    //-------------------------------------------------------------------------

    TFork::TFork(const char* data, size_t offset, size_t limit, const ILeafSkipper& skipper)
        : Node(data, offset, skipper)
        , Data(data)
        , Limit(limit)
        , CurrentDirection(TDirection(0))
    {
#if COMPTRIE_DATA_CHECK
        if (Node.GetOffset() >= Limit - 1)
            ythrow yexception() << "gone beyond the limit, data is corrupted";
#endif
        while (CurrentDirection < D_MAX && !HasDirection(CurrentDirection)) {
            ++CurrentDirection;
        }
    }

    bool TFork::operator==(const TFork& rhs) const {
        return (Data == rhs.Data &&
                Node.GetOffset() == rhs.Node.GetOffset() &&
                CurrentDirection == rhs.CurrentDirection);
    }

    inline bool TFork::NextDirection() {
        do {
            ++CurrentDirection;
        } while (CurrentDirection < D_MAX && !HasDirection(CurrentDirection));
        return CurrentDirection < D_MAX;
    }

    inline bool TFork::PrevDirection() {
        if (CurrentDirection == TDirection(0)) {
            return false;
        }
        do {
            --CurrentDirection;
        } while (CurrentDirection > 0 && !HasDirection(CurrentDirection));
        return HasDirection(CurrentDirection);
    }

    void TFork::LastDirection() {
        CurrentDirection = D_MAX;
        PrevDirection();
    }

    bool TFork::SetDirection(TDirection direction) {
        if (!HasDirection(direction)) {
            return false;
        }
        CurrentDirection = direction;
        return true;
    }

    char TFork::GetLabel() const {
        return Node.GetLabel();
    }

    size_t TFork::GetValueOffset() const {
        return Node.GetLeafOffset();
    }

}
