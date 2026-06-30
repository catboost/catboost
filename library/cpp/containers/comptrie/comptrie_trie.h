#pragma once

#include "comptrie_impl.h"
#include "comptrie_packer.h"
#include "opaque_trie_iterator.h"
#include "leaf_skipper.h"
#include "key_selector.h"

#include <util/generic/buffer.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/memory/blob.h>
#include <util/stream/input.h>
#include <utility>

template <class T, class D, class S>
class TCompactTrieBuilder;

namespace NCompactTrie {
    template <class TTrie>
    class TFirstSymbolIterator;
}

template <class TTrie>
class TSearchIterator;

template <class TTrie>
class TPrefixIterator;

// in case of <char> specialization cannot distinguish between "" and "\0" keys
template <class T = char, class D = ui64, class S = TCompactTriePacker<D>>
class TCompactTrie {
public:
    typedef T TSymbol;
    typedef D TData;
    typedef S TPacker;

    typedef typename TCompactTrieKeySelector<TSymbol>::TKey TKey;
    typedef typename TCompactTrieKeySelector<TSymbol>::TKeyBuf TKeyBuf;

    typedef std::pair<TKey, TData> TValueType;
    typedef std::pair<size_t, TData> TPhraseMatch;
    typedef TVector<TPhraseMatch> TPhraseMatchVector;

    typedef TCompactTrieBuilder<T, D, S> TBuilder;

protected:
    TBlob DataHolder;
    const char* EmptyValue = nullptr;
    TPacker Packer;
    NCompactTrie::TPackerLeafSkipper<TPacker> Skipper = &Packer; // This should be true for every constructor.

public:
    TCompactTrie() = default;

    TCompactTrie(const char* d, size_t len, TPacker packer);
    TCompactTrie(const char* d, size_t len)
        : TCompactTrie{d, len, TPacker{}} {
    }

    TCompactTrie(TBlob data, TPacker packer);
    explicit TCompactTrie(TBlob data)
        : TCompactTrie{std::move(data), TPacker{}} {
    }

    // Skipper should be initialized with &Packer, not with &other.Packer, so you have to redefine these.
    TCompactTrie(const TCompactTrie& other);
    TCompactTrie(TCompactTrie&& other) noexcept;
    TCompactTrie& operator=(const TCompactTrie& other);
    TCompactTrie& operator=(TCompactTrie&& other) noexcept;

    explicit operator bool() const {
        return !IsEmpty();
    }

    void Init(const char* d, size_t len, TPacker packer = TPacker());
    void Init(TBlob data, TPacker packer = TPacker());

    bool IsInitialized() const;
    bool IsEmpty() const;

    bool Find(const TSymbol* key, size_t keylen, TData* value = nullptr) const;
    bool Find(const TKeyBuf& key, TData* value = nullptr) const {
        return Find(key.data(), key.size(), value);
    }

    TData Get(const TSymbol* key, size_t keylen) const {
        TData value;
        if (!Find(key, keylen, &value))
            ythrow yexception() << "key " << TKey(key, keylen).Quote() << " not found in trie";
        return value;
    }
    TData Get(const TKeyBuf& key) const {
        return Get(key.data(), key.size());
    }
    TData GetDefault(const TKeyBuf& key, const TData& def) const {
        TData value;
        if (!Find(key.data(), key.size(), &value))
            return def;
        else
            return value;
    }

    const TBlob& Data() const {
        return DataHolder;
    }

    const NCompactTrie::ILeafSkipper& GetSkipper() const {
        return Skipper;
    }

    TPacker GetPacker() const {
        return Packer;
    }

    bool HasCorrectSkipper() const {
        return Skipper.GetPacker() == &Packer;
    }

    void FindPhrases(const TSymbol* key, size_t keylen, TPhraseMatchVector& matches, TSymbol separator = TSymbol(' ')) const;
    void FindPhrases(const TKeyBuf& key, TPhraseMatchVector& matches, TSymbol separator = TSymbol(' ')) const {
        return FindPhrases(key.data(), key.size(), matches, separator);
    }
    bool FindLongestPrefix(const TSymbol* key, size_t keylen, size_t* prefixLen, TData* value = nullptr, bool* hasNext = nullptr) const;
    bool FindLongestPrefix(const TKeyBuf& key, size_t* prefixLen, TData* value = nullptr, bool* hasNext = nullptr) const {
        return FindLongestPrefix(key.data(), key.size(), prefixLen, value, hasNext);
    }

    // Return trie, containing all tails for the given key
    inline TCompactTrie<T, D, S> FindTails(const TSymbol* key, size_t keylen) const;
    TCompactTrie<T, D, S> FindTails(const TKeyBuf& key) const {
        return FindTails(key.data(), key.size());
    }
    bool FindTails(const TSymbol* key, size_t keylen, TCompactTrie<T, D, S>& res) const;
    bool FindTails(const TKeyBuf& key, TCompactTrie<T, D, S>& res) const {
        return FindTails(key.data(), key.size(), res);
    }

    // same as FindTails(&key, 1), a bit faster
    // return false, if no arc with @label exists
    inline bool FindTails(TSymbol label, TCompactTrie<T, D, S>& res) const;

    class TConstIterator {
    private:
        typedef NCompactTrie::TOpaqueTrieIterator TOpaqueTrieIterator;
        typedef NCompactTrie::TOpaqueTrie TOpaqueTrie;
        friend class TCompactTrie;
        TConstIterator(const TOpaqueTrie& trie, const char* emptyValue, bool atend, TPacker packer);         // only usable from Begin() and End() methods
        TConstIterator(const TOpaqueTrie& trie, const char* emptyValue, const TKeyBuf& key, TPacker packer); // only usable from UpperBound() method

    public:
        TConstIterator() = default;
        bool IsEmpty() const {
            return !Impl;
        } // Almost no other method can be called.

        bool operator==(const TConstIterator& other) const;
        bool operator!=(const TConstIterator& other) const;
        TConstIterator& operator++();
        TConstIterator operator++(int /*unused*/);
        TConstIterator& operator--();
        TConstIterator operator--(int /*unused*/);
        TValueType operator*();

        TKey GetKey() const;
        size_t GetKeySize() const;
        TData GetValue() const;
        void GetValue(TData& data) const;
        const char* GetValuePtr() const;

    private:
        TPacker Packer;
        TCopyPtr<TOpaqueTrieIterator> Impl;
    };

    TConstIterator Begin() const;
    TConstIterator begin() const;
    TConstIterator End() const;
    TConstIterator end() const;

    // Returns an iterator pointing to the smallest key in the trie >= the argument.
    // TODO: misleading name. Should be called LowerBound for consistency with stl.
    // No. It is the STL that has a misleading name.
    // LowerBound of X cannot be greater than X.
    TConstIterator UpperBound(const TKeyBuf& key) const;

    void Print(IOutputStream& os);

    size_t Size() const;

    friend class NCompactTrie::TFirstSymbolIterator<TCompactTrie>;
    friend class TSearchIterator<TCompactTrie>;
    friend class TPrefixIterator<TCompactTrie>;

protected:
    explicit TCompactTrie(const char* emptyValue);
    TCompactTrie(TBlob data, const char* emptyValue, TPacker packer = TPacker());

    bool LookupLongestPrefix(const TSymbol* key, size_t keylen, size_t& prefixLen, const char*& valuepos, bool& hasNext) const;
    bool LookupLongestPrefix(const TSymbol* key, size_t keylen, size_t& prefixLen, const char*& valuepos) const {
        bool hasNext;
        return LookupLongestPrefix(key, keylen, prefixLen, valuepos, hasNext);
    }
    void LookupPhrases(const char* datapos, size_t len, const TSymbol* key, size_t keylen, TVector<TPhraseMatch>& matches, TSymbol separator) const;
};

template <class T = char, class D = ui64, class S = TCompactTriePacker<D>>
class TCompactTrieHolder: public TCompactTrie<T, D, S>, NNonCopyable::TNonCopyable {
private:
    typedef TCompactTrie<T, D, S> TBase;
    TArrayHolder<char> Storage;

public:
    TCompactTrieHolder(IInputStream& is, size_t len);
};

//------------------------//
// Implementation section //
//------------------------//

// TCompactTrie

template <class T, class D, class S>
TCompactTrie<T, D, S>::TCompactTrie(TBlob data, TPacker packer)
{
    Init(std::move(data), packer);
}

template <class T, class D, class S>
TCompactTrie<T, D, S>::TCompactTrie(const char* d, size_t len, TPacker packer)
{
    Init(d, len, packer);
}

template <class T, class D, class S>
TCompactTrie<T, D, S>::TCompactTrie(const char* emptyValue)
    : EmptyValue(emptyValue)
{
}

template <class T, class D, class S>
TCompactTrie<T, D, S>::TCompactTrie(TBlob data, const char* emptyValue, TPacker packer)
    : DataHolder(std::move(data))
    , EmptyValue(emptyValue)
    , Packer(packer)
{
}

template <class T, class D, class S>
TCompactTrie<T, D, S>::TCompactTrie(const TCompactTrie& other)
    : DataHolder(other.DataHolder)
    , EmptyValue(other.EmptyValue)
    , Packer(other.Packer)
{
}

template <class T, class D, class S>
TCompactTrie<T, D, S>::TCompactTrie(TCompactTrie&& other) noexcept
    : DataHolder(std::move(other.DataHolder))
    , EmptyValue(std::move(other.EmptyValue))
    , Packer(std::move(other.Packer))
{
}

template <class T, class D, class S>
TCompactTrie<T, D, S>& TCompactTrie<T, D, S>::operator=(const TCompactTrie& other) {
    if (this != &other) {
        DataHolder = other.DataHolder;
        EmptyValue = other.EmptyValue;
        Packer = other.Packer;
    }
    return *this;
}

template <class T, class D, class S>
TCompactTrie<T, D, S>& TCompactTrie<T, D, S>::operator=(TCompactTrie&& other) noexcept {
    if (this != &other) {
        DataHolder = std::move(other.DataHolder);
        EmptyValue = std::move(other.EmptyValue);
        Packer = std::move(other.Packer);
    }
    return *this;
}

template <class T, class D, class S>
void TCompactTrie<T, D, S>::Init(const char* d, size_t len, TPacker packer) {
    Init(TBlob::NoCopy(d, len), packer);
}

template <class T, class D, class S>
void TCompactTrie<T, D, S>::Init(TBlob data, TPacker packer) {
    using namespace NCompactTrie;

    DataHolder = std::move(data);
    Packer = packer;

    const char* datapos = DataHolder.AsCharPtr();
    size_t len = DataHolder.Length();
    if (!len)
        return;

    const char* const dataend = datapos + len;

    const char* emptypos = datapos;
    char flags = LeapByte(emptypos, dataend, 0);
    if (emptypos && (flags & MT_FINAL)) {
        Y_ASSERT(emptypos <= dataend);
        EmptyValue = emptypos;
    }
}

template <class T, class D, class S>
bool TCompactTrie<T, D, S>::IsInitialized() const {
    return DataHolder.Data() != nullptr;
}

template <class T, class D, class S>
bool TCompactTrie<T, D, S>::IsEmpty() const {
    return DataHolder.Size() == 0 && EmptyValue == nullptr;
}

template <class T, class D, class S>
bool TCompactTrie<T, D, S>::Find(const TSymbol* key, size_t keylen, TData* value) const {
    size_t prefixLen = 0;
    const char* valuepos = nullptr;
    bool hasNext;
    if (!LookupLongestPrefix(key, keylen, prefixLen, valuepos, hasNext) || prefixLen != keylen)
        return false;
    if (value)
        Packer.UnpackLeaf(valuepos, *value);
    return true;
}

template <class T, class D, class S>
void TCompactTrie<T, D, S>::FindPhrases(const TSymbol* key, size_t keylen, TPhraseMatchVector& matches, TSymbol separator) const {
    LookupPhrases(DataHolder.AsCharPtr(), DataHolder.Length(), key, keylen, matches, separator);
}

template <class T, class D, class S>
inline TCompactTrie<T, D, S> TCompactTrie<T, D, S>::FindTails(const TSymbol* key, size_t keylen) const {
    TCompactTrie<T, D, S> ret;
    FindTails(key, keylen, ret);
    return ret;
}

template <class T, class D, class S>
bool TCompactTrie<T, D, S>::FindTails(const TSymbol* key, size_t keylen, TCompactTrie<T, D, S>& res) const {
    using namespace NCompactTrie;

    size_t len = DataHolder.Length();

    if (!key || !len)
        return false;

    if (!keylen) {
        res = *this;
        return true;
    }

    const char* datastart = DataHolder.AsCharPtr();
    const char* datapos = datastart;
    const char* const dataend = datapos + len;

    const TSymbol* keyend = key + keylen;
    const char* value = nullptr;

    while (key != keyend) {
        T label = *(key++);
        if (!NCompactTrie::Advance(datapos, dataend, value, label, Packer))
            return false;

        if (key == keyend) {
            if (datapos) {
                Y_ASSERT(datapos >= datastart);
                res = TCompactTrie<T, D, S>(TBlob::NoCopy(datapos, dataend - datapos), value);
            } else {
                res = TCompactTrie<T, D, S>(value);
            }
            return true;
        } else if (!datapos) {
            return false; // No further way
        }
    }

    return false;
}

template <class T, class D, class S>
inline bool TCompactTrie<T, D, S>::FindTails(TSymbol label, TCompactTrie<T, D, S>& res) const {
    using namespace NCompactTrie;

    const size_t len = DataHolder.Length();
    if (!len)
        return false;

    const char* datastart = DataHolder.AsCharPtr();
    const char* dataend = datastart + len;
    const char* datapos = datastart;
    const char* value = nullptr;

    if (!NCompactTrie::Advance(datapos, dataend, value, label, Packer))
        return false;

    if (datapos) {
        Y_ASSERT(datapos >= datastart);
        res = TCompactTrie<T, D, S>(TBlob::NoCopy(datapos, dataend - datapos), value);
    } else {
        res = TCompactTrie<T, D, S>(value);
    }

    return true;
}

template <class T, class D, class S>
typename TCompactTrie<T, D, S>::TConstIterator TCompactTrie<T, D, S>::Begin() const {
    NCompactTrie::TOpaqueTrie self(DataHolder.AsCharPtr(), DataHolder.Length(), Skipper);
    return TConstIterator(self, EmptyValue, false, Packer);
}

template <class T, class D, class S>
typename TCompactTrie<T, D, S>::TConstIterator TCompactTrie<T, D, S>::begin() const {
    return Begin();
}

template <class T, class D, class S>
typename TCompactTrie<T, D, S>::TConstIterator TCompactTrie<T, D, S>::End() const {
    NCompactTrie::TOpaqueTrie self(DataHolder.AsCharPtr(), DataHolder.Length(), Skipper);
    return TConstIterator(self, EmptyValue, true, Packer);
}

template <class T, class D, class S>
typename TCompactTrie<T, D, S>::TConstIterator TCompactTrie<T, D, S>::end() const {
    return End();
}

template <class T, class D, class S>
size_t TCompactTrie<T, D, S>::Size() const {
    size_t res = 0;
    for (TConstIterator it = Begin(); it != End(); ++it)
        ++res;
    return res;
}

template <class T, class D, class S>
typename TCompactTrie<T, D, S>::TConstIterator TCompactTrie<T, D, S>::UpperBound(const TKeyBuf& key) const {
    NCompactTrie::TOpaqueTrie self(DataHolder.AsCharPtr(), DataHolder.Length(), Skipper);
    return TConstIterator(self, EmptyValue, key, Packer);
}

template <class T, class D, class S>
void TCompactTrie<T, D, S>::Print(IOutputStream& os) {
    typedef typename ::TCompactTrieKeySelector<T>::TKeyBuf TSBuffer;
    for (TConstIterator it = Begin(); it != End(); ++it) {
        os << TSBuffer((*it).first.data(), (*it).first.size()) << "\t" << (*it).second << Endl;
    }
}

template <class T, class D, class S>
bool TCompactTrie<T, D, S>::FindLongestPrefix(const TSymbol* key, size_t keylen, size_t* prefixLen, TData* value, bool* hasNext) const {
    const char* valuepos = nullptr;
    size_t tempPrefixLen = 0;
    bool tempHasNext;
    bool found = LookupLongestPrefix(key, keylen, tempPrefixLen, valuepos, tempHasNext);
    if (prefixLen)
        *prefixLen = tempPrefixLen;
    if (found && value)
        Packer.UnpackLeaf(valuepos, *value);
    if (hasNext)
        *hasNext = tempHasNext;
    return found;
}

template <class T, class D, class S>
bool TCompactTrie<T, D, S>::LookupLongestPrefix(const TSymbol* key, size_t keylen, size_t& prefixLen, const char*& valuepos, bool& hasNext) const {
    using namespace NCompactTrie;

    const char* datapos = DataHolder.AsCharPtr();
    size_t len = DataHolder.Length();

    prefixLen = 0;
    hasNext = false;
    bool found = false;

    if (EmptyValue) {
        valuepos = EmptyValue;
        found = true;
    }

    if (!key || !len)
        return found;

    const char* const dataend = datapos + len;

    const T* keyend = key + keylen;
    while (key != keyend) {
        T label = *(key++);
        for (i64 i = (i64)ExtraBits<TSymbol>(); i >= 0; i -= 8) {
            const char flags = LeapByte(datapos, dataend, (char)(label >> i));
            if (!datapos) {
                return found; // no such arc
            }

            Y_ASSERT(datapos <= dataend);
            if ((flags & MT_FINAL)) {
                prefixLen = keylen - (keyend - key) - (i ? 1 : 0);
                valuepos = datapos;
                hasNext = flags & MT_NEXT;
                found = true;

                if (!i && key == keyend) { // last byte, and got a match
                    return found;
                }
                datapos += Packer.SkipLeaf(datapos); // skip intermediate leaf nodes
            }

            if (!(flags & MT_NEXT)) {
                return found; // no further way
            }
        }
    }

    return found;
}

template <class T, class D, class S>
void TCompactTrie<T, D, S>::LookupPhrases(
    const char* datapos, size_t len, const TSymbol* key, size_t keylen,
    TVector<TPhraseMatch>& matches, TSymbol separator) const {
    using namespace NCompactTrie;

    matches.clear();

    if (!key || !len)
        return;

    const T* const keystart = key;
    const T* const keyend = key + keylen;
    const char* const dataend = datapos + len;
    while (datapos && key != keyend) {
        T label = *(key++);
        const char* value = nullptr;
        if (!Advance(datapos, dataend, value, label, Packer)) {
            return;
        }
        if (value && (key == keyend || *key == separator)) {
            size_t matchlength = (size_t)(key - keystart);
            D data;
            Packer.UnpackLeaf(value, data);
            matches.push_back(TPhraseMatch(matchlength, data));
        }
    }
}

// TCompactTrieHolder

template <class T, class D, class S>
TCompactTrieHolder<T, D, S>::TCompactTrieHolder(IInputStream& is, size_t len)
    : Storage(new char[len])
{
    if (is.Load(Storage.Get(), len) != len) {
        ythrow yexception() << "bad data load";
    }
    TBase::Init(Storage.Get(), len);
}

//----------------------------------------------------------------------------------------------------------------
// TCompactTrie::TConstIterator

template <class T, class D, class S>
TCompactTrie<T, D, S>::TConstIterator::TConstIterator(const TOpaqueTrie& trie, const char* emptyValue, bool atend, TPacker packer)
    : Packer(packer)
    , Impl(new TOpaqueTrieIterator(trie, emptyValue, atend))
{
}

template <class T, class D, class S>
TCompactTrie<T, D, S>::TConstIterator::TConstIterator(const TOpaqueTrie& trie, const char* emptyValue, const TKeyBuf& key, TPacker packer)
    : Packer(packer)
    , Impl(new TOpaqueTrieIterator(trie, emptyValue, true))
{
    Impl->UpperBound<TSymbol>(key);
}

template <class T, class D, class S>
bool TCompactTrie<T, D, S>::TConstIterator::operator==(const TConstIterator& other) const {
    if (!Impl)
        return !other.Impl;
    if (!other.Impl)
        return false;
    return *Impl == *other.Impl;
}

template <class T, class D, class S>
bool TCompactTrie<T, D, S>::TConstIterator::operator!=(const TConstIterator& other) const {
    return !operator==(other);
}

template <class T, class D, class S>
typename TCompactTrie<T, D, S>::TConstIterator& TCompactTrie<T, D, S>::TConstIterator::operator++() {
    Impl->Forward();
    return *this;
}

template <class T, class D, class S>
typename TCompactTrie<T, D, S>::TConstIterator TCompactTrie<T, D, S>::TConstIterator::operator++(int /*unused*/) {
    TConstIterator copy(*this);
    Impl->Forward();
    return copy;
}

template <class T, class D, class S>
typename TCompactTrie<T, D, S>::TConstIterator& TCompactTrie<T, D, S>::TConstIterator::operator--() {
    Impl->Backward();
    return *this;
}

template <class T, class D, class S>
typename TCompactTrie<T, D, S>::TConstIterator TCompactTrie<T, D, S>::TConstIterator::operator--(int /*unused*/) {
    TConstIterator copy(*this);
    Impl->Backward();
    return copy;
}

template <class T, class D, class S>
typename TCompactTrie<T, D, S>::TValueType TCompactTrie<T, D, S>::TConstIterator::operator*() {
    return TValueType(GetKey(), GetValue());
}

template <class T, class D, class S>
typename TCompactTrie<T, D, S>::TKey TCompactTrie<T, D, S>::TConstIterator::GetKey() const {
    return Impl->GetKey<TSymbol>();
}

template <class T, class D, class S>
size_t TCompactTrie<T, D, S>::TConstIterator::GetKeySize() const {
    return Impl->MeasureKey<TSymbol>();
}

template <class T, class D, class S>
const char* TCompactTrie<T, D, S>::TConstIterator::GetValuePtr() const {
    return Impl->GetValuePtr();
}

template <class T, class D, class S>
typename TCompactTrie<T, D, S>::TData TCompactTrie<T, D, S>::TConstIterator::GetValue() const {
    D data;
    GetValue(data);
    return data;
}

template <class T, class D, class S>
void TCompactTrie<T, D, S>::TConstIterator::GetValue(typename TCompactTrie<T, D, S>::TData& data) const {
    const char* ptr = GetValuePtr();
    if (ptr) {
        Packer.UnpackLeaf(ptr, data);
    } else {
        data = typename TCompactTrie<T, D, S>::TData();
    }
}
