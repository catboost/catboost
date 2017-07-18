#pragma once

#include <util/generic/utility.h>
#include <util/generic/vector.h>

#include <util/str_stl.h>
#include <util/ysaveload.h>

/*
 * There are 2 classes in this file:
 *   - TDenseHash - analog of yhash
 *   - TDenseHashSet - analog of yhash_set
 */

/*
 * Implements dense-hash, in some circumstances it is a lot (2x) faster than yhash.
 * We support only adding new elements.
 * TKey value equal to EmptyMarker (by default, it is TKey()) can not be inserted into hash - it is used as marker of empty element.
 */

template <typename TKey,
          typename TValue,
          typename TKeyHash = THash<TKey>,
          int MaxLoadFactor = 50, // in percents
          int LogInitSize = 8,
          typename TEmptyMarker = TKey>
class TDenseHash {
public:
    TDenseHash(const TEmptyMarker& emptyMarker = TEmptyMarker(), const TValue& defaultValue = TValue(), size_t initSize=0)
        : EmptyMarker(emptyMarker)
        , DefaultValue(defaultValue)
    {
        MakeEmpty(initSize);
    }

    TDenseHash(const TDenseHash&) = default;

    TDenseHash(TDenseHash&& init) {
        Swap(init);
    }

    TDenseHash& operator= (const TDenseHash&) = default;

    bool operator== (const TDenseHash& rhs) const {
        if (Size() != rhs.Size()) {
            return false;
        }
        for (const auto it : *this) {
            if (!rhs.Has(it.Key())) {
                return false;
            }
        }
        return true;
    }

    void Clear() {
        size_t currentSize = Buckets.size();
        Buckets.clear();
        Buckets.resize(currentSize, TItem(EmptyMarker, DefaultValue));
        NumFilled = 0;
    }

    void MakeEmpty(size_t initSize=0) {
        if (!initSize) {
            initSize = 1 << LogInitSize;
        } else {
            initSize = FastClp2(initSize);
        }
        BucketMask = initSize - 1;
        NumFilled = 0;
        yvector<TItem>(initSize, TItem(EmptyMarker, DefaultValue)).swap(Buckets);
        GrowThreshold = Max<size_t>(1, initSize * MaxLoadFactor / 100) - 1;
    }

    template <typename K>
    Y_FORCE_INLINE TValue* FindPtr(const K& key) {
        return ProcessBucket<TValue*>(
            key,
            [&](size_t idx) { return &Buckets[idx].Value; },
            [](size_t) { return nullptr; }
        );
    }

    template <typename K>
    Y_FORCE_INLINE const TValue* FindPtr(const K& key) const {
        return ProcessBucket<const TValue*>(
            key,
            [&](size_t idx) { return &Buckets[idx].Value; },
            [](size_t) { return nullptr; }
        );
    }

    template <typename K>
    Y_FORCE_INLINE bool Has(const K& key) const {
        return ProcessBucket<bool>(
            key,
            [](size_t) { return true; },
            [](size_t) { return false; }
        );
    }

    template <typename K>
    Y_FORCE_INLINE const TValue& Get(const K& key) const {
        return ProcessBucket<const TValue&>(
            key,
            [&](size_t idx) -> const TValue& { return Buckets[idx].Value; },
            [&](size_t) -> const TValue& { return DefaultValue; }
        );
    }

    // gets existing item or inserts new
    template <typename K>
    Y_FORCE_INLINE TValue& GetMutable(const K& key, bool* newValueWasInserted = nullptr) {
        bool newValueWasInsertedInternal;
        TValue* res = &GetMutableNoGrow(key, &newValueWasInsertedInternal);
        // It is important to grow table only if new key was inserted
        // otherwise we may invalidate all references into this table
        // which is unexpected when table was not actually modified
        if (MaybeGrow()) {
            res = &GetMutableNoGrow(key, nullptr);
        }
        if (newValueWasInserted) {
            *newValueWasInserted = newValueWasInsertedInternal;
        }
        return *res;
    }

    // might loop forever if there are not enough free buckets
    // it's users responsibility to be sure that it doesn't happen
    template <typename K>
    Y_FORCE_INLINE TValue& UnsafeGetMutableNoGrow(const K& key, bool* newValueWasInserted = nullptr) {
        return GetMutableNoGrow(key, newValueWasInserted);
    }

    size_t Capacity() const {
        return Buckets.capacity();
    }

    bool Empty() const {
        return Size() == 0;
    }

    size_t Size() const {
        return NumFilled;
    }

    template <int maxFillPercents, int logInitSize>
    void Swap(TDenseHash<TKey, TValue, TKeyHash, maxFillPercents, logInitSize>& other) {
        Buckets.swap(other.Buckets);
        DoSwap(BucketMask, other.BucketMask);
        DoSwap(NumFilled, other.NumFilled);
        DoSwap(GrowThreshold, other.GrowThreshold);
        DoSwap(EmptyMarker, other.EmptyMarker);
        DoSwap(DefaultValue, other.DefaultValue);
    }

    Y_SAVELOAD_DEFINE(BucketMask, NumFilled, GrowThreshold, Buckets, EmptyMarker, DefaultValue);

private:
    template <typename THash, typename TVal>
    class TIteratorBase {
        friend class TDenseHash;

        template <typename THash2, typename TVal2>
        friend class TIteratorBase;

        THash* Hash;
        size_t Idx;

        // used only to implement end()
        TIteratorBase(THash* hash, size_t initIdx)
            : Hash(hash)
            , Idx(initIdx)
        {
        }

    public:
        TIteratorBase(THash& hash)
            : Hash(&hash)
            , Idx(0)
        {
            if (Hash->EmptyMarker == Hash->Buckets[Idx].Key) {
                Next();
            }
        }

        template <typename THash2, typename TVal2>
        TIteratorBase(const TIteratorBase<THash2, TVal2>& it)
            : Hash(it.Hash)
            , Idx(it.Idx)
        {
        }

        static TIteratorBase CreateEmpty() {
            return TIteratorBase(nullptr, 0);
        }

        TIteratorBase& operator=(const TIteratorBase& rhs) {
            Hash = rhs.Hash;
            Idx = rhs.Idx;
            return *this;
        }

        void Next() {
            ++Idx;
            while (Idx < Hash->Buckets.size() && Hash->EmptyMarker == Hash->Buckets[Idx].Key) {
                ++Idx;
            }
        }

        TIteratorBase& operator++() {
            Next();
            return *this;
        }

        TIteratorBase& operator*() {
            // yes return ourself
            return *this;
        }

        bool Ok() const {
            return Idx < Hash->Buckets.size();
        }

        const TKey& Key() const {
            return Hash->Buckets[Idx].Key;
        }

        TVal& Value() const {
            return Hash->Buckets[Idx].Value;
        }

        THash* GetHash() {
            return Hash;
        }

        bool operator== (const TIteratorBase& rhs) const {
            Y_ASSERT(Hash == rhs.Hash);
            return Idx == rhs.Idx;
        }

        bool operator!= (const TIteratorBase& rhs) const {
            return !(*this == rhs);
        }
    };

public:
    typedef TIteratorBase<const TDenseHash, const TValue> TConstIterator;
    typedef TIteratorBase<TDenseHash, TValue> TIterator;

    TIterator begin() {
        return TIterator(*this);
    }

    TIterator end() {
        return TIterator(this, Buckets.size());
    }

    TConstIterator begin() const {
        return TConstIterator(*this);
    }

    TConstIterator end() const {
        return TConstIterator(this, Buckets.size());
    }

    template <typename K>
    Y_FORCE_INLINE TIterator Find(const K& key) {
        return ProcessBucket<TIterator>(
            key,
            [&](size_t idx) { return TIterator(this, idx); },
            [&](size_t) { return end(); }
        );
    }

    template <typename K>
    Y_FORCE_INLINE TConstIterator Find(const K& key) const {
        return ProcessBucket<TConstIterator>(
            key,
            [&](size_t idx) { return TConstIterator(this, idx); },
            [&](size_t) { return end(); }
        );
    }

    template <typename TIteratorType>
    Y_FORCE_INLINE void Insert(const TIteratorType& iterator) {
        GetMutable(iterator.Key()) = iterator.Value();
    }

    template <typename K, typename V>
    Y_FORCE_INLINE void Insert(const K& key, const V& value) {
        GetMutable(key) = value;
    }

    bool Grow(size_t to = 0, bool force = false) {
        if (!to) {
            to = Buckets.size() * 2;
        } else {
            to = FastClp2(to);
            if (to <= Buckets.size() && !force) {
                return false;
            }
        }
        yvector<TItem> oldBuckets(to, TItem(EmptyMarker, DefaultValue));
        oldBuckets.swap(Buckets);

        BucketMask = Buckets.size() - 1;
        GrowThreshold = Max<size_t>(1, Buckets.size() * (MaxLoadFactor / 100.f)) - 1;

        for (TItem& item : oldBuckets) {
            if (EmptyMarker != item.Key) {
                ProcessBucket<void>(
                    item.Key,
                    [&](size_t idx) { Buckets[idx] = std::move(item); },
                    [&](size_t idx) { Buckets[idx] = std::move(item); }
                );
            }
        }
        return true;
    }

private:
    struct TItem {
        TKey Key;
        TValue Value;

        TItem(const TKey& key = TKey(), const TValue& value = TValue())
            : Key(key)
            , Value(value)
        {
        }

        TItem (const TItem&) = default;
        TItem& operator= (const TItem&) = default;

        TItem& operator= (TItem&& rhs) {
            Key = std::move(rhs.Key);
            Value = std::move(rhs.Value);
            return *this;
        }

        Y_SAVELOAD_DEFINE(Key, Value);
    };

    size_t BucketMask;
    size_t NumFilled;
    size_t GrowThreshold;
    yvector<TItem> Buckets;

    TEmptyMarker EmptyMarker;
    TValue DefaultValue;

    template <typename K>
    Y_FORCE_INLINE TValue& GetMutableNoGrow(const K& key, bool* newValueWasInserted = nullptr) {
        return ProcessBucket<TValue&>(
            key,
            [&](size_t idx) -> TValue& {
                if (!!newValueWasInserted) {
                    *newValueWasInserted = false;
                }
                return Buckets[idx].Value;
            },
            [&](size_t idx) -> TValue& {
                ++NumFilled;
                Buckets[idx].Key = key;
                if (!!newValueWasInserted) {
                    *newValueWasInserted = true;
                }
                return Buckets[idx].Value;
            }
        );
    }

    Y_FORCE_INLINE bool MaybeGrow() {
        if (NumFilled < GrowThreshold) {
            return false;
        }
        Grow();
        return true;
    }

    template <typename K>
    Y_FORCE_INLINE size_t FindBucket(const K& key) const {
        return ProcessBucket<size_t>(
            key,
            [](size_t idx) { return idx; },
            [](size_t idx) { return idx; }
        );
    }

    template <typename TResult, typename TAnyKey, typename TOnFound, typename TOnEmpty>
    Y_FORCE_INLINE TResult ProcessBucket(const TAnyKey& key, const TOnFound& onFound, const TOnEmpty& onEmpty) const {
        size_t idx = TKeyHash()(key) & BucketMask;
        for (size_t numProbes = 1; EmptyMarker != Buckets[idx].Key; ++numProbes) {
            if (Buckets[idx].Key == key) {
                return onFound(idx);
            }
            idx = (idx + numProbes) & BucketMask;
        }
        return onEmpty(idx);
    }

    // Exact copy-paste of function above, but I don't know how to avoid it
    template <typename TResult, typename TAnyKey, typename TOnFound, typename TOnEmpty>
    Y_FORCE_INLINE TResult ProcessBucket(const TAnyKey& key, const TOnFound& onFound, const TOnEmpty& onEmpty) {
        size_t idx = TKeyHash()(key) & BucketMask;
        for (size_t numProbes = 1; EmptyMarker != Buckets[idx].Key; ++numProbes) {
            if (Buckets[idx].Key == key) {
                return onFound(idx);
            }
            idx = (idx + numProbes) & BucketMask;
        }
        return onEmpty(idx);
    }
};


template <typename TKey,
          typename TKeyHash = THash<TKey>,
          int MaxLoadFactor = 50,
          int LogInitSize = 8>
class TDenseHashSet {
public:
    TDenseHashSet(const TKey& emptyMarker = TKey(), size_t initSize=0)
        : EmptyMarker(emptyMarker)
    {
        MakeEmpty(initSize);
    }

    void Clear() {
        size_t currentSize = Buckets.size();
        Buckets.clear();
        Buckets.resize(currentSize, EmptyMarker);
        NumFilled = 0;
    }

    void MakeEmpty(size_t initSize=0) {
        if (!initSize) {
            initSize = 1 << LogInitSize;
        } else {
            initSize = FastClp2(initSize);
        }
        BucketMask = initSize - 1;
        NumFilled = 0;
        yvector<TKey>(initSize, EmptyMarker).swap(Buckets);
        GrowThreshold = Max<size_t>(1, initSize * MaxLoadFactor / 100) - 1;
    }

    template <typename K>
    bool Has(const K& key) const {
        return Buckets[FindBucket(key)] != EmptyMarker;
    }

    // gets existing item or inserts new
    template <typename K>
    bool Insert(const K& key) {
        bool inserted = InsertNoGrow(key);
        if (inserted) {
            MaybeGrow();
        }
        return inserted;
    }

    size_t Capacity() const {
        return Buckets.capacity();
    }

    bool Empty() const {
        return Size() == 0;
    }

    size_t Size() const {
        return NumFilled;
    }

    template <int maxFillPercents, int logInitSize>
    void Swap(TDenseHashSet<TKey, TKeyHash, maxFillPercents, logInitSize>& other) {
        Buckets.swap(other.Buckets);
        DoSwap(BucketMask, other.BucketMask);
        DoSwap(NumFilled, other.NumFilled);
        DoSwap(GrowThreshold, other.GrowThreshold);
        DoSwap(EmptyMarker, other.EmptyMarker);
    }

    Y_SAVELOAD_DEFINE(BucketMask, NumFilled, GrowThreshold, Buckets, EmptyMarker);

private:
    template <typename THash>
    class TIteratorBase {
        friend class TDenseHashSet;

        THash* Hash;
        size_t Idx;

        // used only to implement end()
        TIteratorBase(THash* hash, size_t initIdx)
            : Hash(hash)
            , Idx(initIdx)
        {
        }

    public:
        TIteratorBase(THash& hash)
            : Hash(&hash)
            , Idx(0)
        {
            if (Hash->Buckets[Idx] == Hash->EmptyMarker) {
                Next();
            }
        }

        void Next() {
            ++Idx;
            while (Idx < Hash->Buckets.size() && Hash->Buckets[Idx] == Hash->EmptyMarker) {
                ++Idx;
            }
        }

        TIteratorBase& operator++() {
            Next();
            return *this;
        }

        bool Initialized() const {
            return Hash != nullptr;
        }

        bool Ok() const {
            return Idx < Hash->Buckets.size();
        }

        const TKey& operator* () const {
            return Key();
        }

        const TKey& Key() const {
            return Hash->Buckets[Idx];
        }

        bool operator== (const TIteratorBase& rhs) const {
            Y_ASSERT(Hash == rhs.Hash);
            return Idx == rhs.Idx;
        }

        bool operator!= (const TIteratorBase& rhs) const {
            return !(*this == rhs);
        }
    };

public:
    typedef TIteratorBase<const TDenseHashSet> TConstIterator;

    TConstIterator begin() const {
        return TConstIterator(*this);
    }

    TConstIterator end() const {
        return TConstIterator(this, Buckets.size());
    }

private:
    size_t BucketMask;
    size_t NumFilled;
    size_t GrowThreshold;
    yvector<TKey> Buckets;

    TKey EmptyMarker;

    template <typename K>
    bool InsertNoGrow(const K& key) {
        size_t idx = FindBucket(key);
        if (Buckets[idx] == EmptyMarker) {
            ++NumFilled;
            Buckets[idx] = key;
            return true;
        }
        return false;
    }

    bool MaybeGrow() {
        if (NumFilled < GrowThreshold) {
            return false;
        }

        yvector<TKey> oldBuckets(Buckets.size() * 2, EmptyMarker);
        oldBuckets.swap(Buckets);

        BucketMask = Buckets.size() - 1;
        GrowThreshold = Max<size_t>(1, Buckets.size() * (MaxLoadFactor / 100.f)) - 1;

        NumFilled = 0;
        for (const TKey& key: oldBuckets) {
            if (key != EmptyMarker) {
                InsertNoGrow(key);
            }
        }

        return true;
    }

    template <typename K>
    size_t FindBucket(const K& key) const {
        size_t idx = TKeyHash()(key) & BucketMask;
        for (size_t numProbes = 1; Buckets[idx] != EmptyMarker; ++numProbes) {
            if (Buckets[idx] == key) {
                return idx;
            }
            idx = (idx + numProbes) & BucketMask;
        }
        return idx;
    }
};

template<i64 Value>
struct TConstIntEmptyMarker {
    template <typename T>
    operator T() const {
        return Value;
    }

    template <typename T>
    bool operator== (T rhs) const {
        return (T)Value == rhs;
    }

    template <typename T>
    bool operator!= (T rhs) const {
        return (T)Value != rhs;
    }
};
