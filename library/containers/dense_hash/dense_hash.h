#pragma once

#include <util/generic/utility.h>
#include <util/generic/vector.h>

#include <util/str_stl.h>
#include <util/ysaveload.h>

/*
 * There are 2 classes in this file:
 *   - TDenseHash - analog of THashMap
 *   - TDenseHashSet - analog of THashSet
 */

/*
 * Implements dense-hash, in some circumstances it is a lot (2x) faster than THashMap.
 * We support only adding new elements.
 * TKey value equal to EmptyMarker (by default, it is TKey())
 * can not be inserted into hash - it is used as marker of empty element.
 * TValue type must be default constructible
 */

template <class TKey,
          class TValue,
          class TKeyHash = THash<TKey>,
          size_t MaxLoadFactor = 50, // in percents
          size_t LogInitSize = 8>
class TDenseHash {
private:
    template <class THash, class TVal>
    class TIteratorBase {
        friend class TDenseHash;

        template <class THash2, class TVal2>
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
            if (Hash->EmptyMarker == Hash->Buckets[Idx].first) {
                Next();
            }
        }

        template <class THash2, class TVal2>
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
            while (Idx < Hash->Buckets.size() && Hash->EmptyMarker == Hash->Buckets[Idx].first) {
                ++Idx;
            }
        }

        TIteratorBase& operator++() {
            Next();
            return *this;
        }

        TVal& operator*() {
            return Hash->Buckets[Idx];
        }

        TVal* operator->() {
            return &Hash->Buckets[Idx];
        }

        THash* GetHash() {
            return Hash;
        }

        bool operator==(const TIteratorBase& rhs) const {
            Y_ASSERT(Hash == rhs.Hash);
            return Idx == rhs.Idx;
        }

        bool operator!=(const TIteratorBase& rhs) const {
            return !(*this == rhs);
        }
    };

public:
    using key_type = TKey;
    using mapped_type = TValue;
    using value_type = std::pair<const key_type, mapped_type>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using hasher = TKeyHash;
    using key_equal = std::equal_to<key_type>; // TODO(tender-bum): template argument
    // using allocator_type = ...
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*; // TODO(tender-bum): std::allocator_traits<Alloc>::pointer;
    using const_pointer = const value_type*; // TODO(tender-bum):
                                             // std::allocator_traits<Alloc>::const_pointer;
    using iterator = TIteratorBase<TDenseHash, value_type>;
    using const_iterator = TIteratorBase<const TDenseHash, const value_type>;

public:
    TDenseHash(const key_type& emptyMarker = key_type{}, size_type initSize = 0)
        : EmptyMarker(emptyMarker)
    {
        MakeEmpty(initSize);
    }

    TDenseHash(const TDenseHash&) = default;

    TDenseHash(TDenseHash&& init) {
        Swap(init);
    }

    TDenseHash& operator=(const TDenseHash& rhs) {
        EmptyMarker = rhs.EmptyMarker;
        NumFilled = rhs.EmptyMarker;
        BucketMask = rhs.BucketMask;
        GrowThreshold = rhs.GrowThreshold;
        Buckets.clear();
        for (const auto& b : rhs.Buckets) {
            Buckets.emplace_back(b.first, b.second);
        }
        return *this;
    }

    friend bool operator==(const TDenseHash& lhs, const TDenseHash& rhs) {
        return lhs.Size() == rhs.Size() &&
            AllOf(lhs, [&rhs](const auto& v) { return rhs.Has(v.first); });
    }

    void Clear() {
        for (auto& bucket : Buckets) {
            if (bucket.first != EmptyMarker) {
                SetValue(bucket, EmptyMarker, mapped_type{});
            }
        }
        NumFilled = 0;
    }

    void MakeEmpty(size_type initSize = 0) {
        if (!initSize) {
            initSize = 1 << LogInitSize;
        } else {
            initSize = FastClp2(initSize);
        }
        BucketMask = initSize - 1;
        NumFilled = 0;
        TVector<value_type> tmp;
        for (size_type i = 0; i < initSize; ++i) {
            tmp.emplace_back(EmptyMarker, mapped_type{});
        }
        tmp.swap(Buckets);
        GrowThreshold = Max<size_type>(1, initSize * MaxLoadFactor / 100) - 1;
    }

    template <class K>
    mapped_type* FindPtr(const K& key) {
        return ProcessBucket<mapped_type*>(
            key,
            [&](size_type idx) { return &Buckets[idx].second; },
            [](size_type) { return nullptr; });
    }

    template <class K>
    const mapped_type* FindPtr(const K& key) const {
        return ProcessBucket<const mapped_type*>(
            key,
            [&](size_type idx) { return &Buckets[idx].second; },
            [](size_type) { return nullptr; });
    }

    template <class K>
    bool Has(const K& key) const {
        return ProcessBucket<bool>(
            key,
            [](size_type) { return true; },
            [](size_type) { return false; });
    }

    template <class K>
    mapped_type Get(const K& key) const {
        return ProcessBucket<mapped_type>(
            key,
            [&](size_type idx) -> mapped_type { return Buckets[idx].second; },
            [&](size_type) -> mapped_type { return mapped_type{}; });
    }

    // TODO(tender-bum) remove this
    template <class K>
    const mapped_type& GetRef(const K& key, const mapped_type& alternative) const {
        return ProcessBucket<const mapped_type&>(
            key,
            [&](size_type idx) -> const mapped_type& { return Buckets[idx].second; },
            [&](size_type) -> const mapped_type& { return alternative; });
    }

    template <class K>
    const mapped_type& GetRef(const K& key, mapped_type&& alternative) const = delete;

    // gets existing item or inserts new
    template <class K>
    mapped_type& GetMutable(const K& key, bool* newValueWasInserted = nullptr) {
        bool newValueWasInsertedInternal;
        mapped_type* res = &GetMutableNoGrow(key, &newValueWasInsertedInternal);
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
    template <class K>
    mapped_type& UnsafeGetMutableNoGrow(const K& key, bool* newValueWasInserted = nullptr) {
        return GetMutableNoGrow(key, newValueWasInserted);
    }

    size_type Capacity() const {
        return Buckets.capacity();
    }

    bool Empty() const {
        return Size() == 0;
    }

    size_type Size() const {
        return NumFilled;
    }

    template <size_type maxFillPercents, size_type logInitSize>
    void Swap(TDenseHash<key_type, mapped_type, hasher, maxFillPercents, logInitSize>& other) {
        Buckets.swap(other.Buckets);
        DoSwap(BucketMask, other.BucketMask);
        DoSwap(NumFilled, other.NumFilled);
        DoSwap(GrowThreshold, other.GrowThreshold);
        DoSwap(EmptyMarker, other.EmptyMarker);
    }

    void Save(IOutputStream* s) const {
        ::SaveMany(s, BucketMask, NumFilled, GrowThreshold);
        ::SaveSize(s, Buckets.size());
        for (const auto& b : Buckets) {
            ::Save(s, b.first);
            ::Save(s, b.second);
        }
        mapped_type defaultValue;
        ::SaveMany(s, EmptyMarker, defaultValue);
    }

    void Load(IInputStream* s) {
        ::LoadMany(s, BucketMask, NumFilled, GrowThreshold);
        // We need to do so because we can't load const fields
        struct TPairMimic {
            key_type First;
            mapped_type Second;
            Y_SAVELOAD_DEFINE(First, Second);
        };
        TVector<TPairMimic> tmp;
        ::Load(s, tmp);
        Buckets.clear();
        for (auto& v : tmp) {
            Buckets.emplace_back(std::move(v.First), std::move(v.Second));
        }
        ::Load(s, EmptyMarker);
        mapped_type defaultValue;
        ::Load(s, defaultValue);
    }

public:
    iterator begin() {
        return iterator(*this);
    }

    iterator end() {
        return iterator(this, Buckets.size());
    }

    const_iterator begin() const {
        return const_iterator(*this);
    }

    const_iterator end() const {
        return const_iterator(this, Buckets.size());
    }

    template <class K>
    iterator Find(const K& key) {
        return ProcessBucket<iterator>(
            key,
            [&](size_type idx) { return iterator(this, idx); },
            [&](size_type) { return end(); });
    }

    template <class K>
    const_iterator Find(const K& key) const {
        return ProcessBucket<const_iterator>(
            key,
            [&](size_type idx) { return const_iterator(this, idx); },
            [&](size_type) { return end(); });
    }

    template <class TIteratorType>
    void Insert(const TIteratorType& iterator) {
        GetMutable(iterator.first) = iterator.second;
    }

    template <class K, class V>
    void Insert(const K& key, const V& value) {
        GetMutable(key) = value;
    }

    bool Grow(size_type to = 0, bool force = false) {
        if (!to) {
            to = Buckets.size() * 2;
        } else {
            to = FastClp2(to);
            if (to <= Buckets.size() && !force) {
                return false;
            }
        }
        TVector<value_type> oldBuckets;
        for (size_type i = 0; i < to; ++i) {
            oldBuckets.emplace_back(EmptyMarker, mapped_type{});
        }
        oldBuckets.swap(Buckets);

        BucketMask = Buckets.size() - 1;
        GrowThreshold = Max<size_type>(1, Buckets.size() * (MaxLoadFactor / 100.f)) - 1;

        for (auto& item : oldBuckets) {
            if (EmptyMarker != item.first) {
                ProcessBucket<void>(
                    item.first,
                    [&](size_type) { Y_FAIL(); },
                    [&](size_type idx) { SetValue(Buckets[idx], std::move(item)); });
            }
        }
        return true;
    }

private:
    key_type EmptyMarker;
    size_type NumFilled;
    size_type BucketMask;
    size_type GrowThreshold;
    TVector<value_type> Buckets;

private:
    template <class... Args>
    void SetValue(value_type& bucket, Args&&... args) {
        // Tricky way to set value of type with const fields
        bucket.~value_type();
        new (&bucket) value_type(std::forward<Args>(args)...);
    }

    template <class K>
    mapped_type& GetMutableNoGrow(const K& key, bool* newValueWasInserted = nullptr) {
        return ProcessBucket<mapped_type&>(
            key,
            [&](size_type idx) -> mapped_type& {
                if (!!newValueWasInserted) {
                    *newValueWasInserted = false;
                }
                return Buckets[idx].second;
            },
            [&](size_type idx) -> mapped_type& {
                ++NumFilled;
                SetValue(Buckets[idx], key, mapped_type{});
                if (!!newValueWasInserted) {
                    *newValueWasInserted = true;
                }
                return Buckets[idx].second;
            });
    }

    bool MaybeGrow() {
        if (NumFilled < GrowThreshold) {
            return false;
        }
        Grow();
        return true;
    }

    template <class K>
    size_type FindBucket(const K& key) const {
        return ProcessBucket<size_type>(
            key,
            [](size_type idx) { return idx; },
            [](size_type idx) { return idx; });
    }

    template <class TResult, class TAnyKey, class TOnFound, class TOnEmpty>
    TResult ProcessBucket(const TAnyKey& key, const TOnFound& onFound, const TOnEmpty& onEmpty) const {
        size_type idx = hasher{}(key) & BucketMask;
        for (size_type numProbes = 1; EmptyMarker != Buckets[idx].first; ++numProbes) {
            if (Buckets[idx].first == key) {
                return onFound(idx);
            }
            idx = (idx + numProbes) & BucketMask;
        }
        return onEmpty(idx);
    }

    // Exact copy-paste of function above, but I don't know how to avoid it
    template <class TResult, class TAnyKey, class TOnFound, class TOnEmpty>
    TResult ProcessBucket(const TAnyKey& key, const TOnFound& onFound, const TOnEmpty& onEmpty) {
        size_type idx = hasher{}(key) & BucketMask;
        for (size_type numProbes = 1; EmptyMarker != Buckets[idx].first; ++numProbes) {
            if (Buckets[idx].first == key) {
                return onFound(idx);
            }
            idx = (idx + numProbes) & BucketMask;
        }
        return onEmpty(idx);
    }
};

template <class TKey,
          class TKeyHash = THash<TKey>,
          size_t MaxLoadFactor = 50,
          size_t LogInitSize = 8>
class TDenseHashSet {
public:
    TDenseHashSet(const TKey& emptyMarker = TKey(), size_t initSize = 0)
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

    void MakeEmpty(size_t initSize = 0) {
        if (!initSize) {
            initSize = 1 << LogInitSize;
        } else {
            initSize = FastClp2(initSize);
        }
        BucketMask = initSize - 1;
        NumFilled = 0;
        TVector<TKey>(initSize, EmptyMarker).swap(Buckets);
        GrowThreshold = Max<size_t>(1, initSize * MaxLoadFactor / 100) - 1;
    }

    template <class K>
    bool Has(const K& key) const {
        return Buckets[FindBucket(key)] != EmptyMarker;
    }

    // gets existing item or inserts new
    template <class K>
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

    template <size_t maxFillPercents, size_t logInitSize>
    void Swap(TDenseHashSet<TKey, TKeyHash, maxFillPercents, logInitSize>& other) {
        Buckets.swap(other.Buckets);
        DoSwap(BucketMask, other.BucketMask);
        DoSwap(NumFilled, other.NumFilled);
        DoSwap(GrowThreshold, other.GrowThreshold);
        DoSwap(EmptyMarker, other.EmptyMarker);
    }

    Y_SAVELOAD_DEFINE(BucketMask, NumFilled, GrowThreshold, Buckets, EmptyMarker);

private:
    template <class THash>
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

        const TKey& operator*() const {
            return Key();
        }

        const TKey& Key() const {
            return Hash->Buckets[Idx];
        }

        bool operator==(const TIteratorBase& rhs) const {
            Y_ASSERT(Hash == rhs.Hash);
            return Idx == rhs.Idx;
        }

        bool operator!=(const TIteratorBase& rhs) const {
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
    TVector<TKey> Buckets;

    TKey EmptyMarker;

    template <class K>
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

        TVector<TKey> oldBuckets(Buckets.size() * 2, EmptyMarker);
        oldBuckets.swap(Buckets);

        BucketMask = Buckets.size() - 1;
        GrowThreshold = Max<size_t>(1, Buckets.size() * (MaxLoadFactor / 100.f)) - 1;

        NumFilled = 0;
        for (const TKey& key : oldBuckets) {
            if (key != EmptyMarker) {
                InsertNoGrow(key);
            }
        }

        return true;
    }

    template <class K>
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
