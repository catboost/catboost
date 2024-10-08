#pragma once

#include "fwd.h"

#include <util/generic/bitops.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/mapfindptr.h>

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
          class TKeyHash,
          size_t MaxLoadFactor,
          size_t LogInitSize>
class TDenseHash : public TMapOps<TDenseHash<TKey, TValue, TKeyHash, MaxLoadFactor, LogInitSize>> {
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

        TIteratorBase(const TIteratorBase&) = default;

        static TIteratorBase CreateEmpty() {
            return TIteratorBase(nullptr, 0);
        }

        TIteratorBase& operator=(const TIteratorBase&) = default;

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

        const TVal* operator->() const {
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
    TDenseHash(TDenseHash&&) = default;

    TDenseHash& operator=(const TDenseHash& rhs) {
        TDenseHash tmp{ rhs };
        return *this = std::move(tmp);
    }

    TDenseHash& operator=(TDenseHash&&) = default;

    friend bool operator==(const TDenseHash& lhs, const TDenseHash& rhs) {
        return lhs.Size() == rhs.Size() &&
            AllOf(lhs, [&rhs](const auto& v) {
                auto it = rhs.find(v.first);
                return it != rhs.end() && *it == v;
            });
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
    bool Has(const K& key) const {
        return ProcessKey<bool>(
            key,
            [](size_type) { return true; },
            [](size_type) { return false; });
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
        // TODO(tender-bum): make SaveLoad great again
        ::SaveMany(s, BucketMask, NumFilled, GrowThreshold);
        // We need to do so because Buckets may be serialized as a pod-array
        // that doesn't correspond to the previous behaviour
        ::SaveSize(s, Buckets.size());
        for (const auto& b : Buckets) {
            ::Save(s, b.first);
            ::Save(s, b.second);
        }
        mapped_type defaultValue{};
        ::SaveMany(s, EmptyMarker, defaultValue);
    }

    void Load(IInputStream* s) {
        // TODO(tender-bum): make SaveLoad great again
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
    iterator find(const K& key) {
        return ProcessKey<iterator>(
            key,
            [&](size_type idx) { return iterator(this, idx); },
            [&](size_type) { return end(); });
    }

    template <class K>
    const_iterator find(const K& key) const {
        return ProcessKey<const_iterator>(
            key,
            [&](size_type idx) { return const_iterator(this, idx); },
            [&](size_type) { return end(); });
    }

    template <class K>
    const TValue& at(const K& key) const {
        return ProcessKey<const TValue&>(
            key,
            [&](size_type idx) -> const TValue& { return Buckets[idx].second; },
            [&](size_type) -> const TValue& { throw std::out_of_range("TDenseHash: missing key"); });
    }

    template <class K>
    TValue& at(const K& key) {
        return ProcessKey<TValue&>(
            key,
            [&](size_type idx) -> TValue& { return Buckets[idx].second; },
            [&](size_type) -> TValue& { throw std::out_of_range("TDenseHash: missing key"); });
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
        TVector<value_type> oldBuckets(Reserve(to));
        for (size_type i = 0; i < to; ++i) {
            oldBuckets.emplace_back(EmptyMarker, mapped_type{});
        }
        oldBuckets.swap(Buckets);

        BucketMask = Buckets.size() - 1;
        GrowThreshold = Max<size_type>(1, Buckets.size() * (MaxLoadFactor / 100.f)) - 1;

        for (auto& item : oldBuckets) {
            if (EmptyMarker != item.first) {
                SetValue(FindProperBucket(item.first), std::move(item));
            }
        }
        return true;
    }

    // Grow to size with which GrowThreshold will be higher then passed value
    //
    // (to) = (desired_num_filled + 2) * (100.f / MaxLoadFactor) + 2 after conversion to size_type
    // is not less than x := (desired_num_filled + 2) * (100.f / MaxLoadFactor) + 1 and FastClp2(to) is not less that (to)
    // (to) * (MaxLoadFactor / 100.f) >= x * (MaxLoadFactor / 100.f) = (desired_num_filled + 2) + (MaxLoadFactor / 100.f).
    // This require calculations with two or more significand decimal places
    // to have no less than (desired_num_filled + 2) after second conversion to size_type.
    // In that case after substracting 1 we got GrowThreshold >= desired_num_filled + 1
    //
    bool ReserveSpace(size_type desired_num_filled, bool force = false) {
        size_type to = Max<size_type>(1, (desired_num_filled + 2) * (100.f / MaxLoadFactor) + 2);
        return Grow(to, force);
    }

    // We need this overload because we want to optimize insertion when somebody inserts value_type.
    // So we don't need to extract the key.
    // This overload also allows brace enclosed initializer to be inserted.
    std::pair<iterator, bool> insert(const value_type& t) {
        size_type hs = hasher{}(t.first);
        auto p = GetBucketInfo(hs & BucketMask, t.first);
        if (p.second) {
            ++NumFilled;
            if (NumFilled >= GrowThreshold) {
                Grow();
                p.first = FindProperBucket(hs & BucketMask, t.first);
            }
            SetValue(p.first, t);
            return { iterator{ this, p.first }, true };
        }
        return { iterator{ this, p.first }, false };
    }

    // We need this overload because we want to optimize insertion when somebody inserts value_type.
    // So we don't need to extract the key.
    // This overload also allows brace enclosed initializer to be inserted.
    std::pair<iterator, bool> insert(value_type&& t) {
        size_type hs = hasher{}(t.first);
        auto p = GetBucketInfo(hs & BucketMask, t.first);
        if (p.second) {
            ++NumFilled;
            if (NumFilled >= GrowThreshold) {
                Grow();
                p.first = FindProperBucket(hs & BucketMask, t.first);
            }
            SetValue(p.first, std::move(t));
            return { iterator{ this, p.first }, true };
        }
        return { iterator{ this, p.first }, false };
    }

    // Standart integration. This overload is equivalent to emplace(std::forward<P>(p)).
    template <class P>
    std::enable_if_t<!std::is_same<std::decay_t<P>, value_type>::value,
    std::pair<iterator, bool>> insert(P&& p) {
        return emplace(std::forward<P>(p));
    }

    // Not really emplace because we need to know the key anyway. So we need to construct value_type.
    template <class... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        return insert(value_type{ std::forward<Args>(args)... });
    }

    template <class K>
    mapped_type& operator[](K&& key) {
        size_type hs = hasher{}(key);
        auto p = GetBucketInfo(hs & BucketMask, key);
        if (p.second) {
            ++NumFilled;
            if (NumFilled >= GrowThreshold) {
                Grow();
                p.first = FindProperBucket(hs & BucketMask, key);
            }
            SetValue(p.first, std::forward<K>(key), mapped_type{});
        }
        return Buckets[p.first].second;
    }

private:
    key_type EmptyMarker;
    size_type NumFilled;
    size_type BucketMask;
    size_type GrowThreshold;
    TVector<value_type> Buckets;

private:
    // Tricky way to set value of type with const fields
    template <class... Args>
    void SetValue(value_type& bucket, Args&&... args) {
        bucket.~value_type();
        new (&bucket) value_type(std::forward<Args>(args)...);
    }

    template <class... Args>
    void SetValue(size_type idx, Args&&... args) {
        SetValue(Buckets[idx], std::forward<Args>(args)...);
    }

    template <class K>
    size_type FindProperBucket(size_type idx, const K& key) const {
        return ProcessIndex<size_type>(
            idx,
            key,
            [](size_type idx) { return idx; },
            [](size_type idx) { return idx; });
    }

    template <class K>
    size_type FindProperBucket(const K& key) const {
        return FindProperBucket(hasher{}(key) & BucketMask, key);
    }

    // { idx, is_empty }
    template <class K>
    std::pair<size_type, bool> GetBucketInfo(size_type idx, const K& key) const {
        return ProcessIndex<std::pair<size_type, bool>>(
            idx,
            key,
            [](size_type idx) { return std::make_pair(idx, false); },
            [](size_type idx) { return std::make_pair(idx, true); });
    }

    template <class R, class K, class OnFound, class OnEmpty>
    R ProcessIndex(size_type idx, const K& key, OnFound f0, OnEmpty f1) const {
        for (size_type numProbes = 1; EmptyMarker != Buckets[idx].first; ++numProbes) {
            if (Buckets[idx].first == key) {
                return f0(idx);
            }
            idx = (idx + numProbes) & BucketMask;
        }
        return f1(idx);
    }

    template <class R, class K, class OnFound, class OnEmpty>
    R ProcessKey(const K& key, OnFound&& f0, OnEmpty&& f1) const {
        return ProcessIndex<R>(
            hasher{}(key) & BucketMask, key, std::forward<OnFound>(f0), std::forward<OnEmpty>(f1));
    }
};

template <class TKey,
          class TKeyHash,
          size_t MaxLoadFactor,
          size_t LogInitSize>
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
