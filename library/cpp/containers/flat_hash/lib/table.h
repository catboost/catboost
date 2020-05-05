#pragma once

#include "iterator.h"
#include "concepts/container.h"
#include "concepts/size_fitter.h"

#include <util/generic/utility.h>

#include <functional>

namespace NFlatHash {

namespace NPrivate {

template <class T>
struct TTypeIdentity { using type = T; };

}  // namespace NPrivate

template <
    class Hash,
    class KeyEqual,
    class Container,
    class KeyGetter,
    class Probing,
    class SizeFitter,
    class Expander,
    // Used in the TSet to make iterator behave as const_iterator
    template <class> class IteratorModifier = NPrivate::TTypeIdentity>
class TTable {
private:
    static_assert(NConcepts::ContainerV<Container>);
    static_assert(NConcepts::SizeFitterV<SizeFitter>);

    template <class C, class V>
    class TIteratorImpl : public TIterator<C, V> {
    private:
        using TBase = TIterator<C, V>;
        friend class TTable;

        using TBase::TBase;

    public:
        TIteratorImpl() : TBase(nullptr, 0) {}
    };

public:
    using value_type = typename Container::value_type;
    using size_type = typename Container::size_type;
    using difference_type = typename Container::difference_type;
    using hasher = Hash;
    using key_equal = KeyEqual;

    using reference = value_type&;
    using const_reference = const value_type&;

    using iterator = TIteratorImpl<typename IteratorModifier<Container>::type,
                                   typename IteratorModifier<value_type>::type>;
    using const_iterator = TIteratorImpl<const Container, const value_type>;
    using allocator_type = typename Container::allocator_type;
    using pointer = typename Container::pointer;
    using const_pointer = typename Container::const_pointer;

private:
    TTable(Container buckets)
        : Buckets_(std::move(buckets))
    {
        SizeFitter_.Update(bucket_count());
    }

    static constexpr size_type INIT_SIZE = 8;

public:
    template <class... Rest>
    TTable(size_type initSize, Rest&&... rest)
        : Buckets_(initSize == 0 ? INIT_SIZE : SizeFitter_.EvalSize(initSize),
                   std::forward<Rest>(rest)...)
    {
        SizeFitter_.Update(bucket_count());
    }

    TTable(const TTable&) = default;
    TTable(TTable&& rhs)
        : SizeFitter_(std::move(rhs.SizeFitter_))
        , Buckets_(std::move(rhs.Buckets_))
        , Hasher_(std::move(rhs.Hasher_))
        , KeyEqual_(std::move(rhs.KeyEqual_))
    {
        TTable tmp{ Buckets_.Clone(INIT_SIZE) };
        tmp.swap(rhs);
    }

    TTable& operator=(const TTable&) = default;
    TTable& operator=(TTable&& rhs) {
        TTable tmp(std::move(rhs));
        swap(tmp);
        return *this;
    }

    // Iterators
    iterator begin() { return &Buckets_; }
    const_iterator begin() const { return const_cast<TTable*>(this)->begin(); }
    const_iterator cbegin() const { return begin(); }

    iterator end() { return { &Buckets_, bucket_count() }; }
    const_iterator end() const { return const_cast<TTable*>(this)->end(); }
    const_iterator cend() const { return end(); }

    // Capacity
    bool empty() const noexcept { return size() == 0; }
    size_type size() const noexcept { return Buckets_.Taken(); }

    // Modifiers
    void clear() {
        Container tmp(Buckets_.Clone(bucket_count()));
        Buckets_.Swap(tmp);
    }

    std::pair<iterator, bool> insert(const value_type& value) { return InsertImpl(value); }
    std::pair<iterator, bool> insert(value_type&& value) { return InsertImpl(std::move(value)); }

    template <class T>
    std::enable_if_t<!std::is_same_v<std::decay_t<T>, value_type>,
    std::pair<iterator, bool>> insert(T&& value) {
        return insert(value_type(std::forward<T>(value)));
    }

    iterator insert(const_iterator, const value_type& value) { // TODO(tender-bum)
        return insert(value).first;
    }
    iterator insert(const_iterator, value_type&& value) { // TODO(tender-bum)
        return insert(std::move(value)).first;
    }

    template <class T>
    iterator insert(const_iterator, T&& value) { // TODO(tender-bum)
        return insert(value_type(std::forward<T>(value))).first;
    }

    template <class InputIt>
    void insert(InputIt first, InputIt last) {
        while (first != last) {
            insert(*first++);
        }
    }

    void insert(std::initializer_list<value_type> il) {
        insert(il.begin(), il.end());
    }

    template <class... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        return insert(value_type(std::forward<Args>(args)...));
    }

    template <class... Args>
    iterator emplace_hint(const_iterator, Args&&... args) { // TODO(tender-bum)
        return emplace(std::forward<Args>(args)...).first;
    }

    void erase(const_iterator pos) {
        static_assert(NConcepts::RemovalContainerV<Container>,
                      "That kind of table doesn't allow erasing. Use another table instead.");
        if constexpr (NConcepts::RemovalContainerV<Container>) {
            Buckets_.DeleteNode(pos.Idx_);
        }
    }

    void erase(const_iterator f, const_iterator l) {
        while (f != l) {
            auto nxt = f;
            ++nxt;
            erase(f);
            f = nxt;
        }
    }

    template <class K>
    std::enable_if_t<!std::is_convertible_v<K, iterator> && !std::is_convertible_v<K, const_iterator>,
    size_type> erase(const K& key) {
        auto it = find(key);
        if (it != end()) {
            erase(it);
            return 1;
        }
        return 0;
    }

    void swap(TTable& rhs)
    noexcept(noexcept(std::declval<Container>().Swap(std::declval<Container&>())))
    {
        DoSwap(SizeFitter_, rhs.SizeFitter_);
        Buckets_.Swap(rhs.Buckets_);
        DoSwap(Hasher_, rhs.Hasher_);
        DoSwap(KeyEqual_, rhs.KeyEqual_);
    }

    // Lookup
    template <class K>
    size_type count(const K& key) const { return contains(key); }

    template <class K>
    iterator find(const K& key) {
        size_type hs = hash_function()(key);
        auto idx = FindProperBucket(hs, key);
        if (Buckets_.IsTaken(idx)) {
            return { &Buckets_, idx };
        }
        return end();
    }

    template <class K>
    const_iterator find(const K& key) const { return const_cast<TTable*>(this)->find(key); }

    template <class K>
    bool contains(const K& key) const {
        size_type hs = hash_function()(key);
        return Buckets_.IsTaken(FindProperBucket(hs, key));
    }

    // Bucket interface
    size_type bucket_count() const noexcept { return Buckets_.Size(); }
    size_type bucket_size(size_type idx) const { return Buckets_.IsTaken(idx); }

    // Hash policy
    float load_factor() const noexcept {
        return (float)(bucket_count() - Buckets_.Empty()) / bucket_count();
    }

    void rehash(size_type sz) {
        if (sz != 0) {
            auto newBuckets = SizeFitter_.EvalSize(sz);
            size_type occupied = bucket_count() - Buckets_.Empty();
            if (Expander::NeedGrow(occupied, newBuckets)) {
                newBuckets = Max(newBuckets, SizeFitter_.EvalSize(Expander::SuitableSize(size())));
            }
            RehashImpl(newBuckets);
        } else {
            RehashImpl(SizeFitter_.EvalSize(Expander::SuitableSize(size())));
        }
    }

    void reserve(size_type sz) { rehash(sz); } // TODO(tender-bum)

    // Observers
    constexpr auto hash_function() const noexcept { return Hasher_; }
    constexpr auto key_eq() const noexcept { return KeyEqual_; }

public:
    template <class T>
    std::pair<iterator, bool> InsertImpl(T&& value) {
        return TryCreate(KeyGetter::Apply(value), [&](size_type idx) {
            Buckets_.InitNode(idx, std::forward<T>(value));
        });
    }

    template <class T, class F>
    Y_FORCE_INLINE std::pair<iterator, bool> TryCreate(const T& key, F nodeInit) {
        size_type hs = hash_function()(key);
        size_type idx = FindProperBucket(hs, key);
        if (!Buckets_.IsTaken(idx)) {
            if (Expander::WillNeedGrow(bucket_count() - Buckets_.Empty(), bucket_count())) {
                RehashImpl();
                idx = FindProperBucket(hs, key);
            }
            nodeInit(idx);
            return { iterator{ &Buckets_, idx }, true };
        }
        return { iterator{ &Buckets_, idx }, false };
    }

    template <class K>
    size_type FindProperBucket(size_type hs, const K& key) const {
        return Probing::FindBucket(SizeFitter_, hs, bucket_count(), [&](size_type idx) {
            if constexpr (NConcepts::RemovalContainerV<Container>) {
                return Buckets_.IsEmpty(idx) ||
                       Buckets_.IsTaken(idx) && key_eq()(KeyGetter::Apply(Buckets_.Node(idx)), key);
            } else {
                return Buckets_.IsEmpty(idx) || key_eq()(KeyGetter::Apply(Buckets_.Node(idx)), key);
            }
        });
    }

    void RehashImpl() {
        if constexpr (NConcepts::RemovalContainerV<Container>) {
            size_type occupied = bucket_count() - Buckets_.Empty();
            if (size() < occupied / 2) {
                rehash(bucket_count()); // Just clearing all deleted elements
            } else {
                RehashImpl(SizeFitter_.EvalSize(Expander::EvalNewSize(bucket_count())));
            }
        } else {
            RehashImpl(SizeFitter_.EvalSize(Expander::EvalNewSize(bucket_count())));
        }
    }

    void RehashImpl(size_type newSize) {
        TTable tmp = Buckets_.Clone(newSize);
        for (auto& value : *this) {
            size_type hs = hash_function()(KeyGetter::Apply(value));
            tmp.Buckets_.InitNode(
                tmp.FindProperBucket(hs, KeyGetter::Apply(value)), std::move_if_noexcept(value));
        }
        swap(tmp);
    }

public:
    SizeFitter SizeFitter_;
    Container Buckets_;
    hasher Hasher_;
    key_equal KeyEqual_;
};

}  // namespace NFlatHash
