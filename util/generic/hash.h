#pragma once

#include "fwd.h"

#include "hash_table.h"

template <class Key, class T, class HashFcn, class EqualKey, class Alloc>
class THashMap: public TMapOps<THashMap<Key, T, HashFcn, EqualKey, Alloc>> {
private:
    using ht = THashTable<std::pair<const Key, T>, Key, HashFcn, TSelect1st, EqualKey, Alloc>;
    ht rep;

public:
    using key_type = typename ht::key_type;
    using value_type = typename ht::value_type;
    using hasher = typename ht::hasher;
    using key_equal = typename ht::key_equal;
    using allocator_type = typename ht::allocator_type;
    using node_allocator_type = typename ht::node_allocator_type;
    using mapped_type = T;

    using size_type = typename ht::size_type;
    using difference_type = typename ht::difference_type;
    using pointer = typename ht::pointer;
    using const_pointer = typename ht::const_pointer;
    using reference = typename ht::reference;
    using const_reference = typename ht::const_reference;

    using iterator = typename ht::iterator;
    using const_iterator = typename ht::const_iterator;
    using insert_ctx = typename ht::insert_ctx;

    hasher hash_function() const {
        return rep.hash_function();
    }
    key_equal key_eq() const {
        return rep.key_eq();
    }

public:
    THashMap()
        : rep(0, hasher(), key_equal())
    {
    }
    template <class TAllocParam>
    explicit THashMap(TAllocParam* allocParam, size_type n = 0)
        : rep(n, hasher(), key_equal(), allocParam)
    {
    }
    explicit THashMap(size_type n)
        : rep(n, hasher(), key_equal())
    {
    }
    THashMap(size_type n, const hasher& hf)
        : rep(n, hf, key_equal())
    {
    }
    THashMap(size_type n, const hasher& hf, const key_equal& eql)
        : rep(n, hf, eql)
    {
    }
    template <class TAllocParam>
    explicit THashMap(size_type n, TAllocParam* allocParam)
        : rep(n, hasher(), key_equal(), allocParam)
    {
    }
    template <class InputIterator>
    THashMap(InputIterator f, InputIterator l)
        : rep(0, hasher(), key_equal())
    {
        rep.insert_unique(f, l);
    }
    template <class InputIterator>
    THashMap(InputIterator f, InputIterator l, size_type n)
        : rep(n, hasher(), key_equal())
    {
        rep.insert_unique(f, l);
    }
    template <class InputIterator>
    THashMap(InputIterator f, InputIterator l, size_type n,
             const hasher& hf)
        : rep(n, hf, key_equal())
    {
        rep.insert_unique(f, l);
    }
    template <class InputIterator>
    THashMap(InputIterator f, InputIterator l, size_type n,
             const hasher& hf, const key_equal& eql)
        : rep(n, hf, eql)
    {
        rep.insert_unique(f, l);
    }

    THashMap(const std::initializer_list<std::pair<Key, T>>& list)
        : rep(list.size(), hasher(), key_equal())
    {
        for (const auto& v : list) {
            rep.insert_unique_noresize(v);
        }
    }

    // THashMap has implicit copy/move constructors and copy-/move-assignment operators
    // because its implementation is backed by THashTable.
    // See hash_ut.cpp

public:
    size_type size() const noexcept {
        return rep.size();
    }
    yssize_t ysize() const noexcept {
        return (yssize_t)rep.size();
    }
    size_type max_size() const noexcept {
        return rep.max_size();
    }

    Y_PURE_FUNCTION bool empty() const noexcept {
        return rep.empty();
    }
    explicit operator bool() const noexcept {
        return !empty();
    }
    void swap(THashMap& hs) {
        rep.swap(hs.rep);
    }

    iterator begin() {
        return rep.begin();
    }
    iterator end() {
        return rep.end();
    }
    const_iterator begin() const {
        return rep.begin();
    }
    const_iterator end() const {
        return rep.end();
    }
    const_iterator cbegin() const {
        return rep.begin();
    }
    const_iterator cend() const {
        return rep.end();
    }

public:
    template <class InputIterator>
    void insert(InputIterator f, InputIterator l) {
        rep.insert_unique(f, l);
    }

    std::pair<iterator, bool> insert(const value_type& obj) {
        return rep.insert_unique(obj);
    }

    template <class M>
    std::pair<iterator, bool> insert_or_assign(const Key& k, M&& value) {
        auto result = try_emplace(k, std::forward<M>(value));
        if (!result.second) {
            result.first->second = std::forward<M>(value);
        }
        return result;
    }

    template <class M>
    std::pair<iterator, bool> insert_or_assign(Key&& k, M&& value) {
        auto result = try_emplace(std::move(k), std::forward<M>(value));
        if (!result.second) {
            result.first->second = std::forward<M>(value);
        }
        return result;
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        return rep.emplace_unique(std::forward<Args>(args)...);
    }

    std::pair<iterator, bool> insert_noresize(const value_type& obj) {
        return rep.insert_unique_noresize(obj);
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace_noresize(Args&&... args) {
        return rep.emplace_unique_noresize(std::forward<Args>(args)...);
    }

    template <class TheObj>
    iterator insert_direct(const TheObj& obj, const insert_ctx& ins) {
        return rep.insert_direct(obj, ins);
    }

    template <typename... Args>
    iterator emplace_direct(const insert_ctx& ins, Args&&... args) {
        return rep.emplace_direct(ins, std::forward<Args>(args)...);
    }

    template <typename TKey, typename... Args>
    std::pair<iterator, bool> try_emplace(TKey&& key, Args&&... args) {
        insert_ctx ctx = nullptr;
        iterator it = find(key, ctx);
        if (it == end()) {
            it = rep.emplace_direct(ctx, std::piecewise_construct,
                                    std::forward_as_tuple(std::forward<TKey>(key)),
                                    std::forward_as_tuple(std::forward<Args>(args)...));
            return {it, true};
        }
        return {it, false};
    }

    template <class TheKey>
    iterator find(const TheKey& key) {
        return rep.find(key);
    }

    template <class TheKey>
    const_iterator find(const TheKey& key) const {
        return rep.find(key);
    }

    template <class TheKey>
    iterator find(const TheKey& key, insert_ctx& ins) {
        return rep.find_i(key, ins);
    }

    template <class TheKey>
    bool contains(const TheKey& key) const {
        return rep.find(key) != rep.end();
    }
    bool contains(const key_type& key) const {
        return rep.find(key) != rep.end();
    }

    template <class TheKey>
    bool contains(const TheKey& key, insert_ctx& ins) {
        return rep.find_i(key, ins) != rep.end();
    }

    template <class TKey>
    T& operator[](const TKey& key) {
        insert_ctx ctx = nullptr;
        iterator it = find(key, ctx);

        if (it != end()) {
            return it->second;
        }

        return rep.emplace_direct(ctx, std::piecewise_construct, std::forward_as_tuple(key), std::forward_as_tuple())->second;
    }

    template <class TheKey>
    const T& at(const TheKey& key) const {
        using namespace ::NPrivate;
        const_iterator it = find(key);

        if (Y_UNLIKELY(it == end())) {
            ::NPrivate::ThrowKeyNotFoundInHashTableException(MapKeyToString(key));
        }

        return it->second;
    }

    template <class TheKey>
    T& at(const TheKey& key) {
        using namespace ::NPrivate;
        iterator it = find(key);

        if (Y_UNLIKELY(it == end())) {
            ::NPrivate::ThrowKeyNotFoundInHashTableException(MapKeyToString(key));
        }

        return it->second;
    }

    template <class TKey>
    size_type count(const TKey& key) const {
        return rep.count(key);
    }

    template <class TKey>
    std::pair<iterator, iterator> equal_range(const TKey& key) {
        return rep.equal_range(key);
    }

    template <class TKey>
    std::pair<const_iterator, const_iterator> equal_range(const TKey& key) const {
        return rep.equal_range(key);
    }

    template <class TKey>
    size_type erase(const TKey& key) {
        return rep.erase_one(key);
    }

    void erase(iterator it) {
        rep.erase(it);
    }
    void erase(iterator f, iterator l) {
        rep.erase(f, l);
    }
    Y_REINITIALIZES_OBJECT void clear() {
        rep.clear();
    }
    Y_REINITIALIZES_OBJECT void clear(size_t downsize_hint) {
        rep.clear(downsize_hint);
    }
    Y_REINITIALIZES_OBJECT void basic_clear() {
        rep.basic_clear();
    }
    void release_nodes() {
        rep.release_nodes();
    }

    // if (stHash != NULL) bucket_count() must be equal to stHash->bucket_count()
    template <class KeySaver>
    int save_for_st(IOutputStream* stream, KeySaver& ks, sthash<int, int, THash<int>, TEqualTo<int>, typename KeySaver::TSizeType>* stHash = nullptr) const {
        return rep.template save_for_st<KeySaver>(stream, ks, stHash);
    }

public:
    void reserve(size_type hint) {
        rep.reserve(hint);
    }
    size_type bucket_count() const {
        return rep.bucket_count();
    }
    size_type bucket_size(size_type n) const {
        return rep.bucket_size(n);
    }
    node_allocator_type& GetNodeAllocator() {
        return rep.GetNodeAllocator();
    }
    const node_allocator_type& GetNodeAllocator() const {
        return rep.GetNodeAllocator();
    }
};

template <class Key, class T, class HashFcn, class EqualKey, class Alloc>
inline bool operator==(const THashMap<Key, T, HashFcn, EqualKey, Alloc>& hm1, const THashMap<Key, T, HashFcn, EqualKey, Alloc>& hm2) {
    if (hm1.size() != hm2.size()) {
        return false;
    }
    for (const auto& it1 : hm1) {
        auto it2 = hm2.find(it1.first);
        if ((it2 == hm2.end()) || !(it1 == *it2)) {
            return false;
        }
    }
    return true;
}

template <class Key, class T, class HashFcn, class EqualKey, class Alloc>
inline bool operator!=(const THashMap<Key, T, HashFcn, EqualKey, Alloc>& hm1, const THashMap<Key, T, HashFcn, EqualKey, Alloc>& hm2) {
    return !(hm1 == hm2);
}
