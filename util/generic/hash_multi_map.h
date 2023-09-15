#pragma once

#include "fwd.h"
#include "hash_table.h"

template <class Key, class T, class HashFcn, class EqualKey, class Alloc>
class THashMultiMap {
private:
    using ht = THashTable<std::pair<const Key, T>, Key, HashFcn, TSelect1st, EqualKey, Alloc>;
    ht rep;

public:
    using key_type = typename ht::key_type;
    using value_type = typename ht::value_type;
    using hasher = typename ht::hasher;
    using key_equal = typename ht::key_equal;
    using mapped_type = T;
    using allocator_type = typename ht::allocator_type;

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
    THashMultiMap()
        : rep(0, hasher(), key_equal())
    {
    }
    template <typename TAllocParam>
    explicit THashMultiMap(TAllocParam* allocParam)
        : rep(0, hasher(), key_equal(), allocParam)
    {
    }
    explicit THashMultiMap(size_type n)
        : rep(n, hasher(), key_equal())
    {
    }
    THashMultiMap(size_type n, const hasher& hf)
        : rep(n, hf, key_equal())
    {
    }
    THashMultiMap(size_type n, const hasher& hf, const key_equal& eql)
        : rep(n, hf, eql)
    {
    }

    template <class InputIterator>
    THashMultiMap(InputIterator f, InputIterator l)
        : rep(0, hasher(), key_equal())
    {
        rep.insert_equal(f, l);
    }
    template <class InputIterator>
    THashMultiMap(InputIterator f, InputIterator l, size_type n)
        : rep(n, hasher(), key_equal())
    {
        rep.insert_equal(f, l);
    }
    template <class InputIterator>
    THashMultiMap(InputIterator f, InputIterator l, size_type n, const hasher& hf)
        : rep(n, hf, key_equal())
    {
        rep.insert_equal(f, l);
    }
    template <class InputIterator>
    THashMultiMap(InputIterator f, InputIterator l, size_type n, const hasher& hf, const key_equal& eql)
        : rep(n, hf, eql)
    {
        rep.insert_equal(f, l);
    }

    THashMultiMap(std::initializer_list<std::pair<Key, T>> list)
        : rep(list.size(), hasher(), key_equal())
    {
        for (const auto& v : list) {
            rep.emplace_equal_noresize(v);
        }
    }

    // THashMultiMap has implicit copy/move constructors and copy-/move-assignment operators
    // because its implementation is backed by THashTable.
    // See hash_ut.cpp

public:
    size_type size() const {
        return rep.size();
    }
    yssize_t ysize() const {
        return (yssize_t)rep.size();
    }
    size_type max_size() const {
        return rep.max_size();
    }

    Y_PURE_FUNCTION bool empty() const {
        return rep.empty();
    }
    explicit operator bool() const noexcept {
        return !empty();
    }
    void swap(THashMultiMap& hs) {
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
        rep.insert_equal(f, l);
    }

    iterator insert(const value_type& obj) {
        return rep.insert_equal(obj);
    }

    template <typename... Args>
    iterator emplace(Args&&... args) {
        return rep.emplace_equal(std::forward<Args>(args)...);
    }

    iterator insert_noresize(const value_type& obj) {
        return rep.emplace_equal_noresize(obj);
    }

    template <typename... Args>
    iterator emplace_noresize(Args&&... args) {
        return rep.emplace_equal_noresize(std::forward<Args>(args)...);
    }

    template <class TheObj>
    iterator insert_direct(const TheObj& obj, const insert_ctx& ins) {
        return rep.insert_direct(obj, ins);
    }

    template <typename... Args>
    iterator emplace_direct(const insert_ctx& ins, Args&&... args) {
        return rep.emplace_direct(ins, std::forward<Args>(args)...);
    }

    template <class TKey>
    const_iterator find(const TKey& key) const {
        return rep.find(key);
    }

    template <class TKey>
    iterator find(const TKey& key) {
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

    template <class TKey>
    size_type count(const TKey& key) const {
        return rep.count(key);
    }

    template <class TKey>
    std::pair<iterator, iterator> equal_range(const TKey& key) {
        return rep.equal_range(key);
    }

    template <class TKey>
    std::pair<iterator, iterator> equal_range_i(const TKey& key, insert_ctx& ins) {
        return rep.equal_range_i(key, ins);
    }

    template <class TKey>
    std::pair<const_iterator, const_iterator> equal_range(const TKey& key) const {
        return rep.equal_range(key);
    }

    size_type erase(const key_type& key) {
        return rep.erase(key);
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
};

template <class Key, class T, class HF, class EqKey, class Alloc>
inline bool operator==(const THashMultiMap<Key, T, HF, EqKey, Alloc>& hm1, const THashMultiMap<Key, T, HF, EqKey, Alloc>& hm2) {
    // NOTE: copy-pasted from
    // contrib/libs/cxxsupp/libcxx/include/unordered_map
    // and adapted to THashMultiMap
    if (hm1.size() != hm2.size()) {
        return false;
    }
    using const_iterator = typename THashMultiMap<Key, T, HF, EqKey, Alloc>::const_iterator;
    using TEqualRange = std::pair<const_iterator, const_iterator>;
    for (const_iterator it = hm1.begin(), end = hm1.end(); it != end;) {
        TEqualRange eq1 = hm1.equal_range(it->first);
        TEqualRange eq2 = hm2.equal_range(it->first);
        if (std::distance(eq1.first, eq1.second) != std::distance(eq2.first, eq2.second) ||
            !std::is_permutation(eq1.first, eq1.second, eq2.first))
        {
            return false;
        }
        it = eq1.second;
    }
    return true;
}

template <class Key, class T, class HF, class EqKey, class Alloc>
inline bool operator!=(const THashMultiMap<Key, T, HF, EqKey, Alloc>& hm1, const THashMultiMap<Key, T, HF, EqKey, Alloc>& hm2) {
    return !(hm1 == hm2);
}
