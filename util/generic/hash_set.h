#pragma once

#include "fwd.h"
#include "hash.h"

#include <util/system/compiler.h>

#include <initializer_list>
#include <utility>

#undef value_type

template <class Value, class HashFcn, class EqualKey, class Alloc>
class THashSet {
private:
    using ht = THashTable<Value, Value, HashFcn, ::TIdentity, EqualKey, Alloc>;
    ht rep;

    using mutable_iterator = typename ht::iterator;

public:
    using key_type = typename ht::key_type;
    using value_type = typename ht::value_type;
    using hasher = typename ht::hasher;
    using key_equal = typename ht::key_equal;
    using allocator_type = typename ht::allocator_type;
    using node_allocator_type = typename ht::node_allocator_type;

    using size_type = typename ht::size_type;
    using difference_type = typename ht::difference_type;
    using pointer = typename ht::const_pointer;
    using const_pointer = typename ht::const_pointer;
    using reference = typename ht::const_reference;
    using const_reference = typename ht::const_reference;

    using iterator = typename ht::const_iterator;
    using const_iterator = typename ht::const_iterator;
    using insert_ctx = typename ht::insert_ctx;

    hasher hash_function() const {
        return rep.hash_function();
    }
    key_equal key_eq() const {
        return rep.key_eq();
    }

public:
    THashSet() {
    }
    template <class TT>
    explicit THashSet(TT* allocParam, size_type n = 0)
        : rep(n, hasher(), key_equal(), allocParam)
    {
    }
    explicit THashSet(size_type n)
        : rep(n, hasher(), key_equal())
    {
    }
    THashSet(size_type n, const hasher& hf)
        : rep(n, hf, key_equal())
    {
    }
    THashSet(size_type n, const hasher& hf, const key_equal& eql)
        : rep(n, hf, eql)
    {
    }

    THashSet(std::initializer_list<value_type> list)
        : rep(list.size(), hasher(), key_equal())
    {
        rep.insert_unique(list.begin(), list.end());
    }
    THashSet(std::initializer_list<value_type> list, size_type n)
        : rep(n, hasher(), key_equal())
    {
        rep.insert_unique(list.begin(), list.end());
    }
    THashSet(std::initializer_list<value_type> list, size_type n, const hasher& hf)
        : rep(n, hf, key_equal())
    {
        rep.insert_unique(list.begin(), list.end());
    }
    THashSet(std::initializer_list<value_type> list, size_type n, const hasher& hf, const key_equal& eql)
        : rep(n, hf, eql)
    {
        rep.insert_unique(list.begin(), list.end());
    }

    template <class InputIterator>
    THashSet(InputIterator f, InputIterator l)
        : rep(0, hasher(), key_equal())
    {
        rep.insert_unique(f, l);
    }
    template <class InputIterator>
    THashSet(InputIterator f, InputIterator l, size_type n)
        : rep(n, hasher(), key_equal())
    {
        rep.insert_unique(f, l);
    }
    template <class InputIterator>
    THashSet(InputIterator f, InputIterator l, size_type n,
             const hasher& hf)
        : rep(n, hf, key_equal())
    {
        rep.insert_unique(f, l);
    }
    template <class InputIterator>
    THashSet(InputIterator f, InputIterator l, size_type n,
             const hasher& hf, const key_equal& eql)
        : rep(n, hf, eql)
    {
        rep.insert_unique(f, l);
    }

    // THashSet has implicit copy/move constructors and copy-/move-assignment operators
    // because its implementation is backed by THashTable.
    // See hash_ut.cpp

public:
    size_type size() const {
        return rep.size();
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
    void swap(THashSet& hs) {
        rep.swap(hs.rep);
    }

    iterator begin() const {
        return rep.begin();
    }
    iterator end() const {
        return rep.end();
    }
    iterator cbegin() const {
        return rep.begin();
    }
    iterator cend() const {
        return rep.end();
    }

public:
    void insert(std::initializer_list<value_type> ilist) {
        insert(ilist.begin(), ilist.end());
    }

    template <class InputIterator>
    void insert(InputIterator f, InputIterator l) {
        rep.insert_unique(f, l);
    }

    std::pair<iterator, bool> insert(const value_type& obj) {
        std::pair<mutable_iterator, bool> p = rep.insert_unique(obj);
        return std::pair<iterator, bool>(p.first, p.second);
    }
    template <typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        std::pair<mutable_iterator, bool> p = rep.emplace_unique(std::forward<Args>(args)...);
        return std::pair<iterator, bool>(p.first, p.second);
    }

    iterator insert(const_iterator, const value_type& obj) { // insert_hint
        std::pair<mutable_iterator, bool> p = rep.insert_unique(obj);
        return p.first;
    }

    std::pair<iterator, bool> insert_noresize(const value_type& obj) {
        std::pair<mutable_iterator, bool> p = rep.insert_unique_noresize(obj);
        return std::pair<iterator, bool>(p.first, p.second);
    }
    template <typename... Args>
    std::pair<iterator, bool> emplace_noresize(Args&&... args) {
        std::pair<mutable_iterator, bool> p = rep.emplace_unique_noresize(std::forward<Args>(args)...);
        return std::pair<iterator, bool>(p.first, p.second);
    }

    template <class TheObj>
    iterator insert_direct(const TheObj& obj, const insert_ctx& ins) {
        return rep.insert_direct(obj, ins);
    }
    template <typename... Args>
    iterator emplace_direct(const insert_ctx& ins, Args&&... args) {
        return rep.emplace_direct(ins, std::forward<Args>(args)...);
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
    template <class TheKey>
    bool contains(const TheKey& key, insert_ctx& ins) {
        return rep.find_i(key, ins) != rep.end();
    }

    template <class TKey>
    size_type count(const TKey& key) const {
        return rep.count(key);
    }

    template <class TKey>
    std::pair<iterator, iterator> equal_range(const TKey& key) const {
        return rep.equal_range(key);
    }

    size_type erase(const key_type& key) {
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

    template <class KeySaver>
    int save_for_st(IOutputStream* stream, KeySaver& ks) const {
        return rep.template save_for_st<KeySaver>(stream, ks);
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
};

template <class Value, class HashFcn, class EqualKey, class Alloc>
inline bool operator==(const THashSet<Value, HashFcn, EqualKey, Alloc>& hs1, const THashSet<Value, HashFcn, EqualKey, Alloc>& hs2) {
    if (hs1.size() != hs2.size()) {
        return false;
    }
    for (const auto& it : hs1) {
        if (!hs2.contains(it)) {
            return false;
        }
    }
    return true;
}

template <class Value, class HashFcn, class EqualKey, class Alloc>
inline bool operator!=(const THashSet<Value, HashFcn, EqualKey, Alloc>& hs1, const THashSet<Value, HashFcn, EqualKey, Alloc>& hs2) {
    return !(hs1 == hs2);
}

template <class Value, class HashFcn, class EqualKey, class Alloc>
class THashMultiSet {
private:
    using ht = THashTable<Value, Value, HashFcn, ::TIdentity, EqualKey, Alloc>;
    ht rep;

public:
    using key_type = typename ht::key_type;
    using value_type = typename ht::value_type;
    using hasher = typename ht::hasher;
    using key_equal = typename ht::key_equal;
    using allocator_type = typename ht::allocator_type;
    using node_allocator_type = typename ht::node_allocator_type;

    using size_type = typename ht::size_type;
    using difference_type = typename ht::difference_type;
    using pointer = typename ht::const_pointer;
    using const_pointer = typename ht::const_pointer;
    using reference = typename ht::const_reference;
    using const_reference = typename ht::const_reference;

    using iterator = typename ht::const_iterator;
    using const_iterator = typename ht::const_iterator;

    hasher hash_function() const {
        return rep.hash_function();
    }
    key_equal key_eq() const {
        return rep.key_eq();
    }

public:
    THashMultiSet()
        : rep(0, hasher(), key_equal())
    {
    }
    explicit THashMultiSet(size_type n)
        : rep(n, hasher(), key_equal())
    {
    }
    THashMultiSet(size_type n, const hasher& hf)
        : rep(n, hf, key_equal())
    {
    }
    THashMultiSet(size_type n, const hasher& hf, const key_equal& eql)
        : rep(n, hf, eql)
    {
    }

    template <class InputIterator>
    THashMultiSet(InputIterator f, InputIterator l)
        : rep(0, hasher(), key_equal())
    {
        rep.insert_equal(f, l);
    }
    template <class InputIterator>
    THashMultiSet(InputIterator f, InputIterator l, size_type n)
        : rep(n, hasher(), key_equal())
    {
        rep.insert_equal(f, l);
    }
    template <class InputIterator>
    THashMultiSet(InputIterator f, InputIterator l, size_type n,
                  const hasher& hf)
        : rep(n, hf, key_equal())
    {
        rep.insert_equal(f, l);
    }
    template <class InputIterator>
    THashMultiSet(InputIterator f, InputIterator l, size_type n,
                  const hasher& hf, const key_equal& eql)
        : rep(n, hf, eql)
    {
        rep.insert_equal(f, l);
    }

    THashMultiSet(std::initializer_list<value_type> list)
        : rep(list.size(), hasher(), key_equal())
    {
        rep.insert_equal(list.begin(), list.end());
    }

    // THashMultiSet has implicit copy/move constructors and copy-/move-assignment operators
    // because its implementation is backed by THashTable.
    // See hash_ut.cpp

public:
    size_type size() const {
        return rep.size();
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
    void swap(THashMultiSet& hs) {
        rep.swap(hs.rep);
    }

    iterator begin() const {
        return rep.begin();
    }
    iterator end() const {
        return rep.end();
    }
    iterator cbegin() const {
        return rep.begin();
    }
    iterator cend() const {
        return rep.end();
    }

public:
    iterator insert(const value_type& obj) {
        return rep.insert_equal(obj);
    }
    template <typename... Args>
    iterator emplace(Args&&... args) {
        return rep.emplace_equal(std::forward<Args>(args)...);
    }
    template <class InputIterator>
    void insert(InputIterator f, InputIterator l) {
        rep.insert_equal(f, l);
    }
    iterator insert_noresize(const value_type& obj) {
        return rep.insert_equal_noresize(obj);
    }

    template <class TKey>
    iterator find(const TKey& key) const {
        return rep.find(key);
    }

    template <class TKey>
    size_type count(const TKey& key) const {
        return rep.count(key);
    }

    template <class TKey>
    std::pair<iterator, iterator> equal_range(const TKey& key) const {
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
};

template <class Val, class HashFcn, class EqualKey, class Alloc>
inline bool operator==(const THashMultiSet<Val, HashFcn, EqualKey, Alloc>& hs1, const THashMultiSet<Val, HashFcn, EqualKey, Alloc>& hs2) {
    if (hs1.size() != hs2.size()) {
        return false;
    }
    EqualKey equalKey;
    auto it = hs1.begin();
    while (it != hs1.end()) {
        const auto& value = *it;
        size_t count = 0;
        for (; (it != hs1.end()) && (equalKey(*it, value)); ++it, ++count) {
        }
        if (hs2.count(value) != count) {
            return false;
        }
    }
    return true;
}

template <class Val, class HashFcn, class EqualKey, class Alloc>
inline bool operator!=(const THashMultiSet<Val, HashFcn, EqualKey, Alloc>& hs1, const THashMultiSet<Val, HashFcn, EqualKey, Alloc>& hs2) {
    return !(hs1 == hs2);
}
