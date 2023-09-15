#pragma once

#include "fwd.h"
#include "mapfindptr.h"

#include <util/memory/alloc.h>
#include <util/system/compiler.h>
#include <util/system/type_name.h>
#include <util/system/yassert.h>
#include <util/str_stl.h>
#include "yexception.h"
#include "typetraits.h"
#include "utility.h"

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <tuple>
#include <utility>

#include <cstdlib>

#include "hash_primes.h"

struct TSelect1st {
    template <class TPair>
    inline const typename TPair::first_type& operator()(const TPair& x) const {
        return x.first;
    }
};

template <class Value>
struct __yhashtable_node {
    /** If the first bit is not set, then this is a pointer to the next node in
     * the list of nodes for the current bucket. Otherwise this is a pointer of
     * type __yhashtable_node**, pointing back into the buckets array.
     *
     * This trick makes it possible to use only one node pointer in a hash table
     * iterator. */
    __yhashtable_node* next;

    /** Value stored in a node. */
    Value val;

    __yhashtable_node& operator=(const __yhashtable_node&) = delete;
};

template <class Value, class Key, class HashFcn,
          class ExtractKey, class EqualKey, class Alloc>
class THashTable;

template <class Key, class T, class HashFcn,
          class EqualKey, typename size_type_f>
class sthash;

template <class Value>
struct __yhashtable_iterator;

template <class Value>
struct __yhashtable_const_iterator;

template <class Value>
struct __yhashtable_iterator {
    using iterator = __yhashtable_iterator<Value>;
    using const_iterator = __yhashtable_const_iterator<Value>;
    using node = __yhashtable_node<Value>;

    using iterator_category = std::forward_iterator_tag;
    using value_type = Value;
    using difference_type = ptrdiff_t;
    using size_type = size_t;
    using reference = Value&;
    using pointer = Value*;

    node* cur;

    explicit __yhashtable_iterator(node* n)
        : cur(n)
    {
    } /*y*/
    __yhashtable_iterator() = default;

    reference operator*() const {
        return cur->val;
    }
    pointer operator->() const {
        return &(operator*());
    }
    iterator& operator++();
    iterator operator++(int);
    friend bool operator==(const iterator& l, const iterator& r) {
        return l.cur == r.cur;
    }
    friend bool operator!=(const iterator& l, const iterator& r) {
        return l.cur != r.cur;
    }
    bool IsEnd() const {
        return !cur;
    }
    Y_FORCE_INLINE explicit operator bool() const noexcept {
        return cur != nullptr;
    }
};

template <class Value>
struct __yhashtable_const_iterator {
    using iterator = __yhashtable_iterator<Value>;
    using const_iterator = __yhashtable_const_iterator<Value>;
    using node = __yhashtable_node<Value>;

    using iterator_category = std::forward_iterator_tag;
    using value_type = Value;
    using difference_type = ptrdiff_t;
    using size_type = size_t;
    using reference = const Value&;
    using pointer = const Value*;

    const node* cur;

    explicit __yhashtable_const_iterator(const node* n)
        : cur(n)
    {
    }
    __yhashtable_const_iterator() {
    }
    __yhashtable_const_iterator(const iterator& it)
        : cur(it.cur)
    {
    }
    reference operator*() const {
        return cur->val;
    }
    pointer operator->() const {
        return &(operator*());
    }
    const_iterator& operator++();
    const_iterator operator++(int);
    friend bool operator==(const const_iterator& l, const const_iterator& r) {
        return l.cur == r.cur;
    }
    friend bool operator!=(const const_iterator& l, const const_iterator& r) {
        return l.cur != r.cur;
    }
    bool IsEnd() const {
        return !cur;
    }
    Y_FORCE_INLINE explicit operator bool() const noexcept {
        return cur != nullptr;
    }
};

/**
 * This class saves some space in allocator-based containers for the most common
 * use case of empty allocators. This is achieved thanks to the application of
 * empty base class optimization (aka EBCO).
 */
template <class Alloc>
class _allocator_base: private Alloc {
public:
    _allocator_base(const Alloc& other)
        : Alloc(other)
    {
    }

    Alloc& _get_alloc() {
        return static_cast<Alloc&>(*this);
    }
    const Alloc& _get_alloc() const {
        return static_cast<const Alloc&>(*this);
    }
    void _set_alloc(const Alloc& allocator) {
        _get_alloc() = allocator;
    }

    void swap(_allocator_base& other) {
        DoSwap(_get_alloc(), other._get_alloc());
    }
};

/**
 * Wrapper for an array of THashTable buckets.
 *
 * Is better than vector for this particular use case. Main differences:
 *   - Occupies one less word on stack.
 *   - Doesn't even try to initialize its elements. It is THashTable's responsibility.
 *   - Presents a better interface in relation to THashTable's marker element trick.
 *
 * Internally this class is just a pointer-size pair, and the data on the heap
 * has the following structure:
 *
 *     +----------+----------------------+----------+-------------------------+
 *     | raw_size | elements ...         | marker   | unused space [optional] |
 *     +----------+----------------------+----------+-------------------------+
 *                 ^                      ^
 *                 |                      |
 *                Data points here       end() points here
 *
 * `raw_size` stores the size of the allocated memory block. It is used to
 * support resizing without reallocation.
 *
 * `marker` is a special marker element that is set by the THashTable that is
 * then used in iterator implementation to know when the end is reached.
 *
 * Unused space at the end of the memory block may not be present.
 */
template <class T, class Alloc>
class _yhashtable_buckets: private _allocator_base<Alloc> {
    using base_type = _allocator_base<Alloc>;

    static_assert(sizeof(T) == sizeof(size_t), "T is expected to be the same size as size_t.");

public:
    using allocator_type = Alloc;
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using TBucketDivisor = ::NPrivate::THashDivisor;

    _yhashtable_buckets(const Alloc& other)
        : base_type(other)
        , Data(nullptr)
        , Size()
    {
    }

    ~_yhashtable_buckets() {
        Y_ASSERT(!Data);
    }

    void initialize_dynamic(TBucketDivisor size) {
        Y_ASSERT(!Data);

        Data = this->_get_alloc().allocate(size() + 2) + 1;
        Size = size;

        *reinterpret_cast<size_type*>(Data - 1) = size() + 2;
    }

    void deinitialize_dynamic() {
        Y_ASSERT(Data);

        this->_get_alloc().deallocate(Data - 1, *reinterpret_cast<size_type*>(Data - 1));
        Data = pointer();
        Size = TBucketDivisor();
    }

    void initialize_static(pointer data, TBucketDivisor size) {
        Y_ASSERT(!Data && data && size() >= 1);

        Data = data;
        Size = size;
    }

    void deinitialize_static() {
        Y_ASSERT(Data);

        Data = pointer();
        Size = TBucketDivisor();
    }

    void resize_noallocate(TBucketDivisor size) {
        Y_ASSERT(size() <= capacity());

        Size = size;
    }

    iterator begin() {
        return Data;
    }
    const_iterator begin() const {
        return Data;
    }
    iterator end() {
        return Data + Size();
    }
    const_iterator end() const {
        return Data + Size();
    }

    pointer data() {
        return Data;
    }
    const_pointer data() const {
        return Data;
    }

    size_type size() const {
        return Size();
    }
    size_type capacity() const {
        return *reinterpret_cast<size_type*>(Data - 1);
    }
    TBucketDivisor ExtSize() const {
        return Size;
    }
    int BucketDivisorHint() const {
        return +Size.Hint;
    }

    allocator_type get_allocator() const {
        return this->_get_alloc();
    }

    const_reference operator[](size_type index) const {
        Y_ASSERT(index <= Size());

        return *(Data + index);
    }

    reference operator[](size_type index) {
        Y_ASSERT(index <= Size());

        return *(Data + index);
    }

    void swap(_yhashtable_buckets& other) {
        base_type::swap(other);
        DoSwap(Data, other.Data);
        DoSwap(Size, other.Size);
    }

private:
    /** Pointer to the first element of the buckets array. */
    pointer Data;

    /** Size of the buckets array. Doesn't take the marker element at the end into account. */
    TBucketDivisor Size;
};

/**
 * This class saves one word in THashTable for the most common use case of empty
 * functors. The exact implementation picks a specialization with storage allocated
 * for the functors if those are non-empty, and another specialization that creates
 * functors on the fly if they are empty. It is expected that empty functors have
 * trivial constructors.
 *
 * Note that this is basically the only way to do it portably. Another option is
 * multiple inheritance from empty functors, but MSVC's empty base class
 * optimization chokes up on multiple empty bases, and we're already using
 * EBCO in _allocator_base.
 *
 * Note that there are no specializations for the case when only one or two
 * of the functors are empty as this is a case that's just way too rare.
 */
template <class HashFcn, class ExtractKey, class EqualKey, class Alloc, bool IsEmpty = std::is_empty<HashFcn>::value&& std::is_empty<ExtractKey>::value&& std::is_empty<EqualKey>::value>
class _yhashtable_base: public _allocator_base<Alloc> {
    using base_type = _allocator_base<Alloc>;

public:
    _yhashtable_base(const HashFcn& hash, const ExtractKey& extract, const EqualKey& equals, const Alloc& alloc)
        : base_type(alloc)
        , hash_(hash)
        , extract_(extract)
        , equals_(equals)
    {
    }

    const EqualKey& _get_key_eq() const {
        return equals_;
    }
    EqualKey& _get_key_eq() {
        return equals_;
    }
    void _set_key_eq(const EqualKey& equals) {
        this->equals_ = equals;
    }

    const ExtractKey& _get_key_extract() const {
        return extract_;
    }
    ExtractKey& _get_key_extract() {
        return extract_;
    }
    void _set_key_extract(const ExtractKey& extract) {
        this->extract_ = extract;
    }

    const HashFcn& _get_hash_fun() const {
        return hash_;
    }
    HashFcn& _get_hash_fun() {
        return hash_;
    }
    void _set_hash_fun(const HashFcn& hash) {
        this->hash_ = hash;
    }

    void swap(_yhashtable_base& other) {
        base_type::swap(other);
        DoSwap(equals_, other.equals_);
        DoSwap(extract_, other.extract_);
        DoSwap(hash_, other.hash_);
    }

private:
    HashFcn hash_;
    ExtractKey extract_;
    EqualKey equals_;
};

template <class HashFcn, class ExtractKey, class EqualKey, class Alloc>
class _yhashtable_base<HashFcn, ExtractKey, EqualKey, Alloc, true>: public _allocator_base<Alloc> {
    using base_type = _allocator_base<Alloc>;

public:
    _yhashtable_base(const HashFcn&, const ExtractKey&, const EqualKey&, const Alloc& alloc)
        : base_type(alloc)
    {
    }

    EqualKey _get_key_eq() const {
        return EqualKey();
    }
    void _set_key_eq(const EqualKey&) {
    }

    ExtractKey _get_key_extract() const {
        return ExtractKey();
    }
    void _set_key_extract(const ExtractKey&) {
    }

    HashFcn _get_hash_fun() const {
        return HashFcn();
    }
    void _set_hash_fun(const HashFcn&) {
    }

    void swap(_yhashtable_base& other) {
        base_type::swap(other);
    }
};

template <class Value, class Key, class HashFcn, class ExtractKey, class EqualKey, class Alloc>
struct _yhashtable_traits {
    using node = __yhashtable_node<Value>;

    using node_allocator_type = TReboundAllocator<Alloc, node>;
    using nodep_allocator_type = TReboundAllocator<Alloc, node*>;

    using base_type = _yhashtable_base<HashFcn, ExtractKey, EqualKey, node_allocator_type>;
};

extern const void* const _yhashtable_empty_data[];

template <class Value, class Key, class HashFcn, class ExtractKey, class EqualKey, class Alloc>
class THashTable: private _yhashtable_traits<Value, Key, HashFcn, ExtractKey, EqualKey, Alloc>::base_type {
    using traits_type = _yhashtable_traits<Value, Key, HashFcn, ExtractKey, EqualKey, Alloc>;
    using base_type = typename traits_type::base_type;
    using node = typename traits_type::node;
    using nodep_allocator_type = typename traits_type::nodep_allocator_type;
    using buckets_type = _yhashtable_buckets<node*, nodep_allocator_type>;
    using TBucketDivisor = ::NPrivate::THashDivisor;

public:
    using key_type = Key;
    using value_type = Value;
    using hasher = HashFcn;
    using key_equal = EqualKey;
    using key_extract = ExtractKey;
    using allocator_type = Alloc;
    using node_allocator_type = typename traits_type::node_allocator_type;

    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;

    node_allocator_type& GetNodeAllocator() {
        return this->_get_alloc();
    }
    const node_allocator_type& GetNodeAllocator() const {
        return this->_get_alloc();
    }
    key_equal key_eq() const {
        return this->_get_key_eq();
    }
    hasher hash_function() const {
        return this->_get_hash_fun();
    }

private:
    template <class KeyL, class KeyR>
    bool equals(const KeyL& l, const KeyR& r) const {
        return this->_get_key_eq()(l, r);
    }

    /* This method is templated to postpone instantiation of key extraction functor. */
    template <class ValueL>
    auto get_key(const ValueL& value) const -> decltype(ExtractKey()(value)) {
        return this->_get_key_extract()(value);
    }

    node* get_node() {
        node* result = this->_get_alloc().allocate(1);
        Y_ASSERT((reinterpret_cast<uintptr_t>(result) & 1) == 0); /* We're using the last bit of the node pointer. */
        return result;
    }
    void put_node(node* p) {
        this->_get_alloc().deallocate(p, 1);
    }

    buckets_type buckets;
    size_type num_elements;

public:
    using iterator = __yhashtable_iterator<Value>;
    using const_iterator = __yhashtable_const_iterator<Value>;
    using insert_ctx = node**;

    friend struct __yhashtable_iterator<Value>;
    friend struct __yhashtable_const_iterator<Value>;

public:
    THashTable()
        : base_type(HashFcn(), ExtractKey(), EqualKey(), node_allocator_type())
        , buckets(nodep_allocator_type())
        , num_elements(0)
    {
        initialize_buckets(buckets, 0);
    }

    THashTable(size_type n, const HashFcn& hf, const EqualKey& eql, const ExtractKey& ext)
        : base_type(hf, ext, eql, node_allocator_type())
        , buckets(nodep_allocator_type())
        , num_elements(0)
    {
        initialize_buckets(buckets, n);
    }

    THashTable(size_type n, const HashFcn& hf, const EqualKey& eql)
        : base_type(hf, ExtractKey(), eql, node_allocator_type())
        , buckets(nodep_allocator_type())
        , num_elements(0)
    {
        initialize_buckets(buckets, n);
    }

    template <class TAllocParam>
    THashTable(size_type n, const HashFcn& hf, const EqualKey& eql, TAllocParam* allocParam)
        : base_type(hf, ExtractKey(), eql, allocParam)
        , buckets(allocParam)
        , num_elements(0)
    {
        initialize_buckets(buckets, n);
    }

    THashTable(const THashTable& ht)
        : base_type(ht._get_hash_fun(), ht._get_key_extract(), ht._get_key_eq(), ht._get_alloc())
        , buckets(ht.buckets.get_allocator())
        , num_elements(0)
    {
        if (ht.empty()) {
            initialize_buckets(buckets, 0);
        } else {
            initialize_buckets_dynamic(buckets, ht.buckets.ExtSize());
            copy_from_dynamic(ht);
        }
    }

    THashTable(THashTable&& ht) noexcept
        : base_type(ht._get_hash_fun(), ht._get_key_extract(), ht._get_key_eq(), ht._get_alloc())
        , buckets(ht.buckets.get_allocator())
        , num_elements(0)
    {
        initialize_buckets(buckets, 0);
        this->swap(ht);
    }

    THashTable& operator=(const THashTable& ht) {
        if (&ht != this) {
            basic_clear();
            this->_set_hash_fun(ht._get_hash_fun());
            this->_set_key_eq(ht._get_key_eq());
            this->_set_key_extract(ht._get_key_extract());
            /* We don't copy allocator for a reason. */

            if (ht.empty()) {
                /* Some of the old code in Arcadia works around the behavior in
                 * clear() by invoking operator= with empty hash as an argument.
                 * It's expected that this will deallocate the buckets array, so
                 * this is what we have to do here. */
                deinitialize_buckets(buckets);
                initialize_buckets(buckets, 0);
            } else {
                if (buckets.capacity() > ht.buckets.size()) {
                    buckets.resize_noallocate(ht.buckets.ExtSize());
                } else {
                    deinitialize_buckets(buckets);
                    initialize_buckets_dynamic(buckets, ht.buckets.ExtSize());
                }

                copy_from_dynamic(ht);
            }
        }
        return *this;
    }

    THashTable& operator=(THashTable&& ht) noexcept {
        basic_clear();
        swap(ht);

        return *this;
    }

    ~THashTable() {
        basic_clear();
        deinitialize_buckets(buckets);
    }

    size_type size() const noexcept {
        return num_elements;
    }
    size_type max_size() const noexcept {
        return size_type(-1);
    }

    Y_PURE_FUNCTION bool empty() const noexcept {
        return size() == 0;
    }

    void swap(THashTable& ht) {
        base_type::swap(ht);
        buckets.swap(ht.buckets);
        DoSwap(num_elements, ht.num_elements);
    }

    iterator begin() {
        for (size_type n = 0; n < buckets.size(); ++n) /*y*/
            if (buckets[n])
                return iterator(buckets[n]); /*y*/
        return end();
    }

    iterator end() {
        return iterator(nullptr);
    } /*y*/

    const_iterator begin() const {
        for (size_type n = 0; n < buckets.size(); ++n) /*y*/
            if (buckets[n])
                return const_iterator(buckets[n]); /*y*/
        return end();
    }

    const_iterator end() const {
        return const_iterator(nullptr);
    } /*y*/

public:
    size_type bucket_count() const {
        return buckets.size();
    } /*y*/

    size_type bucket_size(size_type bucket) const {
        size_type result = 0;
        if (const node* cur = buckets[bucket])
            for (; !((uintptr_t)cur & 1); cur = cur->next)
                result += 1;
        return result;
    }

    template <class OtherValue>
    std::pair<iterator, bool> insert_unique(const OtherValue& obj) {
        reserve(num_elements + 1);
        return insert_unique_noresize(obj);
    }

    template <class OtherValue>
    iterator insert_equal(const OtherValue& obj) {
        reserve(num_elements + 1);
        return emplace_equal_noresize(obj);
    }

    template <typename... Args>
    iterator emplace_equal(Args&&... args) {
        reserve(num_elements + 1);
        return emplace_equal_noresize(std::forward<Args>(args)...);
    }

    template <class OtherValue>
    iterator insert_direct(const OtherValue& obj, insert_ctx ins) {
        return emplace_direct(ins, obj);
    }

    template <typename... Args>
    iterator emplace_direct(insert_ctx ins, Args&&... args) {
        bool resized = reserve(num_elements + 1);
        node* tmp = new_node(std::forward<Args>(args)...);
        if (resized) {
            find_i(get_key(tmp->val), ins);
        }
        tmp->next = *ins ? *ins : (node*)((uintptr_t)(ins + 1) | 1);
        *ins = tmp;
        ++num_elements;
        return iterator(tmp);
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace_unique(Args&&... args) {
        reserve(num_elements + 1);
        return emplace_unique_noresize(std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace_unique_noresize(Args&&... args);

    template <class OtherValue>
    std::pair<iterator, bool> insert_unique_noresize(const OtherValue& obj);

    template <typename... Args>
    iterator emplace_equal_noresize(Args&&... args);

    template <class InputIterator>
    void insert_unique(InputIterator f, InputIterator l) {
        insert_unique(f, l, typename std::iterator_traits<InputIterator>::iterator_category());
    }

    template <class InputIterator>
    void insert_equal(InputIterator f, InputIterator l) {
        insert_equal(f, l, typename std::iterator_traits<InputIterator>::iterator_category());
    }

    template <class InputIterator>
    void insert_unique(InputIterator f, InputIterator l, std::input_iterator_tag) {
        for (; f != l; ++f)
            insert_unique(*f);
    }

    template <class InputIterator>
    void insert_equal(InputIterator f, InputIterator l, std::input_iterator_tag) {
        for (; f != l; ++f)
            insert_equal(*f);
    }

    template <class ForwardIterator>
    void insert_unique(ForwardIterator f, ForwardIterator l, std::forward_iterator_tag) {
        difference_type n = std::distance(f, l);

        reserve(num_elements + n);
        for (; n > 0; --n, ++f)
            insert_unique_noresize(*f);
    }

    template <class ForwardIterator>
    void insert_equal(ForwardIterator f, ForwardIterator l, std::forward_iterator_tag) {
        difference_type n = std::distance(f, l);

        reserve(num_elements + n);
        for (; n > 0; --n, ++f)
            emplace_equal_noresize(*f);
    }

    template <class OtherValue>
    reference find_or_insert(const OtherValue& v);

    template <class OtherKey>
    iterator find(const OtherKey& key) {
        size_type n = bkt_num_key(key);
        node* first;
        for (first = buckets[n];
             first && !equals(get_key(first->val), key);
             first = ((uintptr_t)first->next & 1) ? nullptr : first->next) /*y*/
        {
        }
        return iterator(first); /*y*/
    }

    template <class OtherKey>
    const_iterator find(const OtherKey& key) const {
        size_type n = bkt_num_key(key);
        const node* first;
        for (first = buckets[n];
             first && !equals(get_key(first->val), key);
             first = ((uintptr_t)first->next & 1) ? nullptr : first->next) /*y*/
        {
        }
        return const_iterator(first); /*y*/
    }

    template <class OtherKey>
    iterator find_i(const OtherKey& key, insert_ctx& ins);

    template <class OtherKey>
    size_type count(const OtherKey& key) const {
        const size_type n = bkt_num_key(key);
        size_type result = 0;

        if (const node* cur = buckets[n])
            for (; !((uintptr_t)cur & 1); cur = cur->next)
                if (equals(get_key(cur->val), key))
                    ++result;
        return result;
    }

    template <class OtherKey>
    std::pair<iterator, iterator> equal_range(const OtherKey& key);

    template <class OtherKey>
    std::pair<const_iterator, const_iterator> equal_range(const OtherKey& key) const;

    template <class OtherKey>
    std::pair<iterator, iterator> equal_range_i(const OtherKey& key, insert_ctx& ins);

    template <class OtherKey>
    size_type erase(const OtherKey& key);

    template <class OtherKey>
    size_type erase_one(const OtherKey& key);

    // void (instead of iterator) is intended, see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2023.pdf
    void erase(const iterator& it);
    void erase(iterator first, iterator last);

    void erase(const const_iterator& it);
    void erase(const_iterator first, const_iterator last);

    bool reserve(size_type num_elements_hint);
    Y_REINITIALIZES_OBJECT void basic_clear();

    /**
     * Clears the hashtable without deallocating the nodes.
     *
     * This might come in handy with non-standard allocators, e.g. a pool
     * allocator with a pool that is then cleared manually, thus releasing all
     * the nodes at once.
     */
    void release_nodes() {
        if (empty())
            return; /* Need this check because empty buckets may reside in read-only memory. */

        clear_buckets(buckets);
        num_elements = 0;
    }

    // implemented in save_stl.h
    template <class KeySaver>
    int save_for_st(IOutputStream* stream, KeySaver& ks, sthash<int, int, THash<int>, TEqualTo<int>, typename KeySaver::TSizeType>* stHash = nullptr) const;

    Y_REINITIALIZES_OBJECT void clear(size_type downsize) {
        basic_clear();

        if (downsize < buckets.size()) {
            const TBucketDivisor newSize = HashBucketCountExt(downsize);
            if (newSize() < buckets.size()) {
                Y_ASSERT(newSize() >= 7); /* We cannot downsize static buckets. */
                buckets.resize_noallocate(newSize);
            }
        }
    }

    /**
     * Clears the hashtable and tries to reasonably downsize it. Note that
     * downsizing is mainly for the following use case:
     *
     *     THashTable hash;
     *     for(...) {
     *         if (someCond())
     *             hash.clear();
     *         hash.insert(...);
     *     }
     *
     * Here if at some point `hash` gets really big, then all the following calls
     * to `clear` become really slow as they have to iterate through all the the
     * empty buckets. This is worked around by squeezing the buckets array a little
     * bit with every `clear` call.
     *
     * Alternatively, the user can call `basic_clear`, which doesn't do the
     * downsizing.
     */
    Y_REINITIALIZES_OBJECT void clear() {
        if (num_elements)
            clear((num_elements * 2 + buckets.size()) / 3);
    }

private:
    static void initialize_buckets(buckets_type& buckets, size_type sizeHint) {
        if (sizeHint == 0) {
            buckets.initialize_static(reinterpret_cast<node**>(const_cast<void**>(_yhashtable_empty_data)) + 1, TBucketDivisor::One());
        } else {
            TBucketDivisor size = HashBucketCountExt(sizeHint);
            Y_ASSERT(size() >= 7);

            initialize_buckets_dynamic(buckets, size);
        }
    }

    static void initialize_buckets_dynamic(buckets_type& buckets, TBucketDivisor size) {
        buckets.initialize_dynamic(size);
        memset(buckets.data(), 0, size() * sizeof(*buckets.data()));
        buckets[size()] = (node*)1;
    }

    static void deinitialize_buckets(buckets_type& buckets) {
        if (buckets.size() == 1) {
            buckets.deinitialize_static();
        } else {
            buckets.deinitialize_dynamic();
        }
    }

    static void clear_buckets(buckets_type& buckets) {
        memset(buckets.data(), 0, buckets.size() * sizeof(*buckets.data()));
    }

    template <class OtherKey>
    size_type bkt_num_key(const OtherKey& key) const {
        return bkt_num_key(key, buckets.ExtSize());
    }

    template <class OtherValue>
    size_type bkt_num(const OtherValue& obj) const {
        return bkt_num_key(get_key(obj));
    }

    template <class OtherKey>
    size_type bkt_num_key(const OtherKey& key, TBucketDivisor n) const {
        const size_type bucket = n.Remainder(this->_get_hash_fun()(key));
        Y_ASSERT((0 <= bucket) && (bucket < n()));
        return bucket;
    }

    template <class OtherValue>
    size_type bkt_num(const OtherValue& obj, TBucketDivisor n) const {
        return bkt_num_key(get_key(obj), n);
    }

    template <typename... Args>
    node* new_node(Args&&... val) {
        node* n = get_node();
        n->next = (node*)1; /*y*/ // just for a case
        try {
            new (static_cast<void*>(&n->val)) Value(std::forward<Args>(val)...);
        } catch (...) {
            put_node(n);
            throw;
        }
        return n;
    }

    void delete_node(node* n) {
        n->val.~Value();
        // n->next = (node*) 0xDeadBeeful;
        put_node(n);
    }

    void erase_bucket(const size_type n, node* first, node* last);
    void erase_bucket(const size_type n, node* last);

    void copy_from_dynamic(const THashTable& ht);
};

template <class V>
__yhashtable_iterator<V>& __yhashtable_iterator<V>::operator++() {
    Y_ASSERT(cur);
    cur = cur->next;
    if ((uintptr_t)cur & 1) {
        node** bucket = (node**)((uintptr_t)cur & ~1);
        while (*bucket == nullptr)
            ++bucket;
        Y_ASSERT(*bucket != nullptr);
        cur = (node*)((uintptr_t)*bucket & ~1);
    }
    return *this;
}

template <class V>
inline __yhashtable_iterator<V> __yhashtable_iterator<V>::operator++(int) {
    iterator tmp = *this;
    ++*this;
    return tmp;
}

template <class V>
__yhashtable_const_iterator<V>& __yhashtable_const_iterator<V>::operator++() {
    Y_ASSERT(cur);
    cur = cur->next;
    if ((uintptr_t)cur & 1) {
        node** bucket = (node**)((uintptr_t)cur & ~1);
        while (*bucket == nullptr)
            ++bucket;
        Y_ASSERT(*bucket != nullptr);
        cur = (node*)((uintptr_t)*bucket & ~1);
    }
    return *this;
}

template <class V>
inline __yhashtable_const_iterator<V> __yhashtable_const_iterator<V>::operator++(int) {
    const_iterator tmp = *this;
    ++*this;
    return tmp;
}

template <class V, class K, class HF, class Ex, class Eq, class A>
template <typename... Args>
std::pair<typename THashTable<V, K, HF, Ex, Eq, A>::iterator, bool> THashTable<V, K, HF, Ex, Eq, A>::emplace_unique_noresize(Args&&... args) {
    auto deleter = [&](node* tmp) { delete_node(tmp); };
    node* tmp = new_node(std::forward<Args>(args)...);
    std::unique_ptr<node, decltype(deleter)> guard(tmp, deleter);

    const size_type n = bkt_num(tmp->val);
    node* first = buckets[n];

    if (first)                                                          /*y*/
        for (node* cur = first; !((uintptr_t)cur & 1); cur = cur->next) /*y*/
            if (equals(get_key(cur->val), get_key(tmp->val)))
                return std::pair<iterator, bool>(iterator(cur), false); /*y*/

    guard.release();
    tmp->next = first ? first : (node*)((uintptr_t)&buckets[n + 1] | 1); /*y*/
    buckets[n] = tmp;
    ++num_elements;
    return std::pair<iterator, bool>(iterator(tmp), true); /*y*/
}

template <class V, class K, class HF, class Ex, class Eq, class A>
template <class OtherValue>
std::pair<typename THashTable<V, K, HF, Ex, Eq, A>::iterator, bool> THashTable<V, K, HF, Ex, Eq, A>::insert_unique_noresize(const OtherValue& obj) {
    const size_type n = bkt_num(obj);
    node* first = buckets[n];

    if (first)                                                          /*y*/
        for (node* cur = first; !((uintptr_t)cur & 1); cur = cur->next) /*y*/
            if (equals(get_key(cur->val), get_key(obj)))
                return std::pair<iterator, bool>(iterator(cur), false); /*y*/

    node* tmp = new_node(obj);
    tmp->next = first ? first : (node*)((uintptr_t)&buckets[n + 1] | 1); /*y*/
    buckets[n] = tmp;
    ++num_elements;
    return std::pair<iterator, bool>(iterator(tmp), true); /*y*/
}

template <class V, class K, class HF, class Ex, class Eq, class A>
template <typename... Args>
__yhashtable_iterator<V> THashTable<V, K, HF, Ex, Eq, A>::emplace_equal_noresize(Args&&... args) {
    auto deleter = [&](node* tmp) { delete_node(tmp); };
    node* tmp = new_node(std::forward<Args>(args)...);
    std::unique_ptr<node, decltype(deleter)> guard(tmp, deleter);
    const size_type n = bkt_num(tmp->val);
    node* first = buckets[n];

    if (first)                                                          /*y*/
        for (node* cur = first; !((uintptr_t)cur & 1); cur = cur->next) /*y*/
            if (equals(get_key(cur->val), get_key(tmp->val))) {
                guard.release();
                tmp->next = cur->next;
                cur->next = tmp;
                ++num_elements;
                return iterator(tmp); /*y*/
            }

    guard.release();
    tmp->next = first ? first : (node*)((uintptr_t)&buckets[n + 1] | 1); /*y*/
    buckets[n] = tmp;
    ++num_elements;
    return iterator(tmp); /*y*/
}

template <class V, class K, class HF, class Ex, class Eq, class A>
template <class OtherValue>
typename THashTable<V, K, HF, Ex, Eq, A>::reference THashTable<V, K, HF, Ex, Eq, A>::find_or_insert(const OtherValue& v) {
    reserve(num_elements + 1);

    size_type n = bkt_num_key(get_key(v));
    node* first = buckets[n];

    if (first)                                                          /*y*/
        for (node* cur = first; !((uintptr_t)cur & 1); cur = cur->next) /*y*/
            if (equals(get_key(cur->val), get_key(v)))
                return cur->val;

    node* tmp = new_node(v);
    tmp->next = first ? first : (node*)((uintptr_t)&buckets[n + 1] | 1); /*y*/
    buckets[n] = tmp;
    ++num_elements;
    return tmp->val;
}

template <class V, class K, class HF, class Ex, class Eq, class A>
template <class OtherKey>
__yhashtable_iterator<V> THashTable<V, K, HF, Ex, Eq, A>::find_i(const OtherKey& key, insert_ctx& ins) {
    size_type n = bkt_num_key(key);
    ins = &buckets[n];
    node* first = buckets[n];

    if (first)                                                          /*y*/
        for (node* cur = first; !((uintptr_t)cur & 1); cur = cur->next) /*y*/
            if (equals(get_key(cur->val), key))
                return iterator(cur); /*y*/
    return end();
}

template <class V, class K, class HF, class Ex, class Eq, class A>
template <class OtherKey>
std::pair<__yhashtable_iterator<V>, __yhashtable_iterator<V>> THashTable<V, K, HF, Ex, Eq, A>::equal_range(const OtherKey& key) {
    insert_ctx ctx;
    return equal_range_i(key, ctx);
}

template <class V, class K, class HF, class Ex, class Eq, class A>
template <class OtherKey>
std::pair<__yhashtable_iterator<V>, __yhashtable_iterator<V>> THashTable<V, K, HF, Ex, Eq, A>::equal_range_i(const OtherKey& key, insert_ctx& ins) {
    using pii = std::pair<iterator, iterator>;
    const size_type n = bkt_num_key(key);
    ins = &buckets[n];
    node* first = buckets[n];

    if (first)                                                 /*y*/
        for (; !((uintptr_t)first & 1); first = first->next) { /*y*/
            if (equals(get_key(first->val), key)) {
                for (node* cur = first->next; !((uintptr_t)cur & 1); cur = cur->next)
                    if (!equals(get_key(cur->val), key))
                        return pii(iterator(first), iterator(cur)); /*y*/
                for (size_type m = n + 1; m < buckets.size(); ++m)  /*y*/
                    if (buckets[m])
                        return pii(iterator(first),       /*y*/
                                   iterator(buckets[m])); /*y*/
                return pii(iterator(first), end());       /*y*/
            }
        }
    return pii(end(), end());
}

template <class V, class K, class HF, class Ex, class Eq, class A>
template <class OtherKey>
std::pair<__yhashtable_const_iterator<V>, __yhashtable_const_iterator<V>> THashTable<V, K, HF, Ex, Eq, A>::equal_range(const OtherKey& key) const {
    using pii = std::pair<const_iterator, const_iterator>;
    const size_type n = bkt_num_key(key);
    const node* first = buckets[n];

    if (first)                                                 /*y*/
        for (; !((uintptr_t)first & 1); first = first->next) { /*y*/
            if (equals(get_key(first->val), key)) {
                for (const node* cur = first->next; !((uintptr_t)cur & 1); cur = cur->next)
                    if (!equals(get_key(cur->val), key))
                        return pii(const_iterator(first),          /*y*/
                                   const_iterator(cur));           /*y*/
                for (size_type m = n + 1; m < buckets.size(); ++m) /*y*/
                    if (buckets[m])
                        return pii(const_iterator(first /*y*/),
                                   const_iterator(buckets[m] /*y*/));
                return pii(const_iterator(first /*y*/), end());
            }
        }
    return pii(end(), end());
}

template <class V, class K, class HF, class Ex, class Eq, class A>
template <class OtherKey>
typename THashTable<V, K, HF, Ex, Eq, A>::size_type THashTable<V, K, HF, Ex, Eq, A>::erase(const OtherKey& key) {
    const size_type n = bkt_num_key(key);
    node* first = buckets[n];
    size_type erased = 0;

    if (first) {
        node* cur = first;
        node* next = cur->next;
        while (!((uintptr_t)next & 1)) { /*y*/
            if (equals(get_key(next->val), key)) {
                cur->next = next->next;
                ++erased;
                --num_elements;
                delete_node(next);
                next = cur->next;
            } else {
                cur = next;
                next = cur->next;
            }
        }
        if (equals(get_key(first->val), key)) {
            buckets[n] = ((uintptr_t)first->next & 1) ? nullptr : first->next; /*y*/
            ++erased;
            --num_elements;
            delete_node(first);
        }
    }
    return erased;
}

template <class V, class K, class HF, class Ex, class Eq, class A>
template <class OtherKey>
typename THashTable<V, K, HF, Ex, Eq, A>::size_type THashTable<V, K, HF, Ex, Eq, A>::erase_one(const OtherKey& key) {
    const size_type n = bkt_num_key(key);
    node* first = buckets[n];

    if (first) {
        node* cur = first;
        node* next = cur->next;
        while (!((uintptr_t)next & 1)) { /*y*/
            if (equals(get_key(next->val), key)) {
                cur->next = next->next;
                --num_elements;
                delete_node(next);
                return 1;
            } else {
                cur = next;
                next = cur->next;
            }
        }
        if (equals(get_key(first->val), key)) {
            buckets[n] = ((uintptr_t)first->next & 1) ? nullptr : first->next; /*y*/
            --num_elements;
            delete_node(first);
            return 1;
        }
    }
    return 0;
}

template <class V, class K, class HF, class Ex, class Eq, class A>
void THashTable<V, K, HF, Ex, Eq, A>::erase(const iterator& it) {
    if (node* const p = it.cur) {
        const size_type n = bkt_num(p->val);
        node* cur = buckets[n];

        if (cur == p) {
            buckets[n] = ((uintptr_t)cur->next & 1) ? nullptr : cur->next; /*y*/
            delete_node(cur);
            --num_elements;
        } else {
            node* next = cur->next;
            while (!((uintptr_t)next & 1)) {
                if (next == p) {
                    cur->next = next->next;
                    delete_node(next);
                    --num_elements;
                    break;
                } else {
                    cur = next;
                    next = cur->next;
                }
            }
        }
    }
}

template <class V, class K, class HF, class Ex, class Eq, class A>
void THashTable<V, K, HF, Ex, Eq, A>::erase(iterator first, iterator last) {
    size_type f_bucket = first.cur ? bkt_num(first.cur->val) : buckets.size(); /*y*/
    size_type l_bucket = last.cur ? bkt_num(last.cur->val) : buckets.size();   /*y*/

    if (first.cur == last.cur)
        return;
    else if (f_bucket == l_bucket)
        erase_bucket(f_bucket, first.cur, last.cur);
    else {
        erase_bucket(f_bucket, first.cur, nullptr);
        for (size_type n = f_bucket + 1; n < l_bucket; ++n)
            erase_bucket(n, nullptr);
        if (l_bucket != buckets.size()) /*y*/
            erase_bucket(l_bucket, last.cur);
    }
}

template <class V, class K, class HF, class Ex, class Eq, class A>
inline void
THashTable<V, K, HF, Ex, Eq, A>::erase(const_iterator first, const_iterator last) {
    erase(iterator(const_cast<node*>(first.cur)), iterator(const_cast<node*>(last.cur)));
}

template <class V, class K, class HF, class Ex, class Eq, class A>
inline void THashTable<V, K, HF, Ex, Eq, A>::erase(const const_iterator& it) {
    erase(iterator(const_cast<node*>(it.cur)));
}

template <class V, class K, class HF, class Ex, class Eq, class A>
bool THashTable<V, K, HF, Ex, Eq, A>::reserve(size_type num_elements_hint) {
    const size_type old_n = buckets.size(); /*y*/
    if (num_elements_hint + 1 > old_n) {
        if (old_n != 1 && num_elements_hint <= old_n) // TODO: this if is for backwards compatibility down to order-in-buckets level. Can be safely removed.
            return false;

        const TBucketDivisor n = HashBucketCountExt(num_elements_hint + 1, buckets.BucketDivisorHint() + 1);
        if (n() > old_n) {
            buckets_type tmp(buckets.get_allocator());
            initialize_buckets_dynamic(tmp, n);
#ifdef __STL_USE_EXCEPTIONS
            try {
#endif /* __STL_USE_EXCEPTIONS */
                for (size_type bucket = 0; bucket < old_n; ++bucket) {
                    node* first = buckets[bucket];
                    while (first) {
                        size_type new_bucket = bkt_num(first->val, n);
                        node* next = first->next;
                        buckets[bucket] = ((uintptr_t)next & 1) ? nullptr : next; /*y*/
                        next = tmp[new_bucket];
                        first->next = next ? next : (node*)((uintptr_t) & (tmp[new_bucket + 1]) | 1); /*y*/
                        tmp[new_bucket] = first;
                        first = buckets[bucket];
                    }
                }

                buckets.swap(tmp);
                deinitialize_buckets(tmp);

                return true;
#ifdef __STL_USE_EXCEPTIONS
            } catch (...) {
                for (size_type bucket = 0; bucket < tmp.size() - 1; ++bucket) {
                    while (tmp[bucket]) {
                        node* next = tmp[bucket]->next;
                        delete_node(tmp[bucket]);
                        tmp[bucket] = ((uintptr_t)next & 1) ? nullptr : next /*y*/;
                    }
                }
                throw;
            }
#endif /* __STL_USE_EXCEPTIONS */
        }
    }

    return false;
}

template <class V, class K, class HF, class Ex, class Eq, class A>
void THashTable<V, K, HF, Ex, Eq, A>::erase_bucket(const size_type n, node* first, node* last) {
    node* cur = buckets[n];
    if (cur == first)
        erase_bucket(n, last);
    else {
        node* next;
        for (next = cur->next; next != first; cur = next, next = cur->next)
            ;
        while (next != last) { /*y; 3.1*/
            cur->next = next->next;
            delete_node(next);
            next = ((uintptr_t)cur->next & 1) ? nullptr : cur->next; /*y*/
            --num_elements;
        }
    }
}

template <class V, class K, class HF, class Ex, class Eq, class A>
void THashTable<V, K, HF, Ex, Eq, A>::erase_bucket(const size_type n, node* last) {
    node* cur = buckets[n];
    while (cur != last) {
        node* next = cur->next;
        delete_node(cur);
        cur = ((uintptr_t)next & 1) ? nullptr : next; /*y*/
        buckets[n] = cur;
        --num_elements;
    }
}

template <class V, class K, class HF, class Ex, class Eq, class A>
void THashTable<V, K, HF, Ex, Eq, A>::basic_clear() {
    if (!num_elements) {
        return;
    }

    node** first = buckets.begin();
    node** last = buckets.end();
    for (; first < last; ++first) {
        node* cur = *first;
        if (cur) {                          /*y*/
            while (!((uintptr_t)cur & 1)) { /*y*/
                node* next = cur->next;
                delete_node(cur);
                cur = next;
            }
            *first = nullptr;
        }
    }
    num_elements = 0;
}

template <class V, class K, class HF, class Ex, class Eq, class A>
void THashTable<V, K, HF, Ex, Eq, A>::copy_from_dynamic(const THashTable& ht) {
    Y_ASSERT(buckets.size() == ht.buckets.size() && !ht.empty());

#ifdef __STL_USE_EXCEPTIONS
    try {
#endif                                                      /* __STL_USE_EXCEPTIONS */
        for (size_type i = 0; i < ht.buckets.size(); ++i) { /*y*/
            if (const node* cur = ht.buckets[i]) {
                node* copy = new_node(cur->val);
                buckets[i] = copy;

                for (node* next = cur->next; !((uintptr_t)next & 1); cur = next, next = cur->next) {
                    copy->next = new_node(next->val);
                    copy = copy->next;
                }
                copy->next = (node*)((uintptr_t)&buckets[i + 1] | 1); /*y*/
            }
        }
        num_elements = ht.num_elements;
#ifdef __STL_USE_EXCEPTIONS
    } catch (...) {
        basic_clear();
        throw;
    }
#endif /* __STL_USE_EXCEPTIONS */
}

namespace NPrivate {
    template <class Key>
    inline TString MapKeyToString(const Key&) {
        return TypeName<Key>();
    }

    TString MapKeyToString(TStringBuf key);
    TString MapKeyToString(unsigned short key);
    TString MapKeyToString(short key);
    TString MapKeyToString(unsigned int key);
    TString MapKeyToString(int key);
    TString MapKeyToString(unsigned long key);
    TString MapKeyToString(long key);
    TString MapKeyToString(unsigned long long key);
    TString MapKeyToString(long long key);

    inline TString MapKeyToString(const TString& key) {
        return MapKeyToString(TStringBuf(key));
    }

    inline TString MapKeyToString(const char* key) {
        return MapKeyToString(TStringBuf(key));
    }

    inline TString MapKeyToString(char* key) {
        return MapKeyToString(TStringBuf(key));
    }

    [[noreturn]] void ThrowKeyNotFoundInHashTableException(const TStringBuf keyRepresentation);
}

// Cannot name it just 'Hash' because it clashes with too many class members in the code.
template <class T>
size_t ComputeHash(const T& value) {
    return THash<T>{}(value);
}
