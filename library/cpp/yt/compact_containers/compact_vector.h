#pragma once

#include <util/system/defaults.h>

#include <cstdint>
#include <iterator>
#include <limits>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TCompactVectorOnHeapStorage;

//! A vector-like structure optimized for storing elements inline
//! and with little memory overhead.
/*!
 *  Stores up to #N (<= 254) elements inline.
 *
 *  When capacity starts exceeding #N, moves all elements to heap;
 *  \see #TCompactVectorOnHeapStorage.
 *
 *  When linked with YTAlloc, employs its API to adjust the on-heap capacity in accordance
 *  to the actual size of the allocated region.
 *
 *  Assuming the entropy and the alignment constraints, yields a seemingly optimal memory overhead.
 *  E.g. TCompactVector<uint8_t, 7> takes 8 bytes and TCompactVector<uint32_t, 3> takes 16 bytes.
 *  \see #ByteSize.
 *
 *  Assumes (and asserts) the following:
 *  1) the platform is 64 bit;
 *  2) the highest 8 bits of pointers returned by |malloc| are zeroes;
 *  3) the platform is little-endian.
 */
template <class T, size_t N>
class TCompactVector
{
public:
    static_assert(N < std::numeric_limits<uint8_t>::max());

    using size_type = size_t;
    using difference_type = ptrdiff_t;

    using value_type = T;

    using iterator = T*;
    using const_iterator = const T*;

    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using reverse_iterator = std::reverse_iterator<iterator>;

    using reference = T&;
    using const_reference = const T&;

    using pointer = T*;
    using const_pointer = const T*;

    TCompactVector() noexcept;
    TCompactVector(const TCompactVector& other);
    template <size_t OtherN>
    TCompactVector(const TCompactVector<T, OtherN>& other);
    TCompactVector(TCompactVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>);
    template <size_t OtherN>
    TCompactVector(TCompactVector<T, OtherN>&& other);
    explicit TCompactVector(size_type count);
    TCompactVector(size_type count, const T& value);
    template <class TIterator>
    TCompactVector(TIterator first, TIterator last);
    TCompactVector(std::initializer_list<T> list);

    ~TCompactVector();

    [[nodiscard]] bool empty() const;

    iterator begin();
    const_iterator begin() const;
    iterator end();
    const_iterator end() const;

    reverse_iterator rbegin();
    const_reverse_iterator rbegin() const;
    reverse_iterator rend();
    const_reverse_iterator rend() const;

    size_type size() const;
    size_type capacity() const;
    size_type max_size() const;

    pointer data();
    const_pointer data() const;

    reference operator[](size_type index);
    const_reference operator[](size_type index) const;

    reference front();
    const_reference front() const;
    reference back();
    const_reference back() const;

    void push_back(const T& value);
    void push_back(T&& value);

    template <class... TArgs>
    iterator emplace(const_iterator pos, TArgs&&... args);
    template <class... TArgs>
    reference emplace_back(TArgs&&... args);

    void pop_back();

    iterator erase(const_iterator pos);
    iterator erase(const_iterator first, const_iterator last);

    void clear();

    void resize(size_type newSize);
    void resize(size_type newSize, const T& value);

    void reserve(size_type newCapacity);

    void swap(TCompactVector& other);

    void assign(size_type count, const T& value);
    template <class TIterator>
    void assign(TIterator first, TIterator last);
    void assign(std::initializer_list<T> list);
    template <size_t OtherN>
    void assign(const TCompactVector<T, OtherN>& other);
    template <size_t OtherN>
    void assign(TCompactVector<T, OtherN>&& other);

    TCompactVector& operator=(const TCompactVector& other);
    template <size_t OtherN>
    TCompactVector& operator=(const TCompactVector<T, OtherN>& other);
    TCompactVector& operator=(TCompactVector&& other);
    template <size_t OtherN>
    TCompactVector& operator=(TCompactVector<T, OtherN>&& other);
    TCompactVector& operator=(std::initializer_list<T> list);

    iterator insert(const_iterator pos, const T& value);
    iterator insert(const_iterator pos, T&& value);
    iterator insert(const_iterator pos, size_type count, const T& value);
    template <class TIterator>
    iterator insert(const_iterator pos, TIterator first, TIterator last);
    iterator insert(const_iterator pos, std::initializer_list<T> list);

    void shrink_to_small();

private:
    template <class OtherT, size_t OtherN>
    friend class TCompactVector;

    using TOnHeapStorage = TCompactVectorOnHeapStorage<T>;

    static constexpr size_t ByteSize =
        (sizeof(T) * N + alignof(T) + sizeof(uintptr_t) - 1) &
        ~(sizeof(uintptr_t) - 1);

    struct TInlineMeta
    {
        char Padding[ByteSize - sizeof(uint8_t)];
        //  > 0 indicates inline storage
        // == 0 indicates on-heap storage
        uint8_t SizePlusOne;
    } alias_hack;

    // TODO(aleexfi): Use [[no_unique_address]] when clang will support it on windows.
    template <class = void>
    struct TOnHeapMeta
    {
        char Padding[ByteSize - sizeof(uintptr_t)];
        TOnHeapStorage* Storage;
    } alias_hack;

    template <class _>
        requires (ByteSize == sizeof(uintptr_t))
    struct TOnHeapMeta<_>
    {
        TOnHeapStorage* Storage;
    } alias_hack;

    static_assert(sizeof(TOnHeapMeta<>) == ByteSize);

    union
    {
        T InlineElements_[N];
        TInlineMeta InlineMeta_;
        TOnHeapMeta<> OnHeapMeta_;
    };

    bool IsInline() const;
    void SetSize(size_t newSize);
    void EnsureOnHeapCapacity(size_t newCapacity, bool incremental);
    template <class TPtr, class F>
    reference PushBackImpl(TPtr valuePtr, F&& func);
    template <class F>
    void ResizeImpl(size_t newSize, F&& func);
    template <class TPtr, class UninitializedF, class InitializedF>
    iterator InsertOneImpl(const_iterator pos, TPtr valuePtr, UninitializedF&& uninitializedFunc, InitializedF&& initializedFunc);
    template <class UninitializedF, class InitializedF>
    iterator InsertManyImpl(const_iterator pos, size_t insertCount, UninitializedF&& uninitializedFunc, InitializedF&& initializedFunc);

    static void Destroy(T* first, T* last);
    template <class T1, class T2>
    static void Copy(const T1* srcFirst, const T1* srcLast, T2* dst);
    template <class T1, class T2>
    static void UninitializedCopy(const T1* srcFirst, const T1* srcLast, T2* dst);
    static void Move(T* srcFirst, T* srcLast, T* dst);
    static void MoveBackward(T* srcFirst, T* srcLast, T* dst);
    static void UninitializedMove(T* srcFirst, T* srcLast, T* dst);
};

////////////////////////////////////////////////////////////////////////////////

template <class T, size_t LhsN, size_t RhsN>
bool operator==(const TCompactVector<T, LhsN>& lhs, const TCompactVector<T, RhsN>& rhs);

template <class T, size_t LhsN, size_t RhsN>
bool operator!=(const TCompactVector<T, LhsN>& lhs, const TCompactVector<T, RhsN>& rhs);

template <class T, size_t LhsN, size_t RhsN>
bool operator<(const TCompactVector<T, LhsN>& lhs, const TCompactVector<T, RhsN>& rhs);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define COMPACT_VECTOR_INL_H_
#include "compact_vector-inl.h"
#undef COMPACT_VECTOR_INL_H_
