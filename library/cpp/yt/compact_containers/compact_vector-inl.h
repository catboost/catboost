#ifndef COMPACT_VECTOR_INL_H_
#error "Direct inclusion of this file is not allowed, include compact_vector.h"
// For the sake of sane code completion.
#include "compact_vector.h"
#endif
#undef COMPACT_VECTOR_INL_H_

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/malloc/malloc.h>

#include <library/cpp/yt/misc/hash.h>

#include <util/system/compiler.h>

#include <algorithm>
#include <bit>

#include <string.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

static_assert(sizeof(uintptr_t) == 8);
static_assert(std::endian::native == std::endian::little);

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TCompactVectorOnHeapStorage
{
    T* End;
    T* Capacity;
    T Elements[0];
};

////////////////////////////////////////////////////////////////////////////////

template <class TVector, class TPtr>
class TCompactVectorReallocationPtrAdjuster
{
public:
    TCompactVectorReallocationPtrAdjuster(TVector* vector, TPtr& ptr)
        : Vector_(vector)
        , Ptr_(ptr)
        , Index_(ptr >= Vector_->begin() && ptr <= Vector_->end()
            ? std::distance(Vector_->begin(), const_cast<typename TVector::iterator>(ptr))
            : -1)
    { }

    ~TCompactVectorReallocationPtrAdjuster()
    {
        if (Index_ >= 0) {
            Ptr_ = Vector_->begin() + Index_;
        }
    }

private:
    TVector* const Vector_;
    TPtr& Ptr_;
    const ptrdiff_t Index_;
};

template <class TVector>
class TCompactVectorReallocationPtrAdjuster<TVector, std::nullptr_t>
{
public:
    TCompactVectorReallocationPtrAdjuster(TVector* /*vector*/, std::nullptr_t /*ptr*/)
    { }
};

////////////////////////////////////////////////////////////////////////////////

template <class T, size_t N>
TCompactVector<T, N>::TCompactVector() noexcept
{
    InlineMeta_.SizePlusOne = 1;
}

template <class T, size_t N>
TCompactVector<T, N>::TCompactVector(const TCompactVector& other)
    : TCompactVector()
{
    assign(other.begin(), other.end());
}

template <class T, size_t N>
template <size_t OtherN>
TCompactVector<T, N>::TCompactVector(const TCompactVector<T, OtherN>& other)
    : TCompactVector()
{
    assign(other.begin(), other.end());
}

template <class T, size_t N>
TCompactVector<T, N>::TCompactVector(TCompactVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
    : TCompactVector()
{
    swap(other);
}

template <class T, size_t N>
template <size_t OtherN>
TCompactVector<T, N>::TCompactVector(TCompactVector<T, OtherN>&& other)
    : TCompactVector()
{
    swap(other);
}

template <class T, size_t N>
TCompactVector<T, N>::TCompactVector(size_type count)
    : TCompactVector()
{
    assign(count, T());
}

template <class T, size_t N>
TCompactVector<T, N>::TCompactVector(size_type count, const T& value)
    : TCompactVector()
{
    assign(count, value);
}

template <class T, size_t N>
template <class TIterator>
TCompactVector<T, N>::TCompactVector(TIterator first, TIterator last)
    : TCompactVector()
{
    assign(first, last);
}

template <class T, size_t N>
TCompactVector<T, N>::TCompactVector(std::initializer_list<T> list)
    : TCompactVector()
{
    assign(list.begin(), list.end());
}

template <class T, size_t N>
TCompactVector<T, N>::~TCompactVector()
{
    if (Y_LIKELY(IsInline())) {
        Destroy(&InlineElements_[0], &InlineElements_[InlineMeta_.SizePlusOne - 1]);
    } else {
        auto* storage = OnHeapMeta_.Storage;
        Destroy(storage->Elements, storage->End);
        ::free(storage);
    }
}

template <class T, size_t N>
bool TCompactVector<T, N>::empty() const
{
    if (Y_LIKELY(IsInline())) {
        return InlineMeta_.SizePlusOne == 1;
    } else {
        const auto* storage = OnHeapMeta_.Storage;
        return storage->Elements == storage->End;
    }
}

template <class T, size_t N>
auto TCompactVector<T, N>::begin() -> iterator
{
    return Y_LIKELY(IsInline()) ? &InlineElements_[0] : OnHeapMeta_.Storage->Elements;
}

template <class T, size_t N>
auto TCompactVector<T, N>::begin() const -> const_iterator
{
    return const_cast<TCompactVector*>(this)->begin();
}

template <class T, size_t N>
auto TCompactVector<T, N>::end() -> iterator
{
    return Y_LIKELY(IsInline()) ? &InlineElements_[InlineMeta_.SizePlusOne - 1] : OnHeapMeta_.Storage->End;
}

template <class T, size_t N>
auto TCompactVector<T, N>::end() const -> const_iterator
{
    return const_cast<TCompactVector*>(this)->end();
}

template <class T, size_t N>
auto TCompactVector<T, N>::rbegin() -> reverse_iterator
{
    return static_cast<reverse_iterator>(end());
}

template <class T, size_t N>
auto TCompactVector<T, N>::rbegin() const -> const_reverse_iterator
{
    return static_cast<const_reverse_iterator>(end());
}

template <class T, size_t N>
auto TCompactVector<T, N>::rend() -> reverse_iterator
{
    return static_cast<reverse_iterator>(begin());
}

template <class T, size_t N>
auto TCompactVector<T, N>::rend() const -> const_reverse_iterator
{
    return static_cast<const_reverse_iterator>(begin());
}

template <class T, size_t N>
auto TCompactVector<T, N>::size() const -> size_type
{
    if (Y_LIKELY(IsInline())) {
        return InlineMeta_.SizePlusOne - 1;
    } else {
        const auto* storage = OnHeapMeta_.Storage;
        return storage->End - storage->Elements;
    }
}

template <class T, size_t N>
auto TCompactVector<T, N>::capacity() const -> size_type
{
    if (Y_LIKELY(IsInline())) {
        return N;
    } else {
        const auto* storage = OnHeapMeta_.Storage;
        return storage->Capacity - storage->Elements;
    }
}

template <class T, size_t N>
auto TCompactVector<T, N>::max_size() const -> size_type
{
    return static_cast<size_type>(-1) / sizeof(T);
}

template <class T, size_t N>
auto TCompactVector<T, N>::data() -> pointer
{
    return static_cast<pointer>(begin());
}

template <class T, size_t N>
auto TCompactVector<T, N>::data() const -> const_pointer
{
    return static_cast<const_pointer>(begin());
}

template <class T, size_t N>
auto TCompactVector<T, N>::operator[](size_type index) -> reference
{
    YT_ASSERT(index < size());
    return begin()[index];
}

template <class T, size_t N>
auto TCompactVector<T, N>::operator[](size_type index) const -> const_reference
{
    return const_cast<TCompactVector*>(this)->operator[](index);
}

template <class T, size_t N>
auto TCompactVector<T, N>::front() -> reference
{
    YT_ASSERT(!empty());
    return begin()[0];
}

template <class T, size_t N>
auto TCompactVector<T, N>::front() const -> const_reference
{
    return const_cast<TCompactVector*>(this)->front();
}

template <class T, size_t N>
auto TCompactVector<T, N>::back() -> reference
{
    YT_ASSERT(!empty());
    return end()[-1];
}

template <class T, size_t N>
auto TCompactVector<T, N>::back() const -> const_reference
{
    return const_cast<TCompactVector*>(this)->back();
}

template <class T, size_t N>
void TCompactVector<T, N>::push_back(const T& value)
{
    PushBackImpl(
        &value,
        [&] (T* dst, const T* value) {
            ::new(dst) T(*value);
        });
}

template <class T, size_t N>
void TCompactVector<T, N>::push_back(T&& value)
{
    PushBackImpl(
        &value,
        [&] (T* dst, T* value) {
            ::new(dst) T(std::move(*value));
        });
}

template <class T, size_t N>
template <class... TArgs>
auto TCompactVector<T, N>::emplace(const_iterator pos, TArgs&&... args) -> iterator
{
    return InsertOneImpl(
        pos,
        nullptr,
        [&] (auto* dst, std::nullptr_t) {
            ::new(dst) T(std::forward<TArgs>(args)...);
        },
        [&] (auto* dst, std::nullptr_t) {
            *dst = T(std::forward<TArgs>(args)...);
        });
}

template <class T, size_t N>
template <class... TArgs>
auto TCompactVector<T, N>::emplace_back(TArgs&&... args) -> reference
{
    return PushBackImpl(
        nullptr,
        [&] (T* dst, std::nullptr_t) {
            ::new(dst) T(std::forward<TArgs>(args)...);
        });
}

template <class T, size_t N>
void TCompactVector<T, N>::pop_back()
{
    YT_ASSERT(!empty());

    if (Y_LIKELY(IsInline())) {
        InlineElements_[InlineMeta_.SizePlusOne - 2].T::~T();
        --InlineMeta_.SizePlusOne;
    } else {
        auto* storage = OnHeapMeta_.Storage;
        storage->End[-1].T::~T();
        --storage->End;
    }
}

template <class T, size_t N>
auto TCompactVector<T, N>::erase(const_iterator pos) -> iterator
{
    YT_ASSERT(pos >= begin());
    YT_ASSERT(pos < end());

    auto* mutablePos = const_cast<iterator>(pos);
    Move(mutablePos + 1, end(), mutablePos);
    pop_back();

    return mutablePos;
}

template <class T, size_t N>
auto TCompactVector<T, N>::erase(const_iterator first, const_iterator last) -> iterator
{
    YT_ASSERT(first >= begin());
    YT_ASSERT(last <= end());

    auto* mutableFirst = const_cast<iterator>(first);
    auto* mutableLast = const_cast<iterator>(last);
    auto count = std::distance(mutableFirst, mutableLast);

    if (Y_LIKELY(IsInline())) {
        auto* end = &InlineElements_[0] + InlineMeta_.SizePlusOne - 1;
        Move(mutableLast, end, mutableFirst);
        Destroy(end - count, end);
        InlineMeta_.SizePlusOne -= count;
    } else {
        auto* storage = OnHeapMeta_.Storage;
        auto* end = storage->End;
        Move(mutableLast, storage->End, mutableFirst);
        Destroy(end - count, end);
        storage->End -= count;
    }

    return mutableFirst;
}

template <class T, size_t N>
void TCompactVector<T, N>::clear()
{
    if (Y_LIKELY(IsInline())) {
        Destroy(&InlineElements_[0], &InlineElements_[InlineMeta_.SizePlusOne - 1]);
        InlineMeta_.SizePlusOne = 1;
    } else {
        auto* storage = OnHeapMeta_.Storage;
        Destroy(storage->Elements, storage->End);
        storage->End = storage->Elements;
    }
}

template <class T, size_t N>
void TCompactVector<T, N>::resize(size_type newSize)
{
    ResizeImpl(
        newSize,
        [] (auto* dst) {
            ::new(dst) T();
        });
}

template <class T, size_t N>
void TCompactVector<T, N>::resize(size_type newSize, const T& value)
{
    ResizeImpl(
        newSize,
        [&] (auto* dst) {
            ::new(dst) T(value);
        });
}

template <class T, size_t N>
void TCompactVector<T, N>::reserve(size_t newCapacity)
{
    if (Y_UNLIKELY(newCapacity > N)) {
        EnsureOnHeapCapacity(newCapacity, /*incremental*/ false);
    }
}

template <class T, size_t N>
void TCompactVector<T, N>::swap(TCompactVector& other)
{
    if (this == &other) {
        return;
    }

    if (!IsInline() && !other.IsInline()) {
        std::swap(OnHeapMeta_.Storage, other.OnHeapMeta_.Storage);
        return;
    }

    auto* lhs = this;
    auto* rhs = &other;
    if (lhs->size() < rhs->size()) {
        std::swap(lhs, rhs);
    }

    size_t rhsSize = rhs->size();
    size_t lhsSize = lhs->size();
    if (lhsSize > rhs->capacity()) {
        rhs->EnsureOnHeapCapacity(lhs->size(), /*incremental*/ false);
    }

    for (size_t index = 0; index < rhsSize; ++index) {
        std::swap((*lhs)[index], (*rhs)[index]);
    }

    UninitializedMove(lhs->begin() + rhsSize, lhs->end(), rhs->end());
    Destroy(lhs->begin() + rhsSize, lhs->end());

    rhs->SetSize(lhsSize);
    lhs->SetSize(rhsSize);
}

template <class T, size_t N>
void TCompactVector<T, N>::assign(size_type count, const T& value)
{
    clear();

    if (Y_UNLIKELY(count > capacity())) {
        EnsureOnHeapCapacity(count, /*incremental*/ false);
    }

    auto* dst = begin();
    std::uninitialized_fill(dst, dst + count, value);

    SetSize(count);
}

template <class T, size_t N>
template <class TIterator>
void TCompactVector<T, N>::assign(TIterator first, TIterator last)
{
    clear();

    auto count = std::distance(first, last);
    if (Y_UNLIKELY(count > static_cast<ptrdiff_t>(capacity()))) {
        EnsureOnHeapCapacity(count, /*incremental*/ false);
    }

    std::uninitialized_copy(first, last, begin());

    SetSize(count);
}

template <class T, size_t N>
void TCompactVector<T, N>::assign(std::initializer_list<T> list)
{
    assign(list.begin(), list.end());
}

template <class T, size_t N>
template <size_t OtherN>
void TCompactVector<T, N>::assign(const TCompactVector<T, OtherN>& other)
{
    if constexpr(N == OtherN) {
        if (this == &other) {
            return;
        }
    }

    auto otherSize = other.size();
    auto otherBegin = other.begin();

    if (capacity() >= otherSize) {
        const auto* src = other.begin();
        auto* dst = begin();

        auto thisSize = size();
        auto copySize = std::min(thisSize, otherSize);
        Copy(src, src + copySize, dst);
        src += copySize;
        dst += copySize;

        auto uninitializedCopySize = otherSize - copySize;
        UninitializedCopy(src, src + uninitializedCopySize, dst);
        // NB: src += uninitializedCopySize is not needed.
        dst += uninitializedCopySize;

        if (thisSize > otherSize) {
            Destroy(dst, end());
        }

        SetSize(otherSize);
        return;
    }

    clear();

    EnsureOnHeapCapacity(otherSize, /*incremental*/ false);

    YT_ASSERT(!IsInline());
    auto* storage = OnHeapMeta_.Storage;
    UninitializedCopy(otherBegin, otherBegin + otherSize, storage->Elements);
    storage->End = storage->Elements + otherSize;
}

template <class T, size_t N>
template <size_t OtherN>
void TCompactVector<T, N>::assign(TCompactVector<T, OtherN>&& other)
{
    if constexpr(N == OtherN) {
        if (this == &other) {
            return;
        }
    }

    clear();

    if (!other.IsInline()) {
        if (Y_UNLIKELY(!IsInline())) {
            ::free(OnHeapMeta_.Storage);
        }
        OnHeapMeta_.Storage = other.OnHeapMeta_.Storage;
        other.InlineMeta_.SizePlusOne = 1;
        return;
    }

    auto otherSize = other.size();
    if (Y_UNLIKELY(otherSize > capacity())) {
        EnsureOnHeapCapacity(otherSize, /*incremental*/ false);
    }

    auto* otherBegin = other.begin();
    UninitializedMove(otherBegin, otherBegin + otherSize, begin());
    SetSize(otherSize);

    other.clear();
}

template <class T, size_t N>
auto TCompactVector<T, N>::operator=(const TCompactVector& other) -> TCompactVector&
{
    assign(other);
    return *this;
}

template <class T, size_t N>
template <size_t OtherN>
auto TCompactVector<T, N>::operator=(const TCompactVector<T, OtherN>& other) -> TCompactVector&
{
    assign(other);
    return *this;
}

template <class T, size_t N>
auto TCompactVector<T, N>::operator=(TCompactVector&& other) -> TCompactVector&
{
    assign(std::move(other));
    return *this;
}

template <class T, size_t N>
template <size_t OtherN>
auto TCompactVector<T, N>::operator=(TCompactVector<T, OtherN>&& other) -> TCompactVector&
{
    assign(std::move(other));
    return *this;
}

template <class T, size_t N>
auto TCompactVector<T, N>::operator=(std::initializer_list<T> list) -> TCompactVector&
{
    assign(list);
    return *this;
}

template <class T, size_t N>
auto TCompactVector<T, N>::insert(const_iterator pos, const T& value) -> iterator
{
    return InsertOneImpl(
        pos,
        &value,
        [&] (auto* dst, const auto* value) {
            ::new(dst) T(*value);
        },
        [&] (auto* dst, const auto* value) {
            *dst = *value;
        });
}

template <class T, size_t N>
auto TCompactVector<T, N>::insert(const_iterator pos, T&& value) -> iterator
{
    return InsertOneImpl(
        pos,
        &value,
        [&] (auto* dst, auto* value) {
            ::new(dst) T(std::move(*value));
        },
        [&] (auto* dst, auto* value) {
            *dst = std::move(*value);
        });
}

template <class T, size_t N>
auto TCompactVector<T, N>::insert(const_iterator pos, size_type count, const T& value) -> iterator
{
    return InsertManyImpl(
        pos,
        count,
        [&] (auto* dstFirst, auto* dstLast) {
            for (auto* dst = dstFirst; dst != dstLast; ++dst) {
                ::new(dst) T(value);
            }
        },
        [&] (auto* dstFirst, auto* dstLast) {
            for (auto* dst = dstFirst; dst != dstLast; ++dst) {
                *dst = value;
            }
        });
}

template <class T, size_t N>
template <class TIterator>
auto TCompactVector<T, N>::insert(const_iterator pos, TIterator first, TIterator last) -> iterator
{
    auto current = first;
    return InsertManyImpl(
        pos,
        std::distance(first, last),
        [&] (auto* dstFirst, auto* dstLast) {
            for (auto* dst = dstFirst; dst != dstLast; ++dst) {
                ::new(dst) T(*current++);
            }
        },
        [&] (auto* dstFirst, auto* dstLast) {
            for (auto* dst = dstFirst; dst != dstLast; ++dst) {
                *dst = *current++;
            }
        });
}

template <class T, size_t N>
auto TCompactVector<T, N>::insert(const_iterator pos, std::initializer_list<T> list) -> iterator
{
    return insert(pos, list.begin(), list.end());
}

template <class T, size_t N>
void TCompactVector<T, N>::shrink_to_small()
{
    if (Y_LIKELY(IsInline())) {
        return;
    }

    auto size = this->size();
    if (size > N) {
        return;
    }

    auto* storage = OnHeapMeta_.Storage;
    UninitializedMove(storage->Elements, storage->End, &InlineElements_[0]);
    Destroy(storage->Elements, storage->End);
    ::free(storage);

    InlineMeta_.SizePlusOne = size + 1;
}

template <class T, size_t N>
bool TCompactVector<T, N>::IsInline() const
{
    return InlineMeta_.SizePlusOne != 0;
}

template <class T, size_t N>
void TCompactVector<T, N>::SetSize(size_t newSize)
{
    if (Y_LIKELY(IsInline())) {
        InlineMeta_.SizePlusOne = newSize + 1;
    } else {
        auto* storage = OnHeapMeta_.Storage;
        storage->End = storage->Elements + newSize;
    }
}

template <class T, size_t N>
void TCompactVector<T, N>::EnsureOnHeapCapacity(size_t newCapacity, bool incremental)
{
    newCapacity = std::max(newCapacity, N + 1);
    if (incremental) {
        newCapacity = std::max(newCapacity, capacity() * 2);
    }

    auto byteSize = sizeof(TOnHeapStorage) + newCapacity * sizeof(T);
    byteSize = nallocx(byteSize, 0);

    newCapacity = (byteSize - sizeof(TOnHeapStorage)) / sizeof(T);

    auto* newStorage = static_cast<TOnHeapStorage*>(::malloc(byteSize));
    YT_VERIFY((reinterpret_cast<uintptr_t>(newStorage) >> 56) == 0);

    newStorage->Capacity = newStorage->Elements + newCapacity;

    size_t size;
    if (IsInline()) {
        size = InlineMeta_.SizePlusOne - 1;
        UninitializedMove(&InlineElements_[0], &InlineElements_[0] + size, newStorage->Elements);
        Destroy(&InlineElements_[0], &InlineElements_[0] + size);
    } else {
        auto* storage = OnHeapMeta_.Storage;
        size = storage->End - storage->Elements;
        UninitializedMove(storage->Elements, storage->End, newStorage->Elements);
        Destroy(storage->Elements, storage->End);
        ::free(storage);
    }

    newStorage->End = newStorage->Elements + size;
    OnHeapMeta_.Storage = newStorage;
}

template <class T, size_t N>
template <class TPtr, class F>
auto TCompactVector<T, N>::PushBackImpl(TPtr valuePtr, F&& func) -> reference
{
    auto sizePlusOne = InlineMeta_.SizePlusOne;
    if (Y_LIKELY(sizePlusOne != 0 && sizePlusOne != N + 1)) {
        auto* dst = &InlineElements_[sizePlusOne - 1];
        func(dst, valuePtr);
        ++InlineMeta_.SizePlusOne;
        return *dst;
    }

    auto hasSpareOnHeapCapacity = [&] {
        if (sizePlusOne != 0) {
            return false;
        }
        auto* storage = OnHeapMeta_.Storage;
        return storage->End < storage->Capacity;
    };

    if (Y_UNLIKELY(!hasSpareOnHeapCapacity())) {
        TCompactVectorReallocationPtrAdjuster<TCompactVector, TPtr> valuePtrAdjuster(this, valuePtr);
        EnsureOnHeapCapacity(0, /*incremental*/ true);
    }

    YT_ASSERT(!IsInline());
    auto* storage = OnHeapMeta_.Storage;
    auto* dst = storage->End++;
    func(dst, valuePtr);

    return *dst;
}

template <class T, size_t N>
template <class F>
void TCompactVector<T, N>::ResizeImpl(size_type newSize, F&& func)
{
    auto size = this->size();
    if (newSize > size) {
        if (Y_UNLIKELY(newSize > capacity())) {
            EnsureOnHeapCapacity(newSize, /*incremental*/ false);
        }

        auto* first = end();
        auto* last = first + newSize - size;
        for (auto* current = first; current != last; ++current) {
            func(current);
        }
    } else if (newSize < size) {
        Destroy(begin() + newSize, end());
    }

    SetSize(newSize);
}

template <class T, size_t N>
template <class TPtr, class UninitializedF, class InitializedF>
auto TCompactVector<T, N>::InsertOneImpl(const_iterator pos, TPtr valuePtr, UninitializedF&& uninitializedFunc, InitializedF&& initializedFunc) -> iterator
{
    YT_ASSERT(pos >= begin());
    YT_ASSERT(pos <= end());

    auto* mutablePos = const_cast<iterator>(pos);

    auto newSize = size() + 1;
    if (Y_UNLIKELY(newSize > capacity())) {
        TCompactVectorReallocationPtrAdjuster<TCompactVector, iterator> mutablePosAdjuster(this, mutablePos);
        TCompactVectorReallocationPtrAdjuster<TCompactVector, TPtr> valuePtrAdjuster(this, valuePtr);
        EnsureOnHeapCapacity(newSize, /*incremental*/ true);
    }

    auto* end = this->end();

    if constexpr(!std::is_same_v<TPtr, std::nullptr_t>) {
        if (valuePtr >= mutablePos && valuePtr < end) {
            ++valuePtr;
        }
    }

    auto moveCount = std::distance(mutablePos, end);
    if (moveCount == 0) {
        uninitializedFunc(end, valuePtr);
    } else {
        if constexpr(std::is_trivially_copyable_v<T>) {
            ::memmove(mutablePos + 1, mutablePos, moveCount * sizeof(T));
        } else {
            ::new(end) T(std::move(end[-1]));
            MoveBackward(mutablePos, end - 1, mutablePos + 1);
        }
        initializedFunc(mutablePos, valuePtr);
    }

    SetSize(newSize);

    return mutablePos;
}

template <class T, size_t N>
template <class UninitializedF, class InitializedF>
auto TCompactVector<T, N>::InsertManyImpl(const_iterator pos, size_t insertCount, UninitializedF&& uninitializedFunc, InitializedF&& initializedFunc) -> iterator
{
    YT_ASSERT(pos >= begin());
    YT_ASSERT(pos <= end());

    auto* mutablePos = const_cast<iterator>(pos);
    if (insertCount == 0) {
        return mutablePos;
    }

    auto size = this->size();
    auto newSize = size + insertCount;
    if (Y_UNLIKELY(newSize > capacity())) {
        auto index = std::distance(begin(), mutablePos);
        EnsureOnHeapCapacity(newSize, /*incremental*/ true);
        mutablePos = begin() + index;
    }

    auto* end = this->end();
    auto moveCount = std::distance(mutablePos, end);
    if constexpr(std::is_trivially_copyable_v<T>) {
        ::memmove(mutablePos + insertCount, mutablePos, moveCount * sizeof(T));
        initializedFunc(mutablePos, mutablePos + insertCount);
    } else {
        if (static_cast<ptrdiff_t>(insertCount) >= moveCount) {
            UninitializedMove(mutablePos, end, mutablePos + insertCount);
            initializedFunc(mutablePos, end);
            uninitializedFunc(end, end + insertCount - moveCount);
        } else {
            auto overlapCount = moveCount - insertCount;
            UninitializedMove(mutablePos + overlapCount, end, mutablePos + overlapCount + insertCount);
            MoveBackward(mutablePos, mutablePos + overlapCount, mutablePos + insertCount);
            initializedFunc(mutablePos, mutablePos + insertCount);
        }
    }

    SetSize(newSize);

    return mutablePos;
}

template <class T, size_t N>
void TCompactVector<T, N>::Destroy(T* first, T* last)
{
    if constexpr(!std::is_trivially_destructible_v<T>) {
        for (auto* current = first; current != last; ++current) {
            current->T::~T();
        }
    }
}

template <class T, size_t N>
template <class T1, class T2>
void TCompactVector<T, N>::Copy(const T1* srcFirst, const T1* srcLast, T2* dst)
{
    if constexpr(std::is_trivially_copyable_v<T1> && std::is_same_v<T1, T2>) {
        ::memcpy(dst, srcFirst, (srcLast - srcFirst) * sizeof(T));
    } else {
        std::copy(srcFirst, srcLast, dst);
    }
}

template <class T, size_t N>
template <class T1, class T2>
void TCompactVector<T, N>::UninitializedCopy(const T1* srcFirst, const T1* srcLast, T2* dst)
{
    if constexpr(std::is_trivially_copyable_v<T1> && std::is_same_v<T1, T2>) {
        ::memcpy(dst, srcFirst, (srcLast - srcFirst) * sizeof(T));
    } else {
        std::uninitialized_copy(srcFirst, srcLast, dst);
    }
}

template <class T, size_t N>
void TCompactVector<T, N>::Move(T* srcFirst, T* srcLast, T* dst)
{
    if constexpr(std::is_trivially_copyable_v<T>) {
        ::memmove(dst, srcFirst, (srcLast - srcFirst) * sizeof(T));
    } else {
        std::move(srcFirst, srcLast, dst);
    }
}

template <class T, size_t N>
void TCompactVector<T, N>::UninitializedMove(T* srcFirst, T* srcLast, T* dst)
{
    if constexpr(std::is_trivially_copyable_v<T>) {
        ::memcpy(dst, srcFirst, (srcLast - srcFirst) * sizeof(T));
    } else {
        std::uninitialized_move(srcFirst, srcLast, dst);
    }
}

template <class T, size_t N>
void TCompactVector<T, N>::MoveBackward(T* srcFirst, T* srcLast, T* dst)
{
    auto* src = srcLast;
    dst += std::distance(srcFirst, srcLast);
    while (src > srcFirst) {
        *--dst = std::move(*--src);
    }
}

/////////////////////////////////////////////////////////////////////////////

template <class T, size_t LhsN, size_t RhsN>
bool operator==(const TCompactVector<T, LhsN>& lhs, const TCompactVector<T, RhsN>& rhs)
{
    if constexpr(LhsN == RhsN) {
        if (&lhs == &rhs) {
            return true;
        }
    }

    if (lhs.size() != rhs.size()) {
        return false;
    }

    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

template <class T, size_t LhsN, size_t RhsN>
bool operator!=(const TCompactVector<T, LhsN>& lhs, const TCompactVector<T, RhsN>& rhs)
{
    return !(lhs == rhs);
}

template <class T, size_t LhsN, size_t RhsN>
bool operator<(const TCompactVector<T, LhsN>& lhs, const TCompactVector<T, RhsN>& rhs)
{
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

////////////////////////////////////////////////////////////////////////////////

template <class T, size_t N>
void swap(TCompactVector<T, N>& lhs, TCompactVector<T, N>& rhs) // NOLINT
{
    lhs.swap(rhs);
}

/////////////////////////////////////////////////////////////////////////////

} // namespace NYT

namespace std {

////////////////////////////////////////////////////////////////////////////////

template <class T, size_t N>
struct hash<NYT::TCompactVector<T, N>>
{
    size_t operator()(const NYT::TCompactVector<T, N>& container) const
    {
        size_t result = 0;
        for (const auto& element : container) {
            NYT::HashCombine(result, element);
        }
        return result;
    }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace std
