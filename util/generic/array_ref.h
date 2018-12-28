#pragma once

#include "vector_ops.h"

#include <util/generic/fwd.h>
#include <util/generic/utility.h>

#include <algorithm>
#include <initializer_list>

/**
 * `TArrayRef` works pretty much like `std::span` with dynamic extent, presenting
 * an array-like interface into a contiguous sequence of objects.
 *
 * It can be used at interface boundaries instead of `TVector` or
 * pointer-size pairs, and is actually a preferred way to pass contiguous data
 * into functions.
 *
 * Note that `TArrayRef` can be auto-constructed from any contiguous container
 * (with `size` and `data` members), and thus you don't have to change client code
 * when swithcing over from passing `TVector` to `TArrayRef`.
 *
 * Note that `TArrayRef` has the same const-semantics as raw pointers:
 * - `TArrayRef<T>` is a non-const reference to non-const data (like `T*`);
 * - `TArrayRef<const T>` is a non-const reference to const data (like `const T*`);
 * - `const TArrayRef<T>` is a const reference to non-const data (like `T* const`);
 * - `const TArrayRef<const T>` is a const reference to const data (like `const T* const`).
 */
template <class T>
class TArrayRef: public NVectorOps::TVectorOps<T, TArrayRef<T>> {
public:
    constexpr inline TArrayRef() noexcept
        : T_(nullptr)
        , S_(0)
    {
    }

    constexpr inline TArrayRef(T* data, size_t len) noexcept
        : T_(data)
        , S_(len)
    {
    }

    constexpr inline TArrayRef(T* begin, T* end) noexcept
        : T_(begin)
        , S_(end - begin)
    {
    }

    constexpr inline TArrayRef(std::initializer_list<T> list) noexcept
        : T_(list.begin())
        , S_(list.size())
    {
    }

    template <class Container>
    constexpr inline TArrayRef(Container&& container, decltype(std::declval<T*&>() = container.data(), nullptr) = nullptr) noexcept
        : T_(container.data())
        , S_(container.size())
    {
    }

    template <size_t N>
    constexpr inline TArrayRef(T (&array)[N]) noexcept
        : T_(array)
        , S_(N)
    {
    }

    template <class TT, typename = std::enable_if_t<std::is_same<std::remove_const_t<T>, std::remove_const_t<TT>>::value>>
    bool operator==(const TArrayRef<TT>& other) const noexcept {
        return Size() == other.Size() && std::equal(this->Begin(), this->End(), other.Begin());
    }

    inline ~TArrayRef() = default;

    // TODO: drop
    constexpr inline T* Data() const noexcept {
        return T_;
    }

    // TODO: drop
    constexpr inline size_t Size() const noexcept {
        return S_;
    }

    // TODO: drop
    inline void Swap(TArrayRef& a) noexcept {
        ::DoSwap(T_, a.T_);
        ::DoSwap(S_, a.S_);
    }

    /* STL compatibility. */

    constexpr inline T* data() const noexcept {
        return Data();
    }

    constexpr inline size_t size() const noexcept {
        return Size();
    }

    inline void swap(TArrayRef& a) noexcept {
        Swap(a);
    }

    TArrayRef<T> Slice(size_t offset) const {
        Y_ASSERT(offset <= size());
        return TArrayRef<T>(data() + offset, size() - offset);
    }

    TArrayRef<T> Slice(size_t offset, size_t size) const {
        Y_ASSERT(offset + size <= this->size());

        return TArrayRef<T>(data() + offset, data() + offset + size);
    }

private:
    T* T_;
    size_t S_;
};

template <class Range>
constexpr TArrayRef<const typename Range::value_type> MakeArrayRef(const Range& range) {
    return TArrayRef<const typename Range::value_type>(range);
}

template <class Range>
constexpr TArrayRef<typename Range::value_type> MakeArrayRef(Range& range) {
    return TArrayRef<typename Range::value_type>(range);
}

template <class Range>
constexpr TArrayRef<const typename Range::value_type> MakeConstArrayRef(const Range& range) {
    return TArrayRef<const typename Range::value_type>(range);
}

template <class Range>
constexpr TArrayRef<const typename Range::value_type> MakeConstArrayRef(Range& range) {
    return TArrayRef<const typename Range::value_type>(range);
}

template <class T>
constexpr TArrayRef<T> MakeArrayRef(T* data, size_t size) {
    return TArrayRef<T>(data, size);
}

template <class T>
constexpr TArrayRef<T> MakeArrayRef(T* begin, T* end) {
    return TArrayRef<T>(begin, end);
}
