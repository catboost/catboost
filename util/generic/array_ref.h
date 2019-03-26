#pragma once

#include <util/generic/yexception.h>

#include <algorithm>
#include <initializer_list>
#include <iterator>

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
class TArrayRef {
public:
    using iterator = T*;
    using const_iterator = const T*;
    using reference = T&;
    using const_reference = const T&;
    using value_type = T;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

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
        return (S_ == other.size()) && std::equal(begin(), end(), other.begin());
    }

    constexpr inline T* data() const noexcept {
        return T_;
    }

    constexpr inline size_t size() const noexcept {
        return S_;
    }

    inline bool empty() const noexcept {
        return (S_ == 0);
    }

    inline iterator begin() const noexcept {
        return T_;
    }

    inline iterator end() const noexcept {
        return (T_ + S_);
    }

    inline const_iterator cbegin() const noexcept {
        return T_;
    }

    inline const_iterator cend() const noexcept {
        return (T_ + S_);
    }

    inline reverse_iterator rbegin() const noexcept {
        return reverse_iterator(T_ + S_);
    }

    inline reverse_iterator rend() const noexcept {
        return reverse_iterator(T_);
    }

    inline const_reverse_iterator crbegin() const noexcept {
        return const_reverse_iterator(T_ + S_);
    }

    inline const_reverse_iterator crend() const noexcept {
        return const_reverse_terator(T_);
    }

    inline reference front() const noexcept {
        return *T_;
    }

    inline reference back() const noexcept {
        Y_ASSERT(S_ > 0);

        return *(end() - 1);
    }

    inline reference operator[](size_t n) const noexcept {
        Y_ASSERT(n < S_);

        return *(T_ + n);
    }

    inline reference at(size_t n) const {
        if (n >= S_) {
            ThrowRangeError("array ref range error");
        }

        return (*this)[n];
    }

    inline explicit operator bool() const noexcept {
        return (S_ > 0);
    }

    TArrayRef<T> Slice(size_t offset) const {
        Y_ASSERT(offset <= size());
        return TArrayRef<T>(data() + offset, size() - offset);
    }

    TArrayRef<T> Slice(size_t offset, size_t size) const {
        Y_ASSERT(offset + size <= S_);

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
