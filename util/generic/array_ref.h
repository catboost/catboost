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
 * when switching over from passing `TVector` to `TArrayRef`.
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

    constexpr inline TArrayRef(T* data Y_LIFETIME_BOUND, size_t len) noexcept
        : T_(data)
        , S_(len)
    {
    }

    constexpr inline TArrayRef(T* begin Y_LIFETIME_BOUND, T* end Y_LIFETIME_BOUND) noexcept
        : T_(begin)
        , S_(end - begin)
    {
    }

    constexpr inline TArrayRef(std::initializer_list<T> list Y_LIFETIME_BOUND) noexcept
        : T_(list.begin())
        , S_(list.size())
    {
    }

    template <class Container>
    constexpr inline TArrayRef(Container&& container, decltype(std::declval<T*&>() = container.data(), nullptr) = nullptr) noexcept
        : T_(container.data())
        , S_(container.size())
    {
        static_assert(
            sizeof(decltype(*container.data())) == sizeof(T),
            "Attempt to create TArrayRef from a container of elements with a different size");
    }

    template <size_t N>
    constexpr inline TArrayRef(T (&array)[N] Y_LIFETIME_BOUND) noexcept
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

    constexpr size_t size_bytes() const noexcept {
        return (size() * sizeof(T));
    }

    constexpr inline bool empty() const noexcept {
        return (S_ == 0);
    }

    constexpr inline iterator begin() const noexcept {
        return T_;
    }

    constexpr inline iterator end() const noexcept {
        return (T_ + S_);
    }

    constexpr inline const_iterator cbegin() const noexcept {
        return T_;
    }

    constexpr inline const_iterator cend() const noexcept {
        return (T_ + S_);
    }

    constexpr inline reverse_iterator rbegin() const noexcept {
        return reverse_iterator(T_ + S_);
    }

    constexpr inline reverse_iterator rend() const noexcept {
        return reverse_iterator(T_);
    }

    constexpr inline const_reverse_iterator crbegin() const noexcept {
        return const_reverse_iterator(T_ + S_);
    }

    constexpr inline const_reverse_iterator crend() const noexcept {
        return const_reverse_iterator(T_);
    }

    constexpr inline reference front() const noexcept {
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
            throw std::out_of_range("array ref range error");
        }

        return (*this)[n];
    }

    constexpr inline explicit operator bool() const noexcept {
        return (S_ > 0);
    }

    /**
     * Obtains a ref that is a view over the first `count` elements of this TArrayRef.
     *
     * The behavior is undefined if count > size().
     */
    TArrayRef first(size_t count) const {
        Y_ASSERT(count <= size());
        return TArrayRef(data(), count);
    }

    /**
     * Obtains a ref that is a view over the last `count` elements of this TArrayRef.
     *
     * The behavior is undefined if count > size().
     */
    TArrayRef last(size_t count) const {
        Y_ASSERT(count <= size());
        return TArrayRef(end() - count, end());
    }

    /**
     * Obtains a ref that is a view over the `count` elements of this TArrayRef starting at `offset`.
     *
     * The behavior is undefined in either offset or count is out of range.
     */
    TArrayRef subspan(size_t offset) const {
        Y_ASSERT(offset <= size());
        return TArrayRef(data() + offset, size() - offset);
    }

    TArrayRef subspan(size_t offset, size_t count) const {
        Y_ASSERT(offset + count <= size());
        return TArrayRef(data() + offset, count);
    }

    TArrayRef Slice(size_t offset) const {
        return subspan(offset);
    }

    TArrayRef Slice(size_t offset, size_t size) const {
        return subspan(offset, size);
    }

    /* FIXME:
     * This method is placed here for backward compatibility only and should be removed.
     * Keep in mind that it's behavior is different from Slice():
     *      SubRegion() never throws. It returns empty TArrayRef in case of invalid input.
     *
     * DEPRECATED. DO NOT USE.
     */
    TArrayRef SubRegion(size_t offset, size_t size) const {
        if (size == 0 || offset >= S_) {
            return TArrayRef();
        }

        if (size > S_ - offset) {
            size = S_ - offset;
        }

        return TArrayRef(T_ + offset, size);
    }

    constexpr inline yssize_t ysize() const noexcept {
        return static_cast<yssize_t>(this->size());
    }

private:
    T* T_;
    size_t S_;
};

/**
 * Obtains a view to the object representation of the elements of the TArrayRef arrayRef.
 *
 * Named as its std counterparts, std::as_bytes.
 */
template <typename T>
TArrayRef<const char> as_bytes(TArrayRef<T> arrayRef Y_LIFETIME_BOUND) noexcept {
    return TArrayRef<const char>(
        reinterpret_cast<const char*>(arrayRef.data()),
        arrayRef.size_bytes());
}

/**
 * Obtains a view to the writable object representation of the elements of the TArrayRef arrayRef.
 *
 * Named as its std counterparts, std::as_writable_bytes.
 */
template <typename T>
TArrayRef<char> as_writable_bytes(TArrayRef<T> arrayRef Y_LIFETIME_BOUND) noexcept {
    return TArrayRef<char>(
        reinterpret_cast<char*>(arrayRef.data()),
        arrayRef.size_bytes());
}

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
constexpr TArrayRef<T> MakeArrayRef(T* data Y_LIFETIME_BOUND, size_t size) {
    return TArrayRef<T>(data, size);
}

template <class T>
constexpr TArrayRef<T> MakeArrayRef(T* begin Y_LIFETIME_BOUND, T* end Y_LIFETIME_BOUND) {
    return TArrayRef<T>(begin, end);
}
