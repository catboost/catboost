#pragma once

#include "buffer.h"

#include <util/system/yassert.h>

#include <iterator>

/// References a C-styled array
template <typename T>
class TRegion {
public:
    using TValue = T;
    using TIterator = T*;
    using TConstIterator = const T*;

    using value_type = TValue;
    using reference = value_type&;
    using const_reference = const value_type&;
    using iterator = TIterator;
    using const_iterator = TConstIterator;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    inline TRegion()
        : DataPtr(nullptr)
        , Len(0)
    {
    }

    inline TRegion(T* data, size_t len)
        : DataPtr(data)
        , Len(len)
    {
    }

    inline TRegion(T* begin, T* end)
        : DataPtr(begin)
        , Len(end - begin)
    {
        Y_ASSERT(end >= begin);
    }

    inline TRegion(T& t)
        : DataPtr(&t)
        , Len(1)
    {
    }

    template <typename T2, size_t N>
    inline TRegion(T2 (&array)[N])
        : DataPtr(array)
        , Len(N)
    {
        static_assert(sizeof(T) == sizeof(T2), "expect sizeof(T) == sizeof(T2)");
    }

    template <class T2>
    inline TRegion(const TRegion<T2>& t)
        : DataPtr(t.Data())
        , Len(t.Size())
    {
        static_assert(sizeof(T) == sizeof(T2), "expect sizeof(T) == sizeof(T2)");
    }

    inline TRegion<T> SubRegion(size_t offset, size_t size) const {
        if (size == 0 || offset >= Len) {
            return TRegion<T>();
        }

        if (size > Len - offset) {
            size = Len - offset;
        }

        return TRegion<T>(DataPtr + offset, size);
    }

    inline reference front() noexcept {
        return *begin();
    }

    inline const_reference front() const noexcept {
        return *cbegin();
    }

    inline reference back() noexcept {
        return *rbegin();
    }

    inline const_reference back() const noexcept {
        return *crbegin();
    }

    inline const T* Data() const noexcept {
        return DataPtr;
    }

    inline T* Data() noexcept {
        return DataPtr;
    }

    inline const T* data() const noexcept {
        return Data();
    }

    inline T* data() noexcept {
        return Data();
    }

    inline size_t Size() const noexcept {
        return Len;
    }

    inline size_t size() const noexcept {
        return Size();
    }

    inline bool Empty() const noexcept {
        return !Len;
    }

    inline explicit operator bool() const noexcept {
        return !Empty();
    }

    inline bool empty() const noexcept {
        return Empty();
    }

    inline bool operator<(const TRegion& rhs) const {
        return (DataPtr < rhs.DataPtr) || (DataPtr == rhs.DataPtr && Len < rhs.Len);
    }

    inline bool operator==(const TRegion& rhs) const {
        return (DataPtr == rhs.DataPtr) && (Len == rhs.Len);
    }

    /*
     * some helpers...
     */
    inline T* operator~() noexcept {
        return Data();
    }

    inline const T* operator~() const noexcept {
        return Data();
    }

    inline size_t operator+() const noexcept {
        return Size();
    }

    /*
     * iterator-like interface
     */
    inline TConstIterator Begin() const noexcept {
        return DataPtr;
    }

    inline const_iterator begin() const noexcept {
        return Begin();
    }

    inline TIterator Begin() noexcept {
        return DataPtr;
    }

    inline iterator begin() noexcept {
        return Begin();
    }

    inline const_iterator cbegin() const noexcept {
        return Begin();
    }

    inline TConstIterator End() const noexcept {
        return DataPtr + Len;
    }

    inline const_iterator end() const noexcept {
        return End();
    }

    inline TIterator End() noexcept {
        return DataPtr + Len;
    }

    inline iterator end() noexcept {
        return End();
    }

    inline const_iterator cend() const noexcept {
        return End();
    }

    inline reverse_iterator rbegin() noexcept {
        return reverse_iterator{DataPtr + Len};
    }

    inline const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator{DataPtr + Len};
    }

    inline const_reverse_iterator crbegin() const noexcept {
        return const_reverse_iterator{DataPtr + Len};
    }

    inline reverse_iterator rend() noexcept {
        return reverse_iterator{DataPtr};
    }

    inline const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator{DataPtr};
    }

    inline const_reverse_iterator crend() const noexcept {
        return const_reverse_iterator{DataPtr};
    }

    inline const T& operator[](size_t idx) const noexcept {
        Y_ASSERT(idx < Len);
        return Data()[idx];
    }

    inline T& operator[](size_t idx) noexcept {
        Y_ASSERT(idx < Len);
        return Data()[idx];
    }

    inline yssize_t ysize() const noexcept {
        return static_cast<yssize_t>(this->size());
    }

private:
    T* DataPtr;
    size_t Len;
};

using TDataRegion = TRegion<const char>;
using TMemRegion = TRegion<char>;

// convert to region containers which follow yandex-style memory access
template <typename TCont>
TRegion<const typename TCont::value_type> ToRegion(const TCont& cont) {
    return TRegion<const typename TCont::value_type>(~cont, +cont);
}

template <typename TCont>
TRegion<typename TCont::value_type> ToRegion(TCont& cont) {
    return TRegion<typename TCont::value_type>(~cont, +cont);
}
