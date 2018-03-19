#pragma once

#include <util/generic/typetraits.h>
#include <util/string/ascii.h>

#include <iterator>

template <class T>
struct TLowerCaseIterator: public std::iterator<std::input_iterator_tag, T> {
    using TNonConst = std::remove_const_t<T>;

    inline TLowerCaseIterator(T* c)
        : C(c)
    {
    }

    inline TLowerCaseIterator& operator++() noexcept {
        ++C;

        return *this;
    }

    inline TLowerCaseIterator operator++(int) noexcept {
        return C++;
    }

    inline TNonConst operator*() const noexcept {
        return AsciiToLower(*C);
    }

    T* C;
};

template <class T>
inline bool operator==(const TLowerCaseIterator<T>& l, const TLowerCaseIterator<T>& r) noexcept {
    return l.C == r.C;
}

template <class T>
inline bool operator!=(const TLowerCaseIterator<T>& l, const TLowerCaseIterator<T>& r) noexcept {
    return !(l == r);
}
