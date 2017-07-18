#pragma once

#include "typetraits.h"

#include <algorithm>
#include <initializer_list>

template <class I, class T>
static inline bool IsIn(I f, I l, const T& v);

template <class C, class T>
static inline bool IsIn(const C& c, const T& e);

namespace NIsInHelper {
    Y_HAS_MEMBER(find, FindMethod);
    Y_HAS_SUBTYPE(const_iterator, ConstIterator);
    Y_HAS_SUBTYPE(key_type, KeyType);

    template <class T>
    using TIsAssocCont = std::conditional_t<THasFindMethod<T>::Result && THasConstIterator<T>::Result && THasKeyType<T>::Result, std::true_type, std::false_type>;

    template <class C, class T, bool isAssoc>
    struct TIsInTraits {
        static bool IsIn(const C& c, const T& e) {
            return ::IsIn(c.begin(), c.end(), e);
        }
    };

    template <class C, class T>
    struct TIsInTraits<C, T, true> {
        static bool IsIn(const C& c, const T& e) {
            return c.find(e) != c.end();
        }
    };
}

template <class I, class T>
static inline bool IsIn(I f, I l, const T& v) {
    return std::find(f, l, v) != l;
}

template <class C, class T>
static inline bool IsIn(const C& c, const T& e) {
    using namespace NIsInHelper;
    return TIsInTraits<C, T, TIsAssocCont<C>::value>::IsIn(c, e);
}

template <class T, class U>
static inline bool IsIn(std::initializer_list<T> l, const U& e) {
    return ::IsIn(l.begin(), l.end(), e);
}
