#pragma once

#include "typetraits.h"

#include <algorithm>
#include <initializer_list>

template <class I, class T>
constexpr bool IsIn(I f, I l, const T& v);

template <class C, class T>
constexpr bool IsIn(const C& c, const T& e);

namespace NIsInHelper {
    Y_HAS_MEMBER(find, FindMethod);
    Y_HAS_SUBTYPE(const_iterator, ConstIterator);
    Y_HAS_SUBTYPE(key_type, KeyType);

    template <class T>
    using TIsAssocCont = TConjunction<THasFindMethod<T>, THasConstIterator<T>, THasKeyType<T>>;

    template <class C, class T, bool isAssoc>
    struct TIsInTraits {
        static constexpr bool IsIn(const C& c, const T& e) {
            using std::begin;
            using std::end;
            return ::IsIn(begin(c), end(c), e);
        }
    };

    template <class C, class T>
    struct TIsInTraits<C, T, true> {
        static constexpr bool IsIn(const C& c, const T& e) {
            return c.find(e) != c.end();
        }
    };
}

template <class I, class T>
constexpr bool IsIn(I f, I l, const T& v) {
    return std::find(f, l, v) != l;
}

template <class C, class T>
constexpr bool IsIn(const C& c, const T& e) {
    using namespace NIsInHelper;
    return TIsInTraits<C, T, TIsAssocCont<C>::value>::IsIn(c, e);
}

template <class T, class U>
constexpr bool IsIn(std::initializer_list<T> l, const U& e) {
    return ::IsIn(l.begin(), l.end(), e);
}
