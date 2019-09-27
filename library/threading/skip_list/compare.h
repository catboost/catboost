#pragma once

#include <util/generic/typetraits.h>
#include <util/str_stl.h>

namespace NThreading {
    namespace NImpl {
        Y_HAS_MEMBER(compare);
        Y_HAS_MEMBER(Compare);

        template <typename T>
        inline int CompareImpl(const T& l, const T& r) {
            if (l < r) {
                return -1;
            } else if (r < l) {
                return +1;
            } else {
                return 0;
            }
        }

        template <bool val>
        struct TSmallCompareSelector {
            template <typename T>
            static inline int Compare(const T& l, const T& r) {
                return CompareImpl(l, r);
            }
        };

        template <>
        struct TSmallCompareSelector<true> {
            template <typename T>
            static inline int Compare(const T& l, const T& r) {
                return l.compare(r);
            }
        };

        template <bool val>
        struct TBigCompareSelector {
            template <typename T>
            static inline int Compare(const T& l, const T& r) {
                return TSmallCompareSelector<THascompare<T>::value>::Compare(l, r);
            }
        };

        template <>
        struct TBigCompareSelector<true> {
            template <typename T>
            static inline int Compare(const T& l, const T& r) {
                return l.Compare(r);
            }
        };

        template <typename T>
        struct TCompareSelector: public TBigCompareSelector<THasCompare<T>::value> {
        };
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Generic compare function

    template <typename T>
    inline int Compare(const T& l, const T& r) {
        return NImpl::TCompareSelector<T>::Compare(l, r);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Generic compare functor

    template <typename T>
    struct TCompare {
        inline int operator()(const T& l, const T& r) const {
            return Compare(l, r);
        }
    };

}
