#pragma once

#include <util/system/types.h>

#include <type_traits>

template <class... R>
struct TTypeList;

namespace NTL {
    template <unsigned N, typename TL>
    struct TGetImpl {
        using TResult = typename TGetImpl<N - 1, typename TL::TTail>::TResult;
    };

    template <typename TL>
    struct TGetImpl<0u, TL> {
        using TResult = typename TL::THead;
    };
}

template <>
struct TTypeList<> {
    enum {
        Length = 0
    };

    template <class V>
    struct THave {
        enum {
            Result = false
        };
    };

    template <template <class> class P>
    struct TSelectBy {
        using TResult = TTypeList<>;
    };
};

using TNone = TTypeList<>;

template <class H, class... R>
struct TTypeList<H, R...> {
    using THead = H;
    using TTail = TTypeList<R...>;

    enum {
        Length = 1 + sizeof...(R)
    };

    template <class V>
    struct THave {
        enum {
            Result = std::is_same<H, V>::value || TTail::template THave<V>::Result
        };
    };

    template <unsigned N>
    using TGet = typename ::NTL::TGetImpl<N, TTypeList<H, R...>>::TResult;

    template <template <class> class P>
    struct TSelectBy {
        using TResult = std::conditional_t<P<THead>::Result, THead, typename TTail::template TSelectBy<P>::TResult>;
    };
};

//FIXME: temporary to check overall build
template <class T>
struct TTypeList<T, TNone>: public TTypeList<T> {
};

template <class... R>
using TTypeListBuilder = TTypeList<R...>;

using TCommonSignedInts = TTypeListBuilder<signed char, signed short, signed int, signed long, signed long long>;
using TCommonUnsignedInts = TTypeListBuilder<unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long, bool>;
using TFixedWidthSignedInts = TTypeListBuilder<i8, i16, i32, i64>;
using TFixedWidthUnsignedInts = TTypeListBuilder<ui8, ui16, ui32, ui64>;
using TFloats = TTypeListBuilder<float, double, long double>;

namespace NTL {
    template <class T1, class T2>
    struct TConcat;

    template <class... R1, class... R2>
    struct TConcat<TTypeList<R1...>, TTypeList<R2...>> {
        using TResult = TTypeList<R1..., R2...>;
    };

    template <bool isSigned, class T, class TS, class TU>
    struct TTypeSelectorBase {
        using TSignedInts = typename TConcat<TTypeList<T>, TS>::TResult;
        using TUnsignedInts = TU;
    };

    template <class T, class TS, class TU>
    struct TTypeSelectorBase<false, T, TS, TU> {
        using TSignedInts = TS;
        using TUnsignedInts = typename TConcat<TTypeList<T>, TU>::TResult;
    };

    template <class T, class TS, class TU>
    struct TTypeSelector: public TTypeSelectorBase<((T)(-1) < 0), T, TS, TU> {
    };

    using T1 = TTypeSelector<char, TCommonSignedInts, TCommonUnsignedInts>;
    using T2 = TTypeSelector<wchar_t, T1::TSignedInts, T1::TUnsignedInts>;
}

using TSignedInts = NTL::T2::TSignedInts;
using TUnsignedInts = NTL::T2::TUnsignedInts;

template <unsigned sizeOf>
struct TSizeOfPredicate {
    template <class T>
    struct TResult {
        enum {
            Result = (sizeof(T) == sizeOf)
        };
    };
};
