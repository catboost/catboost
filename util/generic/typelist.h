#pragma once

#include <util/system/types.h>

#include <util/generic/typetraits.h>

#include <type_traits>

template <class... R>
struct TTypeList;

namespace NTL {
    template <unsigned N, typename TL>
    struct TGetImpl {
        using type = typename TGetImpl<N - 1, typename TL::TTail>::type;
    };

    template <typename TL>
    struct TGetImpl<0u, TL> {
        using type = typename TL::THead;
    };
}

template <>
struct TTypeList<> {
    static constexpr size_t Length = 0;

    template <class>
    using THave = std::false_type;

    template <template <class> class P>
    struct TSelectBy {
        using type = TTypeList<>;
    };
};

using TNone = TTypeList<>;

template <class H, class... R>
struct TTypeList<H, R...> {
    using THead = H;
    using TTail = TTypeList<R...>;

    static constexpr size_t Length = 1 + sizeof...(R);

    template <class V>
    using THave = TDisjunction<std::is_same<H, V>, typename TTail::template THave<V>>;

    template <unsigned N>
    using TGet = typename ::NTL::TGetImpl<N, TTypeList<H, R...>>::type;

    template <template <class> class P>
    struct TSelectBy {
        using type = std::conditional_t<P<THead>::value, THead, typename TTail::template TSelectBy<P>::type>;
    };
};

//FIXME: temporary to check overall build
template <class T>
struct TTypeList<T, TNone>: public TTypeList<T> {
};

using TCommonSignedInts = TTypeList<signed char, signed short, signed int, signed long, signed long long>;
using TCommonUnsignedInts = TTypeList<unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long, bool>;
using TFixedWidthSignedInts = TTypeList<i8, i16, i32, i64>;
using TFixedWidthUnsignedInts = TTypeList<ui8, ui16, ui32, ui64>;
using TFloats = TTypeList<float, double, long double>;

namespace NTL {
    template <class T1, class T2>
    struct TConcat;

    template <class... R1, class... R2>
    struct TConcat<TTypeList<R1...>, TTypeList<R2...>> {
        using type = TTypeList<R1..., R2...>;
    };

    template <bool isSigned, class T, class TS, class TU>
    struct TTypeSelectorBase {
        using TSignedInts = typename TConcat<TTypeList<T>, TS>::type;
        using TUnsignedInts = TU;
    };

    template <class T, class TS, class TU>
    struct TTypeSelectorBase<false, T, TS, TU> {
        using TSignedInts = TS;
        using TUnsignedInts = typename TConcat<TTypeList<T>, TU>::type;
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
    using TResult = TBoolConstant<sizeof(T) == sizeOf>;
};

template <typename T>
using TFixedWidthSignedInt = typename TFixedWidthSignedInts::template TSelectBy<TSizeOfPredicate<sizeof(T)>::template TResult>::type;

template <typename T>
using TFixedWidthUnsignedInt = typename TFixedWidthUnsignedInts::template TSelectBy<TSizeOfPredicate<sizeof(T)>::template TResult>::type;

template <typename T>
using TFixedWidthFloat = typename TFloats::template TSelectBy<TSizeOfPredicate<sizeof(T)>::template TResult>::type;
