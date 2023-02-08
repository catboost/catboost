#pragma once

#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/generic/maybe.h>
#include <util/generic/variant.h>
#include <util/stream/output.h>
#include <util/stream/str.h>
#include <util/datetime/base.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

/**
 * Automatically define GTest pretty printer for type that can print itself to util's `IOutputStream`.
 *
 * Note that this macro should be instantiated in the same namespace as the type you're printing, otherwise
 * ADL will not find it.
 *
 * Example:
 *
 * We define a struct `TMyContainer` and an output operator that works with `IOutputStream`. We then use this macro
 * to automatically define GTest pretty printer:
 *
 * ```
 * namespace NMy {
 *     struct TMyContainer {
 *         int x, y;
 *     };
 * }
 *
 * template <>
 * inline void Out<NMy::TMyContainer>(IOutputStream& stream, TTypeTraits<NMy::TMyContainer>::TFuncParam value) {
 *     stream << "{ x=" << value.x << ", y=" << value.y << " }";
 * }
 *
 * namespace NMy {
 *     Y_GTEST_ARCADIA_PRINTER(TMyContainer)
 * }
 * ```
 */
#define Y_GTEST_ARCADIA_PRINTER(T) \
    void PrintTo(const T& value, std::ostream* stream) {   \
        ::TString ss;                \
        ::TStringOutput s{ss};       \
        s << value;                  \
        *stream << ss;               \
    }


template <typename TCharType, typename TCharTraits>
void PrintTo(const TBasicString<TCharType, TCharTraits>& value, std::ostream* stream) {
    *stream << value.Quote().c_str();
}

template <typename TCharType, typename TCharTraits>
void PrintTo(TBasicStringBuf<TCharType, TCharTraits> value, std::ostream* stream) {
    *stream << TBasicString<TCharType, TCharTraits>{value}.Quote().c_str();
}

template <typename T, typename P>
void PrintTo(const TMaybe<T, P>& value, std::ostream* stream) {
    if (value.Defined()) {
        ::testing::internal::UniversalPrint(value.GetRef(), stream);
    } else {
        *stream << "nothing";
    }
}

inline void PrintTo(TNothing /* value */, std::ostream* stream) {
    *stream << "nothing";
}

inline void PrintTo(std::monostate /* value */, std::ostream* stream) {
    *stream << "monostate";
}

inline void PrintTo(TInstant value, std::ostream* stream) {
    *stream << value.ToString();
}

inline void PrintTo(TDuration value, std::ostream* stream) {
    *stream << value.ToString();
}
