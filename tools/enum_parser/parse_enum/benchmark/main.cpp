#include <tools/enum_parser/parse_enum/benchmark/enum.h_serialized.h>
#include <library/cpp/testing/benchmark/bench.h>
#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/stream/null.h>
#include <util/string/cast.h>

namespace {

    template <class TEnum>
    TVector<TEnum> SelectValues(size_t count) {
        auto values = GetEnumAllValues<TEnum>().Materialize();
        SortBy(values, [](const TEnum& v) { return IntHash(static_cast<ui64>(v)); });
        values.crop(count);
        return values;
    }

    template <class TEnum>
    TVector<TStringBuf> SelectStrings(size_t count) {
        TVector<TStringBuf> strings(Reserve(GetEnumItemsCount<TEnum>()));
        for (const auto& [_, s] : GetEnumNames<TEnum>()) {
            strings.push_back(s);
        }
        SortBy(strings, [](const TStringBuf& s) { return THash<TStringBuf>()(s); });
        strings.crop(count);
        return strings;
    }

    template <class TEnum, class TContext>
    void BMToString(TContext& iface) {
        const auto values = SelectValues<TEnum>(5u);
        for (const auto iter : xrange(iface.Iterations())) {
            Y_UNUSED(iter);
            for (const auto value : values) {
                Y_DO_NOT_OPTIMIZE_AWAY(ToString(value).size());
            }
        }
    }

    template <class TEnum, class TContext>
    void BMOut(TContext& iface) {
        const auto values = SelectValues<TEnum>(5u);
        TNullOutput null;
        for (const auto iter : xrange(iface.Iterations())) {
            Y_UNUSED(iter);
            for (const auto value : values) {
                Y_DO_NOT_OPTIMIZE_AWAY(null << value);
            }
        }
    }

    template <class TEnum, class TContext>
    void BMFromString(TContext& iface) {
        const auto strings = SelectStrings<TEnum>(5u);
        for (const auto iter : xrange(iface.Iterations())) {
            Y_UNUSED(iter);
            for (const auto s : strings) {
                Y_DO_NOT_OPTIMIZE_AWAY(FromString<TEnum>(s));
            }
        }
    }

    template <class TEnum, class TContext>
    void BMTryFromString(TContext& iface) {
        auto strings = SelectStrings<TEnum>(5u);
        strings.back() = "fake";

        TEnum value;
        for (const auto iter : xrange(iface.Iterations())) {
            Y_UNUSED(iter);
            for (const auto s : strings) {
                Y_DO_NOT_OPTIMIZE_AWAY(TryFromString<TEnum>(s, value));
            }
        }
    }
}

#define DEFINE_BENCHMARK(name)                     \
    Y_CPU_BENCHMARK(ToString_##name, iface) {      \
        BMToString<name>(iface);                   \
    }                                              \
    Y_CPU_BENCHMARK(Out_##name, iface) {           \
        BMOut<name>(iface);                        \
    }                                              \
    Y_CPU_BENCHMARK(FromString_##name, iface) {    \
        BMFromString<name>(iface);                 \
    }                                              \
    Y_CPU_BENCHMARK(TryFromString_##name, iface) { \
        BMTryFromString<name>(iface);              \
    }

DEFINE_BENCHMARK(ESmallSortedEnum);
DEFINE_BENCHMARK(ESmalUnsortedEnum);
DEFINE_BENCHMARK(EBigSortedEnum);
DEFINE_BENCHMARK(EBigUnsortedEnum);
DEFINE_BENCHMARK(EBigUnsortedDenseEnum);
