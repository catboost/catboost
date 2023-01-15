#include <library/testing/benchmark/bench.h>

#include <util/generic/xrange.h>
#include <util/string/ascii.h>
#include <util/generic/bitmap.h>
#include <util/generic/singleton.h>

namespace {
    struct TUpperMap: public TBitMap<256> {
        inline TUpperMap() noexcept {
            for (unsigned i = 'A'; i <= 'Z'; ++i) {
                Set((ui8)i);
            }
        }

        inline char ToLower(char x) const noexcept {
            return Get((ui8)x) ? x + ('a' - 'A') : x;
        }
    };

    struct TToLowerLookup {
        char Table[256];

        TToLowerLookup() {
            for (size_t i : xrange(256)) {
                Table[i] = AsciiToLower(i);
            }
        }

        char ToLower(char x) const noexcept {
            return Table[(ui8)x];
        }
    };
}

static inline char FastAsciiToLower(char c) {
    return (c >= 'A' && c <= 'Z') ? (c + ('a' - 'A')) : c;
}

static inline char FastAsciiToLower2(char c) {
    return c + ('a' - 'A') * (int)(c >= 'A' && c <= 'Z');
}

Y_CPU_BENCHMARK(AsciiToLower, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);

        for (int j = 0; j < 256; ++j) {
            Y_DO_NOT_OPTIMIZE_AWAY(AsciiToLower(j));
        }
    }
}

Y_CPU_BENCHMARK(AsciiToLowerChar, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);

        for (int j = 0; j < 256; ++j) {
            Y_DO_NOT_OPTIMIZE_AWAY(AsciiToLower((char)j));
        }
    }
}

Y_CPU_BENCHMARK(FastAsciiToLower, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);

        for (int j = 0; j < 256; ++j) {
            Y_DO_NOT_OPTIMIZE_AWAY(FastAsciiToLower(j));
        }
    }
}

Y_CPU_BENCHMARK(FastAsciiToLower2, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);

        for (int j = 0; j < 256; ++j) {
            Y_DO_NOT_OPTIMIZE_AWAY(FastAsciiToLower2(j));
        }
    }
}

Y_CPU_BENCHMARK(BitMapAsciiToLower, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);

        for (int j = 0; j < 256; ++j) {
            Y_DO_NOT_OPTIMIZE_AWAY(Singleton<TUpperMap>()->ToLower(j));
        }
    }
}

Y_CPU_BENCHMARK(LookupAsciiToLower, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);

        for (int j = 0; j < 256; ++j) {
            Y_DO_NOT_OPTIMIZE_AWAY(Singleton<TToLowerLookup>()->ToLower(j));
        }
    }
}

Y_CPU_BENCHMARK(LookupAsciiToLowerNoSingleton, iface) {
    TToLowerLookup lookup;
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);

        for (int j = 0; j < 256; ++j) {
            Y_DO_NOT_OPTIMIZE_AWAY(lookup.ToLower(j));
        }
    }
}

Y_CPU_BENCHMARK(tolower, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);

        for (int j = 0; j < 256; ++j) {
            Y_DO_NOT_OPTIMIZE_AWAY(tolower(j));
        }
    }
}
