#include <library/cpp/testing/benchmark/bench.h>

#include <util/string/strip.h>
#include <util/generic/xrange.h>

static const TString SHORT_STRING = "  foo ";
static const TString LONG_STRING = TString(200, ' ') + TString(200, 'f') + TString(200, ' ');

Y_CPU_BENCHMARK(StripInPlaceShortNoChange, iface) {
    TString s = "foo";
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        StripInPlace(s);
        Y_DO_NOT_OPTIMIZE_AWAY(s);
    }
}

Y_CPU_BENCHMARK(StripInPlaceLongNoChange, iface) {
    TString s = TString{200, 'f'};
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        StripInPlace(s);
        Y_DO_NOT_OPTIMIZE_AWAY(s);
    }
}

Y_CPU_BENCHMARK(StripInPlaceShort, iface) {
    TString s;
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        s.assign(SHORT_STRING.begin(), SHORT_STRING.end());
        StripInPlace(s);
        Y_DO_NOT_OPTIMIZE_AWAY(s);
    }
}

Y_CPU_BENCHMARK(StripInPlaceLong, iface) {
    TString s;
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        s.assign(LONG_STRING.begin(), LONG_STRING.end());
        StripInPlace(s);
        Y_DO_NOT_OPTIMIZE_AWAY(s);
    }
}

Y_CPU_BENCHMARK(StripInPlaceShortMut, iface) {
    TString s = SHORT_STRING;
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        s.append(' ');
        StripInPlace(s);
        Y_DO_NOT_OPTIMIZE_AWAY(s);
    }
}

Y_CPU_BENCHMARK(StripInPlaceLongMut, iface) {
    TString s = LONG_STRING;
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        s.append(100, ' ');
        StripInPlace(s);
        Y_DO_NOT_OPTIMIZE_AWAY(s);
    }
}
