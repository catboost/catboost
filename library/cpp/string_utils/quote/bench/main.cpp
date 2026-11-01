#include <library/cpp/string_utils/quote/quote.cpp>
#include <library/cpp/testing/benchmark/bench.h>

#include <library/cpp/resource/resource.h>

#include <util/generic/array_ref.h>
#include <util/string/vector.h>

template <bool withExplicitLength>
static void EscapeBenchmark(size_t iterations, TConstArrayRef<TString> strings) {
    char buf[350'000];
    for (size_t i = 0; i < iterations; ++i) {
        const TString& input = strings[i % strings.size()];
        if constexpr (withExplicitLength) {
            CGIEscape(buf, input.data(), input.size());
        } else {
            CGIEscape(buf, input.c_str());
        }
        Y_FAKE_READ(buf);
    }
}

template <bool withExplicitLength>
static void UnescapeBenchmark(size_t iterations, TConstArrayRef<TString> strings) {
    char buf[350'000];
    for (size_t i = 0; i < iterations; ++i) {
        const TString& input = strings[i % strings.size()];
        if constexpr (withExplicitLength) {
            CGIUnescape(buf, input.data(), input.size());
        } else {
            CGIUnescape(buf, input.c_str());
        }
        Y_FAKE_READ(buf);
    }
}

Y_CPU_BENCHMARK(OldEscapeSmall, iface) {
    const TString inputs[] = {"1234"};
    EscapeBenchmark<false>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(NewEscapeSmall, iface) {
    const TString inputs[] = {"1234"};
    EscapeBenchmark<true>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(OldEscapeMedium, iface) {
    const TString inputs[] = {"!@#$%^&*(){}[]\" &param=!@#$%^&*(){}[]\" &param_param=!@#$%^&*(){}[]\" "};
    EscapeBenchmark<false>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(NewEscapeMedium, iface) {
    const TString inputs[] = {"!@#$%^&*(){}[]\" &param=!@#$%^&*(){}[]\" &param_param=!@#$%^&*(){}[]\" "};
    EscapeBenchmark<true>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(OldEscapeBig, iface) {
    const TString inputs[] = {NResource::Find("/test_files/long_cgi.txt")};
    EscapeBenchmark<false>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(NewEscapeBig, iface) {
    const TString inputs[] = {NResource::Find("/test_files/long_cgi.txt")};
    EscapeBenchmark<true>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(OldEscapeArray, iface) {
    const auto inputs = SplitString(NResource::Find("/test_files/cgi_array.txt"), "\n");
    EscapeBenchmark<false>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(NewEscapeArray, iface) {
    const auto inputs = SplitString(NResource::Find("/test_files/cgi_array.txt"), "\n");
    EscapeBenchmark<true>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(OldEscapeHugeArray, iface) {
    const auto inputs = SplitString(NResource::Find("/test_files/cgi_huge_array.txt"), "\n");
    EscapeBenchmark<false>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(NewEscapeHugeArray, iface) {
    const auto inputs = SplitString(NResource::Find("/test_files/cgi_huge_array.txt"), "\n");
    EscapeBenchmark<true>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(UnescapeCStringWithoutExplicitLengthSmall, iface) {
    const TString inputs[] = {"1234"};
    UnescapeBenchmark<false>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(UnescapeSmall, iface) {
    const TString inputs[] = {"1234"};
    UnescapeBenchmark<true>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(UnescapeCStringWithoutExplicitLengthMedium, iface) {
    const TString inputs[] = {CGIEscapeRet("!@#$%^&*(){}[]\" &param=!@#$%^&*(){}[]\" &param_param=!@#$%^&*(){}[]\" ")};
    UnescapeBenchmark<false>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(UnescapeMedium, iface) {
    const TString inputs[] = {CGIEscapeRet("!@#$%^&*(){}[]\" &param=!@#$%^&*(){}[]\" &param_param=!@#$%^&*(){}[]\" ")};
    UnescapeBenchmark<true>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(UnescapeCStringWithoutExplicitLengthBig, iface) {
    const TString inputs[] = {NResource::Find("/test_files/escaped_long_cgi.txt")};
    UnescapeBenchmark<false>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(UnescapeBig, iface) {
    const TString inputs[] = {NResource::Find("/test_files/escaped_long_cgi.txt")};
    UnescapeBenchmark<true>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(UnescapeCStringWithoutExplicitLengthArray, iface) {
    const auto inputs = SplitString(NResource::Find("/test_files/escaped_cgi_array.txt"), "\n");
    UnescapeBenchmark<false>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(UnescapeArray, iface) {
    const auto inputs = SplitString(NResource::Find("/test_files/escaped_cgi_array.txt"), "\n");
    UnescapeBenchmark<true>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(UnescapeCStringWithoutExplicitLengthHugeArray, iface) {
    const auto inputs = SplitString(NResource::Find("/test_files/escaped_cgi_huge_array.txt"), "\n");
    UnescapeBenchmark<false>(iface.Iterations(), inputs);
}

Y_CPU_BENCHMARK(UnescapeHugeArray, iface) {
    const auto inputs = SplitString(NResource::Find("/test_files/escaped_cgi_huge_array.txt"), "\n");
    UnescapeBenchmark<true>(iface.Iterations(), inputs);
}
