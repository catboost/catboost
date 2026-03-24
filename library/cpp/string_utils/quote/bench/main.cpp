#include <library/cpp/string_utils/quote/quote.cpp>
#include <library/cpp/testing/benchmark/bench.h>

#include <library/cpp/resource/resource.h>

#include <util/string/vector.h>

Y_CPU_BENCHMARK(OldEscapeSmall, iface) {
    const auto n = iface.Iterations();
    TString r = "1234";
    char buf[20];
    for (size_t i = 0; i < n; ++i) {
        CGIEscape(buf, r.c_str());
        Y_FAKE_READ(buf);
    }
}

Y_CPU_BENCHMARK(NewEscapeSmall, iface) {
    const auto n = iface.Iterations();
    TString r = "1234";
    char buf[20];
    for (size_t i = 0; i < n; ++i) {
        CGIEscape(buf, r.begin(), r.size());
        Y_FAKE_READ(buf);
    }
}

Y_CPU_BENCHMARK(OldEscapeMedium, iface) {
    const auto n = iface.Iterations();
    TString kekw = "!@#$%^&*(){}[]\" &param=!@#$%^&*(){}[]\" &param_param=!@#$%^&*(){}[]\" ";
    char buf[300];
    for (size_t i = 0; i < n; ++i) {
        CGIEscape(buf, kekw.c_str());
        Y_FAKE_READ(buf);
    }
}

Y_CPU_BENCHMARK(NewEscapeMedium, iface) {
    const auto n = iface.Iterations();
    TString kekw = "!@#$%^&*(){}[]\" &param=!@#$%^&*(){}[]\" &param_param=!@#$%^&*(){}[]\" ";
    char buf[300];
    for (size_t i = 0; i < n; ++i) {
        CGIEscape(buf, kekw.begin(), kekw.size());
        Y_FAKE_READ(buf);
    }
}

Y_CPU_BENCHMARK(OldEscapeBig, iface) {
    const auto n = iface.Iterations();

    TString kekw = NResource::Find("/test_files/long_cgi.txt");
    char buf[200'000];
    for (size_t i = 0; i < n; ++i) {
        CGIEscape(buf, kekw.c_str());
        Y_FAKE_READ(buf);
    }
}

Y_CPU_BENCHMARK(NewEscapeBig, iface) {
    const auto n = iface.Iterations();

    TString kekw = NResource::Find("/test_files/long_cgi.txt");
    char buf[200'000];
    for (size_t i = 0; i < n; ++i) {
        CGIEscape(buf, kekw.begin(), kekw.size());
        Y_FAKE_READ(buf);
    }
}

Y_CPU_BENCHMARK(OldEscapeArray, iface) {
    const auto n = iface.Iterations();

    TString kek = NResource::Find("/test_files/cgi_array.txt");
    TVector<TString> inputs = SplitString(kek, "\n");
    char buf[350'000];
    for (size_t i = 0; i < n; ++i) {
        TString& kekw = inputs[i % inputs.size()];

        CGIEscape(buf, kekw.c_str());
        Y_FAKE_READ(buf);
    }
}

Y_CPU_BENCHMARK(NewEscapeArray, iface) {
    const auto n = iface.Iterations();

    TString kek = NResource::Find("/test_files/cgi_array.txt");
    TVector<TString> inputs = SplitString(kek, "\n");
    char buf[350'000];
    for (size_t i = 0; i < n; ++i) {
        TString& kekw = inputs[i % inputs.size()];

        CGIEscape(buf, kekw.begin(), kekw.size());
        Y_FAKE_READ(buf);
    }
}

Y_CPU_BENCHMARK(OldEscapeHugeArray, iface) {
    const auto n = iface.Iterations();

    TString kek = NResource::Find("/test_files/cgi_huge_array.txt");
    TVector<TString> inputs = SplitString(kek, "\n");
    char buf[350'000];
    for (size_t i = 0; i < n; ++i) {
        TString& kekw = inputs[i % inputs.size()];

        CGIEscape(buf, kekw.c_str());
        Y_FAKE_READ(buf);
    }
}

Y_CPU_BENCHMARK(NewEscapeHugeArray, iface) {
    const auto n = iface.Iterations();

    TString kek = NResource::Find("/test_files/cgi_huge_array.txt");
    TVector<TString> inputs = SplitString(kek, "\n");
    char buf[350'000];
    for (size_t i = 0; i < n; ++i) {
        TString& kekw = inputs[i % inputs.size()];

        CGIEscape(buf, kekw.begin(), kekw.size());
        Y_FAKE_READ(buf);
    }
}
