#include <benchmark/benchmark.h>

#include <library/cpp/testing/hook/hook.h>
#include <util/generic/scope.h>

int main(int argc, char** argv) {
    NTesting::THook::CallBeforeInit();
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    NTesting::THook::CallBeforeRun();
    Y_DEFER { NTesting::THook::CallAfterRun(); };
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
