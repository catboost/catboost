#pragma once

// Define BENCHMARK_PREFIX and BENCHMARKED_CLASS before including this file.

#include <util/generic/xrange.h>

#define Y_CPU_PREFIXED_BENCHMARK_HELPER(prefix, name, iface) Y_CPU_BENCHMARK(prefix##name, iface)
#define Y_CPU_PREFIXED_BENCHMARK(prefix, name, iface) Y_CPU_PREFIXED_BENCHMARK_HELPER(prefix, name, iface)
#define CONCATENATE3_HELPER(a, b, c) a##b##c
#define CONCATENATE3(a, b, c) CONCATENATE3_HELPER(a, b, c)

namespace {
    namespace CONCATENATE3(N, BENCHMARK_PREFIX, Benchmark) {
        using TBenchmarkedClass = BENCHMARKED_CLASS;

        const auto defaultString = TBenchmarkedClass();
        const auto emptyString = TBenchmarkedClass("");
        const auto lengthOneString = TBenchmarkedClass("1");
        const auto length1KString = TBenchmarkedClass(1000, '1');

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CreateDefault, iface) {
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto result = TBenchmarkedClass();
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CreateFromEmptyLiteral, iface) {
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto result = TBenchmarkedClass("");
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CreateFromLengthOneLiteral, iface) {
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto result = TBenchmarkedClass("1");
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CreateLength1K, iface) {
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto result = TBenchmarkedClass(1000, '1');
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CopyDefaultString, iface) {
            const auto& sourceString = defaultString;
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto result = TBenchmarkedClass(sourceString);
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CopyEmptyString, iface) {
            const auto& sourceString = emptyString;
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto result = TBenchmarkedClass(sourceString);
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CopyLengthOneString, iface) {
            const auto& sourceString = lengthOneString;
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto result = TBenchmarkedClass(sourceString);
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CopyLength1KString, iface) {
            const auto& sourceString = length1KString;
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto result = TBenchmarkedClass(sourceString);
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CopyAndUpdateLengthOneString, iface) {
            const auto& sourceString = lengthOneString;
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto targetString = TBenchmarkedClass(sourceString);
                auto result = targetString[0] = '0';
                Y_DO_NOT_OPTIMIZE_AWAY(targetString);
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CopyAndAppendDefaultString, iface) {
            const auto& sourceString = defaultString;
            const TBenchmarkedClass::size_type insertPosition = sourceString.size();
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto targetString = TBenchmarkedClass(sourceString);
                auto result = targetString.insert(insertPosition, 1, '0');
                Y_DO_NOT_OPTIMIZE_AWAY(targetString);
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CopyAndAppendEmptyString, iface) {
            const auto& sourceString = emptyString;
            const TBenchmarkedClass::size_type insertPosition = sourceString.size();
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto targetString = TBenchmarkedClass(sourceString);
                auto result = targetString.insert(insertPosition, 1, '0');
                Y_DO_NOT_OPTIMIZE_AWAY(targetString);
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CopyAndAppendLengthOneString, iface) {
            const auto& sourceString = lengthOneString;
            const TBenchmarkedClass::size_type insertPosition = sourceString.size();
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto targetString = TBenchmarkedClass(sourceString);
                auto result = targetString.insert(insertPosition, 1, '0');
                Y_DO_NOT_OPTIMIZE_AWAY(targetString);
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CopyAndPrependLengthOneString, iface) {
            const auto& sourceString = lengthOneString;
            const TBenchmarkedClass::size_type insertPosition = 0;
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto targetString = TBenchmarkedClass(sourceString);
                auto result = targetString.insert(insertPosition, 1, '0');
                Y_DO_NOT_OPTIMIZE_AWAY(targetString);
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CopyAndUpdateLength1KString, iface) {
            const auto& sourceString = length1KString;
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto targetString = TBenchmarkedClass(sourceString);
                auto result = targetString[0] = '0';
                Y_DO_NOT_OPTIMIZE_AWAY(targetString);
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CopyAndAppendLength1KString, iface) {
            const auto& sourceString = length1KString;
            const TBenchmarkedClass::size_type insertPosition = sourceString.size();
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto targetString = TBenchmarkedClass(sourceString);
                auto result = targetString.insert(insertPosition, 1, '0');
                Y_DO_NOT_OPTIMIZE_AWAY(targetString);
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }

        Y_CPU_PREFIXED_BENCHMARK(BENCHMARK_PREFIX, CopyAndPrependLength1KString, iface) {
            const auto& sourceString = length1KString;
            const TBenchmarkedClass::size_type insertPosition = 0;
            for (const auto i : xrange(iface.Iterations())) {
                Y_UNUSED(i);
                auto targetString = TBenchmarkedClass(sourceString);
                auto result = targetString.insert(insertPosition, 1, '0');
                Y_DO_NOT_OPTIMIZE_AWAY(targetString);
                Y_DO_NOT_OPTIMIZE_AWAY(result);
            }
        }
    }
}

#undef CONCATENATE3
#undef CONCATENATE3_HELPER
#undef Y_CPU_PREFIXED_BENCHMARK
#undef Y_CPU_PREFIXED_BENCHMARK_HELPER
