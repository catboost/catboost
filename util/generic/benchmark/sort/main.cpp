#include <library/testing/benchmark/bench.h>

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>

Y_CPU_BENCHMARK(Sort1, iface) {
    TVector<int> x = {1};

    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Sort(x);
    }
}

Y_CPU_BENCHMARK(Sort2, iface) {
    TVector<int> x = {2, 1};

    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Sort(x);
    }
}

Y_CPU_BENCHMARK(Sort4, iface) {
    TVector<int> x = {4, 3, 2, 1};

    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Sort(x);
    }
}

Y_CPU_BENCHMARK(Sort16, iface) {
    TVector<int> x = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Sort(x);
    }
}

Y_CPU_BENCHMARK(StableSort1, iface) {
    TVector<int> x = {1};

    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        StableSort(x);
    }
}

Y_CPU_BENCHMARK(StableSort2, iface) {
    TVector<int> x = {2, 1};

    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        StableSort(x);
    }
}

Y_CPU_BENCHMARK(StableSort4, iface) {
    TVector<int> x = {4, 3, 2, 1};

    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        StableSort(x);
    }
}

Y_CPU_BENCHMARK(StableSort16, iface) {
    TVector<int> x = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        StableSort(x);
    }
}
