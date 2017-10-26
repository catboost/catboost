#include <library/testing/benchmark/bench.h>

#include <util/memory/pool.h>
#include <util/generic/xrange.h>
#include <util/stream/output.h>

#define BENCHMARK_POOL_ALLOC(chunkSize, allocSize, allocAlign)                               \
    Y_CPU_BENCHMARK(MemroyPool_chunk##chunkSize##_alloc##allocSize##_align##allocAlign, p) { \
        TMemoryPool pool(chunkSize);                                                         \
        for (auto i : xrange<size_t>(0, p.Iterations())) {                                   \
            (void)i;                                                                         \
            Y_DO_NOT_OPTIMIZE_AWAY(pool.Allocate(allocSize, allocAlign));                    \
        }                                                                                    \
        /*                                                                                   \
                Cerr << "Allocated: " << pool.MemoryAllocated() << Endl;                     \
                Cerr << "Waste:     " << pool.MemoryWaste() << Endl;                         \
        */                                                                                   \
    }

BENCHMARK_POOL_ALLOC(4096, 1, 1)
BENCHMARK_POOL_ALLOC(4096, 2, 2)
BENCHMARK_POOL_ALLOC(4096, 3, 4)
BENCHMARK_POOL_ALLOC(4096, 7, 8)
BENCHMARK_POOL_ALLOC(4096, 17, 16)
BENCHMARK_POOL_ALLOC(4096, 40, 64)
BENCHMARK_POOL_ALLOC(4096, 77, 128)
