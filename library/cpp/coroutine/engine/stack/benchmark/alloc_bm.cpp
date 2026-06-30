#include <benchmark/benchmark.h>

#include <util/generic/vector.h>
#include <util/system/yassert.h>

#include <library/cpp/coroutine/engine/stack/stack_allocator.h>
#include <library/cpp/coroutine/engine/stack/stack_guards.h>
#include <library/cpp/coroutine/engine/stack/stack_pool.h>
#include <library/cpp/coroutine/engine/stack/stack_utils.h>


namespace NCoro::NStack::NBenchmark {

    const char* TestCoroName = "any_name";
    constexpr size_t BigCoroSize = PageSize * 25;
    constexpr size_t SmallCoroSize = PageSize * 4;
    constexpr size_t ManyStacks = 4096;

    void BasicOperations(TStackHolder& stack) {
        Y_ABORT_UNLESS(!stack.Get().empty());
        stack.LowerCanaryOk();
        stack.UpperCanaryOk();
    }

    void WriteStack(TStackHolder& stack) {
        auto memory = stack.Get();
        Y_ABORT_UNLESS(!memory.empty());
        stack.LowerCanaryOk();
        stack.UpperCanaryOk();
        for (size_t i = PageSize / 2; i < memory.size(); i += PageSize * 2) {
            memory[i] = 42;
        }
    }

    static void BM_GetAlignedMemory(benchmark::State& state) {
        char* raw = nullptr;
        char* aligned = nullptr;
        for (auto _ : state) {
            if (NCoro::NStack::GetAlignedMemory(state.range(0), raw, aligned)) {
                free(raw);
            }
        }
    }
    BENCHMARK(BM_GetAlignedMemory)->RangeMultiplier(16)->Range(1, 1024 * 1024);

    static void BM_GetAlignedMemoryReleaseRss(benchmark::State& state) {
        char* raw = nullptr;
        char* aligned = nullptr;
        for (auto _ : state) {
            if (NCoro::NStack::GetAlignedMemory(state.range(0), raw, aligned)) {
                const auto toFree = state.range(0) > 2 ? state.range(0) - 2 : 1;
                ReleaseRss(aligned, toFree);
                free(raw);
            }
        }
    }
    BENCHMARK(BM_GetAlignedMemoryReleaseRss)->RangeMultiplier(16)->Range(1, 1024 * 1024);

    static void BM_PoolAllocator(benchmark::State& state) {
        auto allocator = GetAllocator(TPoolAllocatorSettings{}, (EGuard)state.range(0));
        for (auto _ : state) {
            TStackHolder stack(*allocator, state.range(1), TestCoroName);
            BasicOperations(stack);
        }
    }
    BENCHMARK(BM_PoolAllocator)
        ->Args({(int64_t)EGuard::Canary, BigCoroSize}) // old version - ArgsProduct() is not supported
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize})
        ->Args({(int64_t)EGuard::Page, BigCoroSize})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize});

    static void BM_DefaultAllocator(benchmark::State& state) {
        auto allocator = GetAllocator(Nothing(), (EGuard)state.range(0));
        for (auto _ : state) {
            TStackHolder stack(*allocator, state.range(1), TestCoroName);
            BasicOperations(stack);
        }
    }
    BENCHMARK(BM_DefaultAllocator)
        ->Args({(int64_t)EGuard::Canary, BigCoroSize}) // old version - ArgsProduct() is not supported
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize})
        ->Args({(int64_t)EGuard::Page, BigCoroSize})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize});

    static void BM_PoolAllocatorManyStacksOneAtTime(benchmark::State& state) {
        TPoolAllocatorSettings settings;
        settings.StacksPerChunk = state.range(2);
        auto allocator = GetAllocator(settings, (EGuard)state.range(0));
        for (auto _ : state) {
            for (size_t i = 0; i < ManyStacks; ++i) {
                TStackHolder stack(*allocator, state.range(1), TestCoroName);
                BasicOperations(stack);
            }
        }
    }
    BENCHMARK(BM_PoolAllocatorManyStacksOneAtTime)
        ->Args({(int64_t)EGuard::Canary, BigCoroSize, 1}) // old version - ArgsProduct() is not supported
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize, 1})
        ->Args({(int64_t)EGuard::Page, BigCoroSize, 1})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize, 1})
        ->Args({(int64_t)EGuard::Canary, BigCoroSize, 1024})
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize, 1024})
        ->Args({(int64_t)EGuard::Page, BigCoroSize, 1024})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize, 1024});

    static void BM_DefaultAllocatorManyStacksOneAtTime(benchmark::State& state) {
        auto allocator = GetAllocator(Nothing(), (EGuard)state.range(0));
        for (auto _ : state) {
            for (size_t i = 0; i < ManyStacks; ++i) {
                TStackHolder stack(*allocator, state.range(1), TestCoroName);
                BasicOperations(stack);
            }
        }
    }
    BENCHMARK(BM_DefaultAllocatorManyStacksOneAtTime)
        ->Args({(int64_t)EGuard::Canary, BigCoroSize}) // old version - ArgsProduct() is not supported
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize})
        ->Args({(int64_t)EGuard::Page, BigCoroSize})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize});

    static void BM_PoolAllocatorManyStacks(benchmark::State& state) {
        TPoolAllocatorSettings settings;
        settings.StacksPerChunk = state.range(2);
        auto allocator = GetAllocator(settings, (EGuard)state.range(0));
        TVector<TStackHolder> stacks; // store stacks during benchmark
        stacks.reserve(ManyStacks);
        for (auto _ : state) {
            for (size_t i = 0; i < ManyStacks; ++i) {
                stacks.emplace_back(*allocator, state.range(1), TestCoroName);
                BasicOperations(stacks.back());
            }
        }
    }
    BENCHMARK(BM_PoolAllocatorManyStacks)
        ->Args({(int64_t)EGuard::Canary, BigCoroSize, 1}) // old version - ArgsProduct() is not supported
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize, 1})
        ->Args({(int64_t)EGuard::Page, BigCoroSize, 1})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize, 1})
        ->Args({(int64_t)EGuard::Canary, BigCoroSize, 1024})
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize, 1024})
        ->Args({(int64_t)EGuard::Page, BigCoroSize, 1024})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize, 1024});

    static void BM_DefaultAllocatorManyStacks(benchmark::State& state) {
        auto allocator = GetAllocator(Nothing(), (EGuard)state.range(0));
        TVector<TStackHolder> stacks; // store stacks during benchmark
        stacks.reserve(ManyStacks);
        for (auto _ : state) {
            for (size_t i = 0; i < ManyStacks; ++i) {
                stacks.push_back(TStackHolder(*allocator, state.range(1), TestCoroName));
                BasicOperations(stacks.back());
            }
        }
    }
    BENCHMARK(BM_DefaultAllocatorManyStacks)
        ->Args({(int64_t)EGuard::Canary, BigCoroSize}) // old version - ArgsProduct() is not supported
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize})
        ->Args({(int64_t)EGuard::Page, BigCoroSize})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize});

    // ------------------------------------------------------------------------
    static void BM_PoolAllocatorManyStacksReleased(benchmark::State& state) {
        TPoolAllocatorSettings settings;
        settings.StacksPerChunk = state.range(2);
        auto allocator = GetAllocator(settings, (EGuard)state.range(0));
        TVector<TStackHolder> stacks; // store stacks during benchmark
        stacks.reserve(ManyStacks);
        for (auto _ : state) {
            for (size_t i = 0; i < ManyStacks; ++i) {
                stacks.emplace_back(*allocator, state.range(1), TestCoroName);
                BasicOperations(stacks.back());
            }
            stacks.clear();
        }
    }
    BENCHMARK(BM_PoolAllocatorManyStacksReleased)
        ->Args({(int64_t)EGuard::Canary, BigCoroSize, 1}) // old version - ArgsProduct() is not supported
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize, 1})
        ->Args({(int64_t)EGuard::Page, BigCoroSize, 1})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize, 1})
        ->Args({(int64_t)EGuard::Canary, BigCoroSize, 1024})
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize, 1024})
        ->Args({(int64_t)EGuard::Page, BigCoroSize, 1024})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize, 1024});

    static void BM_DefaultAllocatorManyStacksReleased(benchmark::State& state) {
        auto allocator = GetAllocator(Nothing(), (EGuard)state.range(0));
        TVector<TStackHolder> stacks; // store stacks during benchmark
        stacks.reserve(ManyStacks);
        for (auto _ : state) {
            for (size_t i = 0; i < ManyStacks; ++i) {
                stacks.push_back(TStackHolder(*allocator, state.range(1), TestCoroName));
                BasicOperations(stacks.back());
            }
            stacks.clear();
        }
    }
    BENCHMARK(BM_DefaultAllocatorManyStacksReleased)
        ->Args({(int64_t)EGuard::Canary, BigCoroSize}) // old version - ArgsProduct() is not supported
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize})
        ->Args({(int64_t)EGuard::Page, BigCoroSize})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize});

    // ------------------------------------------------------------------------
    static void BM_PoolAllocatorManyStacksReleasedAndRealloc(benchmark::State& state) {
        TPoolAllocatorSettings settings;
        settings.StacksPerChunk = state.range(2);
        auto allocator = GetAllocator(settings, (EGuard)state.range(0));
        TVector<TStackHolder> stacks; // store stacks during benchmark
        stacks.reserve(ManyStacks);
        for (auto _ : state) {
            for (size_t i = 0; i < ManyStacks; ++i) {
                stacks.emplace_back(*allocator, state.range(1), TestCoroName);
                BasicOperations(stacks.back());
            }
            stacks.clear();
            for (size_t i = 0; i < ManyStacks; ++i) {
                stacks.emplace_back(*allocator, state.range(1), TestCoroName);
                BasicOperations(stacks.back());
            }
        }
    }
    BENCHMARK(BM_PoolAllocatorManyStacksReleasedAndRealloc)
        ->Args({(int64_t)EGuard::Canary, BigCoroSize, 1}) // old version - ArgsProduct() is not supported
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize, 1})
        ->Args({(int64_t)EGuard::Page, BigCoroSize, 1})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize, 1})
        ->Args({(int64_t)EGuard::Canary, BigCoroSize, 1024})
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize, 1024})
        ->Args({(int64_t)EGuard::Page, BigCoroSize, 1024})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize, 1024})
        ->Args({(int64_t)EGuard::Canary, BigCoroSize, 8192})
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize, 8192})
        ->Args({(int64_t)EGuard::Page, BigCoroSize, 8192})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize, 8192});

    static void BM_DefaultAllocatorManyStacksReleasedAndRealloc(benchmark::State& state) {
        auto allocator = GetAllocator(Nothing(), (EGuard)state.range(0));
        TVector<TStackHolder> stacks; // store stacks during benchmark
        stacks.reserve(ManyStacks);
        for (auto _ : state) {
            for (size_t i = 0; i < ManyStacks; ++i) {
                stacks.push_back(TStackHolder(*allocator, state.range(1), TestCoroName));
                BasicOperations(stacks.back());
            }
            stacks.clear();
            for (size_t i = 0; i < ManyStacks; ++i) {
                stacks.push_back(TStackHolder(*allocator, state.range(1), TestCoroName));
                BasicOperations(stacks.back());
            }
        }
    }
    BENCHMARK(BM_DefaultAllocatorManyStacksReleasedAndRealloc)
        ->Args({(int64_t)EGuard::Canary, BigCoroSize}) // old version - ArgsProduct() is not supported
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize})
        ->Args({(int64_t)EGuard::Page, BigCoroSize})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize});

    // ------------------------------------------------------------------------
    static void BM_PoolAllocatorManyStacksMemoryWriteReleasedAndRealloc(benchmark::State& state) {
        TPoolAllocatorSettings settings;
        settings.StacksPerChunk = state.range(2);
        auto allocator = GetAllocator(settings, (EGuard)state.range(0));
        TVector<TStackHolder> stacks; // store stacks during benchmark
        stacks.reserve(ManyStacks);
        for (auto _ : state) {
            for (size_t i = 0; i < ManyStacks; ++i) {
                stacks.emplace_back(*allocator, state.range(1), TestCoroName);
                WriteStack(stacks.back());
            }
            stacks.clear();
            for (size_t i = 0; i < ManyStacks; ++i) {
                stacks.emplace_back(*allocator, state.range(1), TestCoroName);
                WriteStack(stacks.back());
            }
        }
    }
    BENCHMARK(BM_PoolAllocatorManyStacksMemoryWriteReleasedAndRealloc)
        ->Args({(int64_t)EGuard::Canary, BigCoroSize, 1}) // old version - ArgsProduct() is not supported
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize, 1})
        ->Args({(int64_t)EGuard::Page, BigCoroSize, 1})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize, 1})
        ->Args({(int64_t)EGuard::Canary, BigCoroSize, 1024})
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize, 1024})
        ->Args({(int64_t)EGuard::Page, BigCoroSize, 1024})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize, 1024})
        ->Args({(int64_t)EGuard::Canary, BigCoroSize, 8192})
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize, 8192})
        ->Args({(int64_t)EGuard::Page, BigCoroSize, 8192})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize, 8192});

    static void BM_DefaultAllocatorManyStacksMemoryWriteReleasedAndRealloc(benchmark::State& state) {
        auto allocator = GetAllocator(Nothing(), (EGuard)state.range(0));
        TVector<TStackHolder> stacks; // store stacks during benchmark
        stacks.reserve(ManyStacks);
        for (auto _ : state) {
            for (size_t i = 0; i < ManyStacks; ++i) {
                stacks.push_back(TStackHolder(*allocator, state.range(1), TestCoroName));
                WriteStack(stacks.back());
            }
            stacks.clear();
            for (size_t i = 0; i < ManyStacks; ++i) {
                stacks.push_back(TStackHolder(*allocator, state.range(1), TestCoroName));
                WriteStack(stacks.back());
            }
        }
    }
    BENCHMARK(BM_DefaultAllocatorManyStacksMemoryWriteReleasedAndRealloc)
        ->Args({(int64_t)EGuard::Canary, BigCoroSize}) // old version - ArgsProduct() is not supported
        ->Args({(int64_t)EGuard::Canary, SmallCoroSize})
        ->Args({(int64_t)EGuard::Page, BigCoroSize})
        ->Args({(int64_t)EGuard::Page, SmallCoroSize});

}

BENCHMARK_MAIN();
