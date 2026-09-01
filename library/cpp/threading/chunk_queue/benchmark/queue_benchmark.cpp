#include <benchmark/benchmark.h>

#include <library/cpp/threading/chunk_queue/queue.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/thread/lfqueue.h>

#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <thread>

namespace {

    template <typename T>
    class TLockFreeQueueAdapter: public TLockFreeQueue<T> {
    public:
        bool Dequeue(T& v) {
            return TLockFreeQueue<T>::Dequeue(&v);
        }
    };

    int MakeInt(int64_t i) {
        return static_cast<int>(i);
    }

    // Payload for the non-trivial benchmarks: strings long enough to be
    // allocated in the heap (i.e. longer than the SSO buffer).
    constexpr size_t NumStrings = 64;
    constexpr size_t StringSize = 100;

    TString MakeString(int64_t i) {
        static const TVector<TString> strings = [] {
            TVector<TString> res(NumStrings);
            for (auto& s : res) {
                s = TString(StringSize, 'x');
            }
            return res;
        }();
        return strings[static_cast<size_t>(i) % strings.size()];
    }

    // ----- Runner for SPSC/MPSC/MPMC -----
    //
    // Worker threads are created once and synchronized with std::barrier:
    // spawning threads inside the measured loop would dominate the
    // measurement. SPSC and MPSC benchmarks reuse the same runner with
    // producers/consumers = 1.
    template <class Queue, class MakeValueFn>
    void RunMpmc(benchmark::State& state, Queue& q, MakeValueFn makeValue) {
        const int producers = static_cast<int>(state.range(0));
        const int consumers = static_cast<int>(state.range(1));
        const int64_t per = state.range(2);
        const int64_t total = static_cast<int64_t>(producers) * per;
        using T = std::decay_t<decltype(makeValue(0))>;

        std::atomic<int64_t> got{0};
        std::atomic<bool> stop{false};
        std::barrier sync{static_cast<std::ptrdiff_t>(producers + consumers + 1),
                          [&] { got.store(0, std::memory_order_relaxed); }};

        auto producer = [&] {
            for (;;) {
                sync.arrive_and_wait(); // round start
                if (stop.load(std::memory_order_relaxed)) {
                    return;
                }
                for (int64_t i : xrange(per)) {
                    q.Enqueue(makeValue(i));
                }
                sync.arrive_and_wait(); // round end
            }
        };

        auto consumer = [&] {
            for (;;) {
                sync.arrive_and_wait(); // round start
                if (stop.load(std::memory_order_relaxed)) {
                    return;
                }
                // Count in a local variable and touch the shared counter only
                // when the queue runs dry, to keep contention on `got` low.
                int64_t local = 0;
                for (;;) {
                    T item{};
                    if (q.Dequeue(item)) {
                        benchmark::DoNotOptimize(item);
                        ++local;
                        continue;
                    }
                    if (got.fetch_add(local, std::memory_order_acq_rel) + local >= total) {
                        break;
                    }
                    local = 0;
                    std::this_thread::yield();
                }
                sync.arrive_and_wait(); // round end
            }
        };

        TVector<std::thread> threads;
        threads.reserve(static_cast<size_t>(producers + consumers));
        for (int p = 0; p < producers; ++p) {
            threads.emplace_back(producer);
        }
        for (int c = 0; c < consumers; ++c) {
            threads.emplace_back(consumer);
        }

        for (auto _ : state) {
            sync.arrive_and_wait(); // round start
            sync.arrive_and_wait(); // round end
        }

        stop.store(true, std::memory_order_relaxed);
        sync.arrive_and_wait(); // release the workers so they can observe stop
        for (auto& t : threads) {
            t.join();
        }
        state.SetItemsProcessed(state.iterations() * total);
    }

    // ----- chunk_queue -----

    void BM_CQ_TOneOneQueue(benchmark::State& state) {
        NThreading::TOneOneQueue<int> q;
        RunMpmc(state, q, MakeInt);
    }

    void BM_CQ_TOneOneQueue_String(benchmark::State& state) {
        NThreading::TOneOneQueue<TString> q;
        RunMpmc(state, q, MakeString);
    }

    void BM_CQ_TManyOneQueue(benchmark::State& state) {
        NThreading::TManyOneQueue<int> q;
        RunMpmc(state, q, MakeInt);
    }

    void BM_CQ_TManyOneQueue_String(benchmark::State& state) {
        NThreading::TManyOneQueue<TString> q;
        RunMpmc(state, q, MakeString);
    }

    void BM_CQ_TRelaxedManyOneQueue(benchmark::State& state) {
        NThreading::TRelaxedManyOneQueue<int> q;
        RunMpmc(state, q, MakeInt);
    }

    void BM_CQ_TRelaxedManyOneQueue_String(benchmark::State& state) {
        NThreading::TRelaxedManyOneQueue<TString> q;
        RunMpmc(state, q, MakeString);
    }

    void BM_CQ_TManyManyQueue(benchmark::State& state) {
        NThreading::TManyManyQueue<int> q;
        RunMpmc(state, q, MakeInt);
    }

    void BM_CQ_TManyManyQueue_String(benchmark::State& state) {
        NThreading::TManyManyQueue<TString> q;
        RunMpmc(state, q, MakeString);
    }

    void BM_CQ_TRelaxedManyManyQueue(benchmark::State& state) {
        NThreading::TRelaxedManyManyQueue<int> q;
        RunMpmc(state, q, MakeInt);
    }

    void BM_CQ_TRelaxedManyManyQueue_String(benchmark::State& state) {
        NThreading::TRelaxedManyManyQueue<TString> q;
        RunMpmc(state, q, MakeString);
    }

    // ----- TLockFreeQueue -----

    void BM_LF_TLockFreeQueue(benchmark::State& state) {
        TLockFreeQueueAdapter<int> q;
        RunMpmc(state, q, MakeInt);
    }

    void BM_LF_TLockFreeQueue_String(benchmark::State& state) {
        TLockFreeQueueAdapter<TString> q;
        RunMpmc(state, q, MakeString);
    }

    // ----- int, 4096 elements -----

    BENCHMARK(BM_CQ_TOneOneQueue)->Args({1, 1, 4096})->UseRealTime();
    BENCHMARK(BM_CQ_TManyOneQueue)->Args({4, 1, 4096})->Args({16, 1, 4096})->UseRealTime();
    BENCHMARK(BM_CQ_TRelaxedManyOneQueue)->Args({4, 1, 4096})->Args({16, 1, 4096})->UseRealTime();
    BENCHMARK(BM_CQ_TManyManyQueue)->Args({4, 1, 4096})->Args({16, 1, 4096})->Args({4, 4, 4096})->Args({8, 8, 4096})->UseRealTime();
    BENCHMARK(BM_CQ_TRelaxedManyManyQueue)->Args({4, 1, 4096})->Args({16, 1, 4096})->Args({4, 4, 4096})->Args({8, 8, 4096})->UseRealTime();
    BENCHMARK(BM_LF_TLockFreeQueue)->Args({4, 1, 4096})->Args({16, 1, 4096})->Args({4, 4, 4096})->Args({8, 8, 4096})->UseRealTime();

    // ----- int, 16384 elements -----

    BENCHMARK(BM_CQ_TOneOneQueue)->Args({1, 1, 16384})->UseRealTime();
    BENCHMARK(BM_CQ_TManyOneQueue)->Args({4, 1, 16384})->Args({16, 1, 16384})->UseRealTime();
    BENCHMARK(BM_CQ_TRelaxedManyOneQueue)->Args({4, 1, 16384})->Args({16, 1, 16384})->UseRealTime();
    BENCHMARK(BM_CQ_TManyManyQueue)->Args({4, 1, 16384})->Args({16, 1, 16384})->Args({4, 4, 16384})->Args({8, 8, 16384})->UseRealTime();
    BENCHMARK(BM_CQ_TRelaxedManyManyQueue)->Args({4, 1, 16384})->Args({16, 1, 16384})->Args({4, 4, 16384})->Args({8, 8, 16384})->UseRealTime();
    BENCHMARK(BM_LF_TLockFreeQueue)->Args({4, 1, 16384})->Args({16, 1, 16384})->Args({4, 4, 16384})->Args({8, 8, 16384})->UseRealTime();

    // ----- TString, 4096 elements -----

    BENCHMARK(BM_CQ_TOneOneQueue_String)->Args({1, 1, 4096})->UseRealTime();
    BENCHMARK(BM_CQ_TManyOneQueue_String)->Args({4, 1, 4096})->Args({16, 1, 4096})->UseRealTime();
    BENCHMARK(BM_CQ_TRelaxedManyOneQueue_String)->Args({4, 1, 4096})->Args({16, 1, 4096})->UseRealTime();
    BENCHMARK(BM_CQ_TManyManyQueue_String)->Args({4, 1, 4096})->Args({16, 1, 4096})->Args({4, 4, 4096})->Args({8, 8, 4096})->UseRealTime();
    BENCHMARK(BM_CQ_TRelaxedManyManyQueue_String)->Args({4, 1, 4096})->Args({16, 1, 4096})->Args({4, 4, 4096})->Args({8, 8, 4096})->UseRealTime();
    BENCHMARK(BM_LF_TLockFreeQueue_String)->Args({4, 1, 4096})->Args({16, 1, 4096})->Args({4, 4, 4096})->Args({8, 8, 4096})->UseRealTime();

    // ----- TString, 16384 elements -----

    BENCHMARK(BM_CQ_TOneOneQueue_String)->Args({1, 1, 16384})->UseRealTime();
    BENCHMARK(BM_CQ_TManyOneQueue_String)->Args({4, 1, 16384})->Args({16, 1, 16384})->UseRealTime();
    BENCHMARK(BM_CQ_TRelaxedManyOneQueue_String)->Args({4, 1, 16384})->Args({16, 1, 16384})->UseRealTime();
    BENCHMARK(BM_CQ_TManyManyQueue_String)->Args({4, 1, 16384})->Args({16, 1, 16384})->Args({4, 4, 16384})->Args({8, 8, 16384})->UseRealTime();
    BENCHMARK(BM_CQ_TRelaxedManyManyQueue_String)->Args({4, 1, 16384})->Args({16, 1, 16384})->Args({4, 4, 16384})->Args({8, 8, 16384})->UseRealTime();
    BENCHMARK(BM_LF_TLockFreeQueue_String)->Args({4, 1, 16384})->Args({16, 1, 16384})->Args({4, 4, 16384})->Args({8, 8, 16384})->UseRealTime();

} // namespace
