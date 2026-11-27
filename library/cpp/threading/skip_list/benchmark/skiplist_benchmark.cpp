#include <library/cpp/threading/skip_list/skiplist.h>

#include <benchmark/benchmark.h>

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/memory/pool.h>
#include <util/random/shuffle.h>
#include <util/string/builder.h>

#include <atomic>
#include <barrier>
#include <concepts>
#include <string>
#include <thread>

namespace NSkipListBenchmark {

    struct TPod32 {
        ui64 Key;
        ui64 Payload[3];

        int Compare(const TPod32& rhs) const {
            return (rhs.Key < Key) - (Key < rhs.Key);
        }
    };

    struct TPod32Wide {
        ui64 Word[4];

        int Compare(const TPod32Wide& rhs) const {
            for (int i = 0; i < 4; ++i) {
                if (Word[i] != rhs.Word[i]) {
                    return (rhs.Word[i] < Word[i]) - (Word[i] < rhs.Word[i]);
                }
            }
            return 0;
        }
    };

} // namespace NSkipListBenchmark

Y_DECLARE_PODTYPE(NSkipListBenchmark::TPod32);
Y_DECLARE_PODTYPE(NSkipListBenchmark::TPod32Wide);

namespace {

    using NSkipListBenchmark::TPod32;
    using NSkipListBenchmark::TPod32Wide;

    enum class EOrder {
        Random,
        Sequential,
        Reverse,
    };

    enum class EQuery {
        Hit,
        Miss,
        Mixed,
    };

    template <class TItem>
    using TList = NThreading::TSkipList<TItem>;

    const char* OrderName(EOrder order) {
        switch (order) {
            case EOrder::Random:
                return "random";
            case EOrder::Sequential:
                return "sequential";
            case EOrder::Reverse:
                return "reverse";
        }
    }

    TVector<ui64> MakeKeys(EOrder order, size_t n) {
        TVector<ui64> keys(n);
        Iota(keys.begin(), keys.end(), ui64{1});
        switch (order) {
            case EOrder::Sequential:
                break;
            case EOrder::Reverse:
                Reverse(keys.begin(), keys.end());
                break;
            case EOrder::Random:
                ShuffleRange(keys);
                break;
        }
        return keys;
    }

    template <class TItem>
    TItem MakeItem(ui64 key) {
        if constexpr (std::same_as<TItem, ui64>) {
            return key;
        } else if constexpr (std::same_as<TItem, TPod32>) {
            return {key, {key * 3, key * 5, key * 7}};
        } else {
            const ui64 bucket = key % 1024;
            return {{bucket, bucket / 2, bucket / 4, key}};
        }
    }

    template <class TItem>
    TVector<TItem> MakeItems(EOrder order, size_t n) {
        const TVector<ui64> keys = MakeKeys(order, n);
        TVector<TItem> items(Reserve(n));
        for (ui64 key : keys) {
            items.push_back(MakeItem<TItem>(key));
        }
        return items;
    }

    void TrashCache(TVector<char>& buf) {
        for (size_t i = 0; i < buf.size(); i += 64) {
            buf[i] = static_cast<char>(i);
        }
        benchmark::DoNotOptimize(buf.data());
        benchmark::ClobberMemory();
    }

    template <class TItem>
    void RunInsert(benchmark::State& state, EOrder order, size_t n) {
        const TVector<TItem> items = MakeItems<TItem>(order, n);
        for (auto _ : state) {
            TMemoryPool pool(1 << 20);
            TList<TItem> list(pool);
            for (const TItem& item : items) {
                benchmark::DoNotOptimize(list.Insert(item));
            }
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    }

    void RunInsertTrash(benchmark::State& state) {
        constexpr size_t BaseN = 1 << 16;
        constexpr size_t Batch = 256;
        const TVector<TPod32> items = MakeItems<TPod32>(EOrder::Random, BaseN + Batch);
        TMemoryPool pool(1 << 20);
        TList<TPod32> list(pool);
        TVector<char> trash(64 << 20);
        for (auto _ : state) {
            state.PauseTiming();
            list.Clear();
            for (size_t i = 0; i < BaseN; ++i) {
                list.Insert(items[i]);
            }
            TrashCache(trash);
            state.ResumeTiming();
            for (size_t i = 0; i < Batch; ++i) {
                benchmark::DoNotOptimize(list.Insert(items[BaseN + i]));
            }
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(Batch));
    }

    template <class TItem>
    void RunSeek(benchmark::State& state, size_t n, EQuery query) {
        const TVector<TItem> items = MakeItems<TItem>(EOrder::Random, n);
        TMemoryPool pool(1 << 20);
        TList<TItem> list(pool);
        for (const TItem& item : items) {
            list.Insert(item);
        }
        constexpr size_t Queries = 8192;
        TVector<TItem> queries(Reserve(Queries));
        for (size_t i = 0; i < Queries; ++i) {
            const bool hit = query == EQuery::Hit || (query == EQuery::Mixed && (i % 2) == 0);
            queries.push_back(hit ? items[i % n] : MakeItem<TItem>(static_cast<ui64>(n + 1 + i)));
        }
        for (auto _ : state) {
            for (const TItem& queryItem : queries) {
                benchmark::DoNotOptimize(list.Contains(queryItem));
            }
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(Queries));
    }

    void RunSeekTrash(benchmark::State& state) {
        constexpr size_t N = 1 << 16;
        constexpr size_t Batch = 256;
        const TVector<TPod32> items = MakeItems<TPod32>(EOrder::Random, N);
        TMemoryPool pool(1 << 20);
        TList<TPod32> list(pool);
        for (const TPod32& item : items) {
            list.Insert(item);
        }
        TVector<TPod32> queries(Reserve(Batch));
        for (size_t i = 0; i < Batch; ++i) {
            queries.push_back(items[(i * 97) % N]);
        }
        TVector<char> trash(64 << 20);
        for (auto _ : state) {
            state.PauseTiming();
            TrashCache(trash);
            state.ResumeTiming();
            for (const TPod32& query : queries) {
                benchmark::DoNotOptimize(list.Contains(query));
            }
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(Batch));
    }

    void RunE2E(benchmark::State& state, int consumers, size_t per) {
        constexpr size_t KeyCount = 1 << 20;
        const TVector<TPod32> items = MakeItems<TPod32>(EOrder::Random, KeyCount);
        TMemoryPool pool(1 << 20);
        TList<TPod32> list(pool);
        std::atomic<bool> stop{false};
        std::atomic<size_t> cursor{0};
        int phase = 0;
        std::barrier sync(static_cast<std::ptrdiff_t>(consumers + 2), [&] {
            phase ^= 1;
            if (phase == 0 && cursor.load(std::memory_order_relaxed) + per > items.size()) {
                list.Clear();
                cursor.store(0, std::memory_order_relaxed);
            }
        });
        auto producer = [&] {
            for (;;) {
                sync.arrive_and_wait();
                if (stop.load(std::memory_order_relaxed)) {
                    return;
                }
                const size_t begin = cursor.fetch_add(per, std::memory_order_relaxed);
                for (size_t i = 0; i < per; ++i) {
                    list.Insert(items[begin + i]);
                }
                sync.arrive_and_wait();
            }
        };
        auto consumer = [&] {
            size_t q = 0;
            for (;;) {
                sync.arrive_and_wait();
                if (stop.load(std::memory_order_relaxed)) {
                    return;
                }
                for (size_t i = 0; i < per; ++i) {
                    benchmark::DoNotOptimize(list.Contains(items[q]));
                    if (++q == items.size()) {
                        q = 0;
                    }
                }
                sync.arrive_and_wait();
            }
        };
        TVector<std::thread> threads(Reserve(static_cast<size_t>(consumers + 1)));
        threads.emplace_back(producer);
        for (int c = 0; c < consumers; ++c) {
            threads.emplace_back(consumer);
        }
        for (auto _ : state) {
            sync.arrive_and_wait();
            sync.arrive_and_wait();
        }
        stop.store(true, std::memory_order_relaxed);
        sync.arrive_and_wait();
        for (std::thread& thread : threads) {
            thread.join();
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(per) * (1 + consumers));
    }

    void Register() {
        constexpr size_t sizes[] = {4096, 65536, 1 << 20};
        for (EOrder order : {EOrder::Random, EOrder::Sequential, EOrder::Reverse}) {
            for (size_t n : sizes) {
                const TString name = TStringBuilder() << "Insert/Pod32/" << OrderName(order) << "/" << n;
                benchmark::RegisterBenchmark(std::string(name), [order, n](benchmark::State& state) {
                    RunInsert<TPod32>(state, order, n);
                });
            }
        }
        for (size_t n : sizes) {
            const TString hits = TStringBuilder() << "Seek/Pod32/hits/" << n;
            benchmark::RegisterBenchmark(std::string(hits), [n](benchmark::State& state) {
                RunSeek<TPod32>(state, n, EQuery::Hit);
            });
            const TString misses = TStringBuilder() << "Seek/Pod32/misses/" << n;
            benchmark::RegisterBenchmark(std::string(misses), [n](benchmark::State& state) {
                RunSeek<TPod32>(state, n, EQuery::Miss);
            });
            const TString small = TStringBuilder() << "Insert/ui64/random/" << n;
            benchmark::RegisterBenchmark(std::string(small), [n](benchmark::State& state) {
                RunInsert<ui64>(state, EOrder::Random, n);
            });
        }
        benchmark::RegisterBenchmark("Insert/Pod32/trash", [](benchmark::State& state) { RunInsertTrash(state); })->Iterations(8);
        benchmark::RegisterBenchmark("Seek/Pod32/mixed/1048576", [](benchmark::State& state) { RunSeek<TPod32>(state, 1 << 20, EQuery::Mixed); });
        benchmark::RegisterBenchmark("Insert/Pod32Wide/random/1048576", [](benchmark::State& state) { RunInsert<TPod32Wide>(state, EOrder::Random, 1 << 20); });
        benchmark::RegisterBenchmark("Seek/Pod32Wide/hits/1048576", [](benchmark::State& state) { RunSeek<TPod32Wide>(state, 1 << 20, EQuery::Hit); });
        benchmark::RegisterBenchmark("Seek/Pod32/trash", [](benchmark::State& state) { RunSeekTrash(state); })->Iterations(8);
        for (int consumers : {0, 1, 4}) {
            const TString name = TStringBuilder() << "E2E/c" << consumers;
            benchmark::RegisterBenchmark(std::string(name), [consumers](benchmark::State& state) {
                RunE2E(state, consumers, 4096);
            })->UseRealTime();
        }
    }

    struct TRegistrar {
        TRegistrar() {
            Register();
        }
    };

    TRegistrar Registrar;

} // namespace
