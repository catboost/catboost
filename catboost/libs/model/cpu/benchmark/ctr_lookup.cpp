#include <catboost/libs/helpers/dense_hash_view.h>
#include <catboost/libs/model/hash.h>

#include <benchmark/benchmark.h>

#include <util/digest/numeric.h>
#include <util/generic/vector.h>
#include <util/system/env.h>

namespace {
    template <size_t Distance>
    ui64 LookupBatch(const NCatboost::TDenseIndexHashView& table, const TVector<ui64>& hashes) {
        const auto buckets = table.GetBuckets();
        const size_t mask = buckets.size() - 1;
        ui64 checksum = 0;
        for (size_t index = 0; index < hashes.size(); ++index) {
            if constexpr (Distance > 0) {
                if (index + Distance < hashes.size()) {
                    Y_PREFETCH_READ(buckets.data() + (hashes[index + Distance] & mask), 3);
                }
            }
            checksum += table.GetIndex(hashes[index]);
        }
        return checksum;
    }
    void CtrLookupPrefetch(benchmark::State& state) {
        const size_t entryCount = state.range(0);
        if (entryCount > 32768 && GetEnv("CATBOOST_BENCH_LARGE").empty()) {
            state.SkipWithMessage("Set CATBOOST_BENCH_LARGE=1 for the beyond-L3 table");
            return;
        }
        TVector<NCatboost::TBucket> buckets(NCatboost::TDenseIndexHashBuilder::GetProperBucketsCount(entryCount));
        NCatboost::TDenseIndexHashBuilder builder(buckets);
        for (size_t index = 0; index < entryCount; ++index) {
            builder.AddIndex(IntHash(ui64(index)));
        }
        NCatboost::TDenseIndexHashView table(buckets);
        const size_t popularCount = state.range(2);
        const size_t queryCount = entryCount >= 1048576 && popularCount == 0 ? 1048576 : 16384;
        TVector<ui64> hashes(queryCount);
        ui64 expected = 0;
        for (size_t index = 0; index < hashes.size(); ++index) {
            if (index % 4 == 0) {
                hashes[index] = IntHash(ui64(entryCount + (popularCount ? index % popularCount : index)));
                expected += NCatboost::TDenseIndexHashView::NotFoundIndex;
            } else {
                const size_t item = (index * 73471) % (popularCount ? popularCount : entryCount);
                hashes[index] = IntHash(ui64(item));
                expected += item;
            }
        }
        using TLookup = ui64 (*)(const NCatboost::TDenseIndexHashView&, const TVector<ui64>&);
        TLookup lookup = nullptr;
        switch (state.range(1)) {
            case 0:
                lookup = LookupBatch<0>;
                break;
            case 4:
                lookup = LookupBatch<4>;
                break;
            case 8:
                lookup = LookupBatch<8>;
                break;
            case 16:
                lookup = LookupBatch<16>;
                break;
            case 32:
                lookup = LookupBatch<32>;
                break;
            case 64:
                lookup = LookupBatch<64>;
                break;
        }
        if (!lookup || lookup(table, hashes) != expected) {
            state.SkipWithError("CTR lookup checksum differs from analytical expected");
            return;
        }
        for (auto _ : state) {
            benchmark::DoNotOptimize(lookup(table, hashes));
        }
        state.SetItemsProcessed(state.iterations() * hashes.size());
    }
    BENCHMARK(CtrLookupPrefetch)->ArgsProduct({{128, 32768, 1048576}, {0, 4, 8, 16, 32, 64}, {0}})->Args({1048576, 0, 128})->Args({1048576, 4, 128})->Args({1048576, 8, 128})->Args({1048576, 16, 128})->Args({1048576, 32, 128})->Args({1048576, 64, 128});
} // namespace
