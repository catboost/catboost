#include <catboost/libs/model/model.h>
#include <catboost/libs/model/cpu/simd.h>
#include <catboost/libs/model/cpu/quantization.h>
#include <benchmark/benchmark.h>
#include <util/generic/map.h>
#include <util/system/env.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>

namespace {
    void ConstantFloatBinarization(benchmark::State& state) {
        using namespace NCB::NModelEvaluation::NDetail;
        auto kernel = GetWideBinarize();
        if (!kernel) {
            state.SkipWithMessage("Wide binarization unavailable");
            return;
        }
        const size_t count = state.range(0), bordersCount = state.range(1), constantPercent = state.range(2);
        constexpr size_t FeatureCount = 64;
        TVector<float> values(FeatureCount * count), borders(bordersCount);
        for (size_t i = 0; i < bordersCount; ++i) {
            borders[i] = float(i) / 4.0f - 20.0f;
        }
        for (size_t f = 0; f < FeatureCount; ++f) {
            for (size_t d = 0; d < count; ++d) {
                const size_t source = constantPercent == 101 ? (d + 1 < count ? 0 : d) : ((f * 37 + 13) % 100 < constantPercent ? 0 : d);
                values[f * count + d] = float((source * 73 + f * 17) % 1021) / 16.0f - 31.0f;
            }
        }
        const size_t width = count * ((bordersCount + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN);
        TVector<ui8> output(FeatureCount * width), expected(output.size());
        for (size_t f = 0; f < FeatureCount; ++f) {
            for (size_t d = 0; d < count; ++d) {
                for (size_t b = 0; b < bordersCount; ++b) {
                    expected[f * width + (b / MAX_VALUES_PER_BIN) * count + d] += values[f * count + d] > borders[b];
                }
            }
            kernel(values.data() + f * count, count, borders.data(), bordersCount, false, 0.0f, output.data() + f * width);
        }
        if (output != expected) {
            state.SkipWithError("Binarization differs from scalar reference");
            return;
        }
        for (auto _ : state) {
            std::fill(output.begin(), output.end(), 0);
            for (size_t f = 0; f < FeatureCount; ++f) {
                kernel(values.data() + f * count, count, borders.data(), bordersCount, false, 0.0f, output.data() + f * width);
            }
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * FeatureCount * count);
    }
    BENCHMARK(ConstantFloatBinarization)->ArgsProduct({{20, 63, 64, 100, 128, 129}, {1, 8, 32, 254, 256}, {0, 50, 100, 101}});

    struct TRuntimeRows {
        TVector<float> Data;
        TVector<TConstArrayRef<float>> Refs;
        NCB::NModelEvaluation::TFeatureLayout Layout;
        TVector<double> Expected, Result;
        TRuntimeRows(TFullModel& model, size_t count, size_t constantPercent) {
            const size_t flat = model.ModelTrees->GetFlatFeatureVectorExpectedSize();
            const size_t stride = Max<size_t>(flat, 6389);
            TVector<ui32> permutation(stride);
            std::iota(permutation.begin(), permutation.end(), 0u);
            std::mt19937 rng(317);
            std::shuffle(permutation.begin(), permutation.end(), rng);
            permutation.resize(flat);
            Layout.FlatIndexes = permutation;
            Data.resize(count * stride, 0.0f);
            for (size_t d = 0; d < count; ++d) {
                for (const auto& feature : model.ModelTrees->GetFloatFeatures()) {
                    if (feature.Borders.empty()) {
                        continue;
                    }
                    const size_t f = feature.Position.FlatIndex;
                    const size_t source = constantPercent == 101 ? (d + 1 < count ? 0 : d) : ((f * 37 + 13) % 100 < constantPercent ? 0 : d);
                    const size_t h = (source * 73 + f * 37 + 19) % 65521;
                    const auto value = feature.Borders[h % feature.Borders.size()];
                    // Controlled model-border workload, not a production traffic dump.
                    Data[d * stride + permutation[f]] = std::nextafter(value,
                                                                       h % 2 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity());
                }
                Refs.emplace_back(Data.data() + d * stride, stride);
            }
            Expected.resize(count * model.GetDimensionsCount());
            Result.resize(Expected.size());
            for (size_t d = 0; d < count; ++d) {
                model.GetCurrentEvaluator()->CalcFlatSingle(Refs[d],
                                                            MakeArrayRef(Expected.data() + d * model.GetDimensionsCount(), model.GetDimensionsCount()), &Layout);
            }
        }
    };
    void EvictRuntimeModel() {
        static TVector<ui8> scratch(128 << 20, 3);
        ui64 sum = 0;
        for (size_t i = 0; i < scratch.size(); i += 64) {
            sum += scratch[i];
        }
        benchmark::DoNotOptimize(sum);
    }
    void RuntimeModel(benchmark::State& state) {
        if (GetEnv("CATBOOST_BENCH_MODEL").empty()) {
            state.SkipWithMessage("Set CATBOOST_BENCH_MODEL");
            return;
        }
        static TFullModel model = ReadModel(GetEnv("CATBOOST_BENCH_MODEL"));
        static TMap<std::pair<size_t, size_t>, std::unique_ptr<TRuntimeRows>> rows;
        const size_t count = state.range(0), fraction = state.range(1), cold = state.range(2);
        auto& ptr = rows[{count, fraction}];
        if (!ptr) {
            ptr = std::make_unique<TRuntimeRows>(model, count, fraction);
        }
        auto& batch = *ptr;
        auto evaluator = model.GetCurrentEvaluator();
        evaluator->CalcFlat(batch.Refs, batch.Result, &batch.Layout);
        if (std::memcmp(batch.Expected.data(), batch.Result.data(), batch.Result.size() * sizeof(double))) {
            state.SkipWithError("Full-model output differs from scalar reference");
            return;
        }
        for (auto _ : state) {
            if (cold) {
                state.PauseTiming();
                EvictRuntimeModel();
                state.ResumeTiming();
            }
            evaluator->CalcFlat(batch.Refs, batch.Result, &batch.Layout);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * count);
        state.counters["rows"] = count;
        state.counters["constant_percent"] = fraction;
        state.counters["trees"] = model.GetTreeCount();
    }
    BENCHMARK(RuntimeModel)->Apply([](benchmark::Benchmark* b) {
        for (int count : {8, 20, 30, 63, 64, 100, 128, 129, 500}) {
            for (int fraction : {0, 50, 100}) {
                b->Args({count, fraction, 0});
            }
        }
        for (int fraction : {0, 50, 100}) {
            b->Args({100, fraction, 1});
        }
    });
} // namespace
