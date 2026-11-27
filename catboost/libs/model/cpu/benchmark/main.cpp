#include <catboost/libs/model/cpu/simd.h>
#include <catboost/libs/model/model.h>

#include <benchmark/benchmark.h>

#include <util/generic/vector.h>

#include <array>

using namespace NCB::NModelEvaluation::NDetail;

namespace {
    void MeasureLeafValues(benchmark::State& state, TLeafValues4Function kernel) {
        if (!kernel) {
            state.SkipWithMessage("SIMD kernel is unavailable on this compiler/CPU/OS");
            return;
        }
        const size_t docCount = state.range(0);
        std::array<TVector<double>, 4> leaves;
        std::array<TVector<ui8>, 4> indexStorage;
        const double* leafPointers[4];
        const ui8* indexPointers[4];
        for (size_t tree = 0; tree < 4; ++tree) {
            leaves[tree].resize(256);
            indexStorage[tree].resize(docCount);
            for (size_t leaf = 0; leaf < 256; ++leaf) {
                leaves[tree][leaf] = (double(leaf) - 127.0) / 8.0;
            }
            for (size_t doc = 0; doc < docCount; ++doc) {
                indexStorage[tree][doc] = (doc * 73 + tree * 29) % 256;
            }
            leafPointers[tree] = leaves[tree].data();
            indexPointers[tree] = indexStorage[tree].data();
        }
        TVector<double> results(docCount, 0.0);
        for (auto _ : state) {
            kernel(docCount, leafPointers, indexPointers, results.data());
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * docCount * 4);
    }

    void LeafValuesAvx2(benchmark::State& state) {
        MeasureLeafValues(state, GetLeafValues4Avx2(state.range(1)));
    }
    BENCHMARK(LeafValuesAvx2)->ArgsProduct({{16, 32, 64, 128}, {0, 8, 16, 32, 64}});

    void LeafValuesAvx512(benchmark::State& state) {
        MeasureLeafValues(state, GetLeafValues4Avx512(state.range(1)));
    }
    BENCHMARK(LeafValuesAvx512)->ArgsProduct({{16, 32, 64, 128}, {0, 8, 16, 32, 64}});

    void EvaluateModel(benchmark::State& state) {
        const size_t docCount = state.range(0);
        const size_t depth = state.range(1);
        TFullModel model;
        auto* trees = model.ModelTrees.GetMutable();
        TVector<int> splits;
        for (size_t feature = 0; feature < depth; ++feature) {
            trees->AddFloatFeature(TFloatFeature(false, feature, feature, {0.5f}, ""));
            splits.push_back(feature);
        }
        constexpr size_t TreeCount = 512;
        for (size_t tree = 0; tree < TreeCount; ++tree) {
            trees->AddBinTree(splits);
            for (size_t leaf = 0; leaf < (size_t(1) << depth); ++leaf) {
                trees->AddLeafValue((double(leaf) - 127.0) * (tree + 1) / 8.0);
            }
        }
        model.UpdateDynamicData();
        TVector<TVector<float>> data(docCount, TVector<float>(depth));
        TVector<TConstArrayRef<float>> features(docCount);
        for (size_t doc = 0; doc < docCount; ++doc) {
            for (size_t feature = 0; feature < depth; ++feature) {
                data[doc][feature] = ((doc * 73) >> feature) & 1;
            }
            features[doc] = data[doc];
        }
        TVector<double> results(docCount);
        model.CalcFlat(features, results);
        for (auto _ : state) {
            model.CalcFlat(features, results);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * docCount * TreeCount);
    }
    BENCHMARK(EvaluateModel)->ArgsProduct({{1, 8, 16, 32, 64, 128, 129, 1024}, {3, 6, 8}});
} // namespace
