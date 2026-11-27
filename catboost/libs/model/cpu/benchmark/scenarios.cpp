#include <catboost/libs/model/model.h>

#include <benchmark/benchmark.h>

#include <util/generic/vector.h>
#include <util/system/env.h>

namespace {
    TFullModel MakeScenario(size_t scenario) {
        const size_t depth = scenario == 3 ? 10 : (scenario == 0 ? 3 : (scenario == 2 ? 8 : 6));
        const size_t featureCount = scenario == 0 ? 8 : (scenario == 2 ? 256 : 64);
        const size_t borderCount = scenario == 0 ? 1 : (scenario == 2 ? 128 : 32);
        const size_t treeCount = scenario == 2 ? 1024 : 512;
        const size_t dimensions = scenario == 4 ? 3 : (scenario == 7 ? 16 : 1);
        TFullModel model;
        auto* trees = model.ModelTrees.GetMutable();
        trees->SetApproxDimension(dimensions);
        for (size_t feature = 0; feature < featureCount; ++feature) {
            TVector<float> borders;
            for (size_t border = 0; border < borderCount; ++border) {
                borders.push_back(float(border) / 4.0f - float(borderCount) / 8.0f);
            }
            trees->AddFloatFeature(TFloatFeature(false, feature, feature, std::move(borders), ""));
        }
        for (size_t tree = 0; tree < treeCount; ++tree) {
            TVector<int> splits;
            for (size_t level = 0; level < depth; ++level) {
                const size_t feature = (tree * 13 + level * 7) % featureCount;
                const size_t border = (tree * 17 + level * 11) % borderCount;
                splits.push_back(feature * borderCount + border);
            }
            trees->AddBinTree(splits);
            for (size_t leaf = 0; leaf < (size_t(1) << depth); ++leaf) {
                for (size_t dimension = 0; dimension < dimensions; ++dimension) {
                    trees->AddLeafValue((double((leaf * 73 + tree * 11 + dimension * 7) % 257) - 128.0) / 64.0);
                }
            }
        }
        model.UpdateDynamicData();
        if (scenario == 5) {
            model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        }
        if (scenario == 6) {
            model.SetScaleAndBias({0.75, {0.125}});
        }
        return model;
    }

    void RunScenario(benchmark::State& state, TFullModel model, size_t docCount, bool normalizedInputs = false, bool identicalDocs = false) {
        const size_t featureCount = model.GetNumFloatFeatures();
        const size_t dimensionCount = model.GetDimensionsCount();
        TVector<TVector<float>> data(docCount, TVector<float>(featureCount));
        TVector<TConstArrayRef<float>> features(docCount);
        for (size_t doc = 0; doc < docCount; ++doc) {
            const size_t sourceDoc = identicalDocs ? 0 : doc;
            for (size_t feature = 0; feature < featureCount; ++feature) {
                data[doc][feature] = normalizedInputs
                                         ? float(int((sourceDoc * 73 + feature * 37) % 2001) - 1000) / 333.0f
                                         : float((sourceDoc * 73 + feature * 37) % 1021) / 16.0f - 31.0f;
            }
            features[doc] = data[doc];
        }
        TVector<double> results(docCount * dimensionCount);
        model.CalcFlat(features, results);
        // Check the benchmark's complete output against single-document evaluation.
        for (size_t doc = 0; doc < docCount; ++doc) {
            TVector<double> expected(dimensionCount);
            model.CalcFlatSingle(features[doc], expected);
            for (size_t dimension = 0; dimension < dimensionCount; ++dimension) {
                if (results[doc * dimensionCount + dimension] != expected[dimension]) {
                    state.SkipWithError("Batch prediction differs from single-document reference");
                    return;
                }
            }
        }
        for (auto _ : state) {
            model.CalcFlat(features, results);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * docCount * model.GetTreeCount());
    }

    void CpuInference(benchmark::State& state) {
        RunScenario(state, MakeScenario(state.range(1)), state.range(0));
    }
    // scenario: 0 shallow, 1 mixed, 2 border-heavy, 3 deep,
    // 4 multiclass, 5 asymmetric, 6 nonidentity scale/bias.
    BENCHMARK(CpuInference)->ArgsProduct({{8, 32, 64, 128, 1024}, {0, 1, 2, 3, 4, 5, 6, 7}});

    void ExternalFloatModel(benchmark::State& state) {
        const auto path = GetEnv("CATBOOST_BENCH_MODEL");
        if (path.empty()) {
            state.SkipWithMessage("Set CATBOOST_BENCH_MODEL to a float-only model file");
            return;
        }
        RunScenario(state, ReadModel(path), state.range(0), true);
    }
    BENCHMARK(ExternalFloatModel)->Arg(8)->Arg(32)->Arg(64)->Arg(128)->Arg(1024);

    void IdenticalFloatDocs(benchmark::State& state) {
        const auto path = GetEnv("CATBOOST_BENCH_MODEL");
        if (path.empty()) {
            state.SkipWithMessage("Set CATBOOST_BENCH_MODEL for the identical-document workload");
            return;
        }
        RunScenario(state, ReadModel(path), state.range(0), true, true);
    }
    BENCHMARK(IdenticalFloatDocs)->Arg(8)->Arg(128)->Arg(1024);

    void ExternalModelPool(benchmark::State& state) {
        const auto path = GetEnv("CATBOOST_BENCH_MODEL");
        if (path.empty()) {
            state.SkipWithMessage("Set CATBOOST_BENCH_MODEL for the rotating model-pool workload");
            return;
        }
        TVector<TFullModel> models;
        for (size_t index = 0; index < 16; ++index) {
            models.push_back(ReadModel(path));
        }
        const size_t docCount = state.range(0);
        const size_t featureCount = models.front().GetNumFloatFeatures();
        TVector<TVector<float>> data(docCount, TVector<float>(featureCount));
        TVector<TConstArrayRef<float>> features(docCount);
        for (size_t doc = 0; doc < docCount; ++doc) {
            for (size_t feature = 0; feature < featureCount; ++feature) {
                data[doc][feature] = float(int((doc * 73 + feature * 37) % 2001) - 1000) / 333.0f;
            }
            features[doc] = data[doc];
        }
        TVector<double> expected(docCount), results(docCount);
        for (size_t doc = 0; doc < docCount; ++doc) {
            models.front().CalcFlatSingle(features[doc], MakeArrayRef(expected.data() + doc, 1));
        }
        for (const auto& model : models) {
            model.CalcFlat(features, results);
            if (results != expected) {
                state.SkipWithError("Model-pool predictions differ from scalar reference");
                return;
            }
        }
        size_t index = 0;
        for (auto _ : state) {
            models[index].CalcFlat(features, results);
            index = (index + 1) % models.size();
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * docCount * models.front().GetTreeCount());
    }
    BENCHMARK(ExternalModelPool)->Arg(8)->Arg(128)->Arg(1024);
} // namespace
