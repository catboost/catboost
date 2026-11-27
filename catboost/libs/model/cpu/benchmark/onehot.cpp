#include <catboost/libs/model/model.h>
#include <catboost/libs/model/cpu/onehot_simd.h>
#include <catboost/libs/cat_feature/cat_feature.h>
#include <benchmark/benchmark.h>

namespace {
    void CatOneHotInference(benchmark::State& state) {
        const size_t count = state.range(0);
        const size_t valueCount = state.range(1);
        TFullModel model;
        auto* trees = model.ModelTrees.GetMutable();
        trees->AddCatFeature(TCatFeature(true, 0, 0, ""));
        TOneHotFeature oneHot;
        oneHot.CatFeatureIndex = 0;
        for (size_t value = 0; value < valueCount; ++value) {
            oneHot.Values.push_back(value);
        }
        trees->AddOneHotFeature(oneHot);
        for (size_t tree = 0; tree < 512; ++tree) {
            TVector<int> splits;
            for (size_t level = 0; level < 6; ++level) {
                splits.push_back((tree * 7 + level * 3) % valueCount);
            }
            trees->AddBinTree(splits);
            for (size_t leaf = 0; leaf < 64; ++leaf) {
                trees->AddLeafValue(double((leaf * 17 + tree * 11) % 129) / 64.0);
            }
        }
        model.UpdateDynamicData();
        TVector<TVector<float>> storage(count, TVector<float>(1));
        TVector<TConstArrayRef<float>> features(count);
        TVector<double> expected(count), actual(count);
        for (size_t doc = 0; doc < count; ++doc) {
            storage[doc][0] = ConvertCatFeatureHashToFloat((doc * 73) % valueCount);
            features[doc] = storage[doc];
            model.CalcFlatSingle(features[doc], MakeArrayRef(expected.data() + doc, 1));
        }
        model.CalcFlat(features, actual);
        if (expected != actual) {
            state.SkipWithError("One-hot batch predictions differ from scalar reference");
            return;
        }
        for (auto _ : state) {
            model.CalcFlat(features, actual);
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * count * 512);
    }
    BENCHMARK(CatOneHotInference)->ArgsProduct({{8, 32, 64, 128, 1024}, {16, 128, 512}});

    void ScalarOneHot(const ui32* hashes, size_t count, const int* values, size_t valueCount, ui8* result) {
        for (size_t doc = 0; doc < count; ++doc) {
            for (size_t block = 0; block < valueCount; block += MAX_VALUES_PER_BIN) {
                for (size_t value = block; value < Min(block + MAX_VALUES_PER_BIN, valueCount); ++value) {
                    result[(block / MAX_VALUES_PER_BIN) * count + doc] |= ui8(hashes[doc] == ui32(values[value])) * (value - block + 1);
                }
            }
        }
    }
    void OneHotKernel(benchmark::State& state) {
        using namespace NCB::NModelEvaluation::NDetail;
        const size_t count = state.range(0), valueCount = state.range(1);
        TOneHotFunction kernel = state.range(2) == 0 ? ScalarOneHot
                                                     : (state.range(2) == 1 ? GetOneHotAvx2() : GetOneHotAvx512());
        if (!kernel) {
            state.SkipWithMessage("SIMD kernel unavailable on this compiler/CPU/OS");
            return;
        }
        TVector<ui32> hashes(count);
        TVector<int> values(valueCount);
        for (size_t index = 0; index < count; ++index) {
            hashes[index] = (index * 73) % (valueCount + 1);
        }
        for (size_t index = 0; index < valueCount; ++index) {
            values[index] = index;
        }
        TVector<ui8> result(count * ((valueCount + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN), 0);
        auto expected = result;
        ScalarOneHot(hashes.data(), count, values.data(), valueCount, expected.data());
        kernel(hashes.data(), count, values.data(), valueCount, result.data());
        if (result != expected) {
            state.SkipWithError("One-hot kernel differs from scalar reference");
            return;
        }
        for (auto _ : state) {
            kernel(hashes.data(), count, values.data(), valueCount, result.data());
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * count);
    }
    BENCHMARK(OneHotKernel)->ArgsProduct({{32, 128}, {16, 128, 512}, {0, 1, 2}});

} // namespace
