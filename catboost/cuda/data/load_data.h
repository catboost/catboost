#pragma once

#include "data_provider.h"
#include "binarization_config.h"
#include "grid_creator.h"
#include "binarizations_manager.h"
#include "data_utils.h"

#include <catboost/cuda/cuda_util/compression_helpers.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/cuda/cuda_util/cpu_random.h>
#include <util/stream/file.h>

class TDataProviderBuilder: public IPoolBuilder {
public:
    using TSimpleCatFeatureBinarizationInfo = ymap<ui32, ymap<ui32, ui32>>;

    TDataProviderBuilder(TBinarizedFeaturesManager& featureManager,
                         IFactory<IGridBuilder>& gridBuilder,
                         TDataProvider& dst,
                         bool isTest = false)
        : FeaturesManager(featureManager)
        , GridBuilder(gridBuilder)
        , DataProvider(dst)
        , IsTest(isTest)
    {
    }

    template <class TContainer>
    TDataProviderBuilder& AddIgnoredFeatures(const TContainer& container) {
        for (auto& f : container) {
            IgnoreFeatures.insert(f);
        }
        return *this;
    }

    void Start(const TPoolColumnsMetaInfo& metaInfo) override {
        DataProvider.Features.clear();

        DataProvider.Baseline.clear();
        DataProvider.Baseline.resize(metaInfo.BaselineCount);

        if (!IsTest) {
            CatFeatureBinarizations.clear();
        }

        Cursor = 0;
        IsDone = false;
        FeatureValues.clear();
        FeatureValues.resize(metaInfo.FactorCount);

        CatFeatureIds = yset<int>(metaInfo.CatFeatureIds.begin(), metaInfo.CatFeatureIds.end());
    }

    TDataProviderBuilder& SetShuffleFlag(bool shuffle) {
        Shuffle = shuffle;
        return *this;
    }

    void SetExistingCatFeaturesBinarization(TSimpleCatFeatureBinarizationInfo&& binarizations) {
        CatFeatureBinarizations = std::move(binarizations);
    }

    void StartNextBlock(ui32 blockSize) override {
        Cursor = DataProvider.Targets.size();
        const auto newDataSize = Cursor + blockSize;

        DataProvider.Targets.resize(newDataSize);
        DataProvider.Weights.resize(newDataSize, 1.0);
        DataProvider.QueryIds.resize(newDataSize);

        for (ui32 i = Cursor; i < DataProvider.QueryIds.size(); ++i) {
            DataProvider.QueryIds[i] = i;
        }

        for (auto& baseline : DataProvider.Baseline) {
            baseline.resize(newDataSize);
        }
        for (ui32 factor = 0; factor < FeatureValues.size(); ++factor) {
            if (IgnoreFeatures.count(factor) == 0) {
                FeatureValues[factor].resize(newDataSize);
            }
        }

        DataProvider.DocIds.resize(newDataSize);
    }

    void AddCatFeature(ui32 localIdx,
                       ui32 featureId,
                       const TStringBuf& feature) override {
        if (IgnoreFeatures.count(featureId) == 0) {
            Y_ASSERT(CatFeatureIds.has(featureId));
            ui32 hash = StringToIntHash(feature);
            //dirty c++ hack to store everything in float-vector
            AddFloatFeature(localIdx, featureId, *reinterpret_cast<float*>(&hash));
        }
    }

    void AddFloatFeature(ui32 localIdx, ui32 featureId, float feature) override {
        if (IgnoreFeatures.count(featureId) == 0) {
            auto& featureColumn = FeatureValues[featureId];
            featureColumn[GetLineIdx(localIdx)] = feature;
        }
    }

    void AddTarget(ui32 localIdx, float value) override {
        DataProvider.Targets[GetLineIdx(localIdx)] = value;
    }

    void AddWeight(ui32 localIdx, float value) override {
        DataProvider.Weights[GetLineIdx(localIdx)] = value;
    }

    void AddQueryId(ui32 localIdx, const TStringBuf& queryId) override {
        DataProvider.QueryIds[GetLineIdx(localIdx)] = StringToIntHash(queryId);
    }

    void AddBaseline(ui32 localIdx, ui32 baselineIdx, double value) override {
        DataProvider.Baseline[baselineIdx][GetLineIdx(localIdx)] = (float)value;
    }

    void AddDocId(ui32 localIdx, const TStringBuf& value) override {
        ui32 docId = 0;
        CB_ENSURE(TryFromString<ui32>(value, docId),
                  "Only ui32 docIds are supported");
        DataProvider.DocIds[GetLineIdx(localIdx)] = docId;
    };

    void SetFeatureIds(const yvector<TString>& featureIds) override {
        FeatureNames = featureIds;
    }

    void Finish() {
        CB_ENSURE(!IsDone, "Error: can't finish more than once");
        DataProvider.Features.reserve(FeatureValues.size());

        DataProvider.Order.resize(DataProvider.Targets.size());
        std::iota(DataProvider.Order.begin(),
                  DataProvider.Order.end(), 0);

        if (Shuffle) {
            TRandom random(0);
            if (DataProvider.QueryIds.size() == 0) {
                MATRIXNET_INFO_LOG << "Warning: dataSet shuffle with query ids is not implemented yet";
            } else {
                std::random_shuffle(DataProvider.Order.begin(),
                                    DataProvider.Order.end(),
                                    random);
            }

            ApplyPermutation(DataProvider.Order, DataProvider.Weights);
            for (auto& baseline : DataProvider.Baseline) {
                ApplyPermutation(DataProvider.Order, baseline);
            }
            ApplyPermutation(DataProvider.Order, DataProvider.Targets);
            ApplyPermutation(DataProvider.Order, DataProvider.QueryIds);
            ApplyPermutation(DataProvider.Order, DataProvider.DocIds);
        }
        yvector<TString> featureNames;

        for (ui32 featureId = 0; featureId < FeatureValues.size(); ++featureId) {
            auto featureName = GetFeatureName(featureId);
            featureNames.push_back(featureName);

            if (FeatureValues[featureId].size() == 0) {
                continue;
            }

            yvector<float> line(DataProvider.Order.size());
            for (ui32 i = 0; i < DataProvider.Order.size(); ++i) {
                line[i] = FeatureValues[featureId][DataProvider.Order[i]];
            }

            if (CatFeatureIds.has(featureId)) {
                static_assert(sizeof(float) == sizeof(ui32), "Error: float size should be equal to ui32 size");
                auto data = ComputeCatFeatureBinarization(featureId,
                                                          reinterpret_cast<ui32*>(line.data()),
                                                          line.size());

                auto& catFeatureBinarization = CatFeatureBinarizations[featureId];
                const ui32 uniqueValues = catFeatureBinarization.size();

                if (uniqueValues <= 1) {
                    if (!IsTest) {
                        MATRIXNET_WARNING_LOG << "Cat Feature #" << featureId << " is empty" << Endl;
                        FeaturesManager.AddEmptyCatFeature(featureId);
                    }
                    continue;
                }

                auto compressedData = CompressVector<ui64>(~data, line.size(), IntLog2(uniqueValues));
                auto catFeature = MakeHolder<TCatFeatureValuesHolder>(featureId,
                                                                      line.size(),
                                                                      std::move(compressedData),
                                                                      uniqueValues,
                                                                      featureName);
                if (IsTest) {
                    FeaturesManager.UpdateUniqueValues(*catFeature);
                } else {
                    FeaturesManager.AddCatFeature(*catFeature);
                }

                DataProvider.Features.push_back(std::move(catFeature));
            } else {
                auto floatFeature = MakeHolder<TFloatValuesHolder>(featureId,
                                                                   std::move(line),
                                                                   featureName);

                auto borders = FeaturesManager.GetOrCreateFloatFeatureBorders(*floatFeature,
                                                                              TBordersBuilder(GridBuilder, floatFeature->GetValues()));

                if (borders.ysize() == 0) {
                    MATRIXNET_WARNING_LOG << "Float Feature #" << featureId << " is empty" << Endl;
                    continue;
                }

                auto binarizedData = BinarizeLine(floatFeature->GetValues().data(), floatFeature->GetValues().size(), borders);
                auto compressedLine = CompressVector<ui64>(binarizedData, IntLog2(borders.size() + 1));

                DataProvider.Features.push_back(MakeHolder<TBinarizedFloatValuesHolder>(featureId,
                                                                                        floatFeature->GetValues().size(),
                                                                                        borders,
                                                                                        std::move(compressedLine),
                                                                                        featureName));
            }

            //Free memory
            {
                auto emptyVec = yvector<float>();
                line.swap(emptyVec);
            }
        }

        GroupQueries(DataProvider.QueryIds,
                     &DataProvider.Queries);

        DataProvider.BuildIndicesRemap();

        if (!IsTest) {
            FeaturesManager.SetTargetBorders(TBordersBuilder(GridBuilder,
                                                             DataProvider.GetTargets())(FeaturesManager.GetTargetBinarizationDescription()));
        }

        DataProvider.FeatureNames = featureNames;
        DataProvider.CatFeatureIds = CatFeatureIds;

        IsDone = true;
    }

    void MoveBinarizationTo(TSimpleCatFeatureBinarizationInfo& dst) {
        CB_ENSURE(IsDone, "Error: can move binarization only after build process is finished");
        dst.swap(CatFeatureBinarizations);
    }

private:
    template <class T>
    void ApplyPermutation(const yvector<ui32>& order, yvector<T>& data) {
        if (data.size()) {
            yvector<T> tmp(data.begin(), data.end());
            for (ui32 i = 0; i < order.size(); ++i) {
                data[i] = tmp[order[i]];
            }
        }
    }
    yvector<ui32> ComputeCatFeatureBinarization(ui32 featureId,
                                                const ui32* hashes,
                                                ui32 hashesSize) {
        auto& binarization = CatFeatureBinarizations[featureId];

        yvector<ui32> bins(hashesSize, 0);
        for (ui32 i = 0; i < hashesSize; ++i) {
            ui32 hash = hashes[i];
            if (binarization.count(hash) == 0) {
                binarization[hash] = (unsigned int)binarization.size();
            }
            bins[i] = binarization[hash];
        }
        return bins;
    }

    inline ui32 GetLineIdx(ui32 localIdx) {
        return Cursor + localIdx;
    }

    inline TString GetFeatureName(ui32 featureId) {
        return FeatureNames.size() ? FeatureNames[featureId] : ToString<ui32>(featureId);
    }

    TBinarizedFeaturesManager& FeaturesManager;
    IFactory<IGridBuilder>& GridBuilder;
    TDataProvider& DataProvider;
    bool IsTest;

    bool Shuffle = true;
    ui32 Cursor = 0;
    bool IsDone = false;
    yvector<yvector<float>> FeatureValues;
    TSimpleCatFeatureBinarizationInfo CatFeatureBinarizations;
    yset<ui32> IgnoreFeatures;
    yvector<TString> FeatureNames;
    yset<int> CatFeatureIds;
};
