#pragma once

#include "data_provider.h"
#include "binarization_config.h"
#include "grid_creator.h"
#include "binarizations_manager.h"
#include "data_utils.h"

#include <catboost/cuda/cuda_util/compression_helpers.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/data/pair.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/cuda/cuda_util/cpu_random.h>
#include <util/stream/file.h>
#include <util/system/spinlock.h>
#include <util/system/sem.h>

class TDataProviderBuilder: public IPoolBuilder {
public:
    using TSimpleCatFeatureBinarizationInfo = ymap<ui32, ymap<int, ui32>>;

    TDataProviderBuilder(TBinarizedFeaturesManager& featureManager,
                         TDataProvider& dst,
                         bool isTest = false,
                         ui32 buildThreads = 1)
        : FeaturesManager(featureManager)
        , DataProvider(dst)
        , IsTest(isTest)
        , BuildThreads(buildThreads)
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
            int hash = StringToIntHash(feature);
            //dirty c++ hack to store everything in float-vector
            AddFloatFeature(localIdx, featureId, ConvertCatFeatureHashToFloat(hash));
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

    void SetPairs(const yvector<TPair>& /*pairs*/) override {
        CB_ENSURE(false, "This function is not implemented in cuda");
    }

    int GetDocCount() const override {
        CB_ENSURE(false, "This function is not implemented in cuda");
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
        featureNames.resize(FeatureValues.size());

        TFastSemaphore binarizationSemaphore(GetThreadLimitForBordersType(FeaturesManager.GetDefaultFloatFeatureBinarizationDescription().BorderSelectionType));
        TAdaptiveLock lock;

        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(BuildThreads - 1);

        yvector<TFeatureColumnPtr> featureColumns(FeatureValues.size());

        NPar::ParallelFor(executor, 0, FeatureValues.size(), [&](ui32 featureId) {
            auto featureName = GetFeatureName(featureId);
            featureNames[featureId] = featureName;

            if (FeatureValues[featureId].size() == 0) {
                return;
            }

            yvector<float> line(DataProvider.Order.size());
            for (ui32 i = 0; i < DataProvider.Order.size(); ++i) {
                line[i] = FeatureValues[featureId][DataProvider.Order[i]];
            }

            if (CatFeatureIds.has(featureId)) {
                static_assert(sizeof(float) == sizeof(ui32), "Error: float size should be equal to ui32 size");
                auto data = ComputeCatFeatureBinarization(featureId,
                                                          reinterpret_cast<int*>(line.data()),
                                                          line.size());

                const auto& catFeatureBinarization = CatFeatureBinarizations[featureId];
                const ui32 uniqueValues = catFeatureBinarization.size();

                if (uniqueValues > 1) {
                    auto compressedData = CompressVector<ui64>(~data, line.size(), IntLog2(uniqueValues));
                    featureColumns[featureId] = MakeHolder<TCatFeatureValuesHolder>(featureId,
                                                                                    line.size(),
                                                                                    std::move(compressedData),
                                                                                    uniqueValues,
                                                                                    featureName);
                }

            } else {
                auto floatFeature = MakeHolder<TFloatValuesHolder>(featureId,
                                                                   std::move(line),
                                                                   featureName);

                yvector<float> borders;

                if (FeaturesManager.IsKnown(*floatFeature)) {
                    borders = FeaturesManager.GetFloatFeatureBorders(*floatFeature);
                }

                if (borders.empty() && !IsTest) {
                    TGuard<TFastSemaphore> guard(binarizationSemaphore);
                    TOnCpuGridBuilderFactory gridBuilderFactory;
                    borders = TBordersBuilder(gridBuilderFactory, floatFeature->GetValues())(FeaturesManager.GetDefaultFloatFeatureBinarizationDescription());
                }
                if (borders.ysize() == 0) {
                    MATRIXNET_WARNING_LOG << "Float Feature #" << featureId << " is empty" << Endl;
                    return;
                }

                auto binarizedData = BinarizeLine(floatFeature->GetValues().data(), floatFeature->GetValues().size(), borders);
                auto compressedLine = CompressVector<ui64>(binarizedData, IntLog2(borders.size() + 1));

                featureColumns[featureId] = MakeHolder<TBinarizedFloatValuesHolder>(featureId,
                                                                                    floatFeature->GetValues().size(),
                                                                                    borders,
                                                                                    std::move(compressedLine),
                                                                                    featureName);
            }

            //Free memory
            {
                auto emptyVec = yvector<float>();
                FeatureValues[featureId].swap(emptyVec);
            }
        });

        for (ui32 featureId = 0; featureId < featureColumns.size(); ++featureId) {
            if (CatFeatureIds.has(featureId)) {
                if (featureColumns[featureId] == nullptr) {
                    if (!IsTest) {
                        MATRIXNET_WARNING_LOG << "Cat Feature #" << featureId << " is empty" << Endl;
                        FeaturesManager.AddEmptyCatFeature(featureId);
                    }
                    continue;
                }
                const TCatFeatureValuesHolder& catFeatureValues = dynamic_cast<TCatFeatureValuesHolder&>(*featureColumns[featureId]);

                if (IsTest) {
                    if (FeaturesManager.GetBinCount(FeaturesManager.GetId(catFeatureValues)) <= 1) {
                        if (catFeatureValues.GetUniqueValues() > FeaturesManager.GetBinCount(FeaturesManager.GetId(catFeatureValues))) {
                            MATRIXNET_WARNING_LOG << "Cat Feature #" << featureId << " is empty in learn set and has #"
                                                  << catFeatureValues.GetUniqueValues() << " unique values in test set"
                                                  << Endl;
                        }
                        continue;
                    }
                    FeaturesManager.UpdateUniqueValues(catFeatureValues);
                } else {
                    FeaturesManager.AddCatFeature(catFeatureValues);
                }
                DataProvider.Features.push_back(std::move(featureColumns[featureId]));

            } else {
                if (featureColumns[featureId] != nullptr) {
                    const TBinarizedFloatValuesHolder& binarizedFloatValuesHolder = dynamic_cast<TBinarizedFloatValuesHolder&>(*featureColumns[featureId]);

                    if (!FeaturesManager.IsKnown(binarizedFloatValuesHolder)) {
                        FeaturesManager.AddFloatFeature(binarizedFloatValuesHolder);
                    }
                    DataProvider.Features.push_back(std::move(featureColumns[featureId]));
                } else {
                    if (!IsTest) {
                        FeaturesManager.AddEmptyFloatFeature(featureId); //for pretty printing in logs and with ignored features
                    }
                }
            }
        }

        GroupQueries(DataProvider.QueryIds,
                     &DataProvider.Queries);

        DataProvider.BuildIndicesRemap();

        if (!IsTest) {
            TOnCpuGridBuilderFactory gridBuilderFactory;
            FeaturesManager.SetTargetBorders(TBordersBuilder(gridBuilderFactory,
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
    ui32 GetThreadLimitForBordersType(EBorderSelectionType borderSelectionType) {
        switch (borderSelectionType) {
            case EBorderSelectionType::MinEntropy:
            case EBorderSelectionType::MaxLogSum:
                return 8;
            default:
                return 16;
        }
    }

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
                                                const int* hashes,
                                                ui32 hashesSize) {
        ymap<int, ui32> binarization;
        {
            TGuard<TAdaptiveLock> guard(CatFeatureBinarizationLock);
            binarization.swap(CatFeatureBinarizations[featureId]);
        }

        yvector<ui32> bins(hashesSize, 0);
        for (ui32 i = 0; i < hashesSize; ++i) {
            auto hash = hashes[i];
            if (binarization.count(hash) == 0) {
                binarization[hash] = (unsigned int)binarization.size();
            }
            bins[i] = binarization[hash];
        }

        {
            TGuard<TAdaptiveLock> guard(CatFeatureBinarizationLock);
            CatFeatureBinarizations[featureId].swap(binarization);
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
    TDataProvider& DataProvider;
    bool IsTest;
    ui32 BuildThreads;
    TAdaptiveLock CatFeatureBinarizationLock;

    bool Shuffle = true;
    ui32 Cursor = 0;
    bool IsDone = false;
    yvector<yvector<float>> FeatureValues;
    TSimpleCatFeatureBinarizationInfo CatFeatureBinarizations;
    yset<ui32> IgnoreFeatures;
    yvector<TString> FeatureNames;
    yset<int> CatFeatureIds;
};
