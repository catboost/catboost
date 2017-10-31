#pragma once

#include "data_provider.h"
#include "binarization_config.h"
#include "grid_creator.h"
#include "binarizations_manager.h"
#include "data_utils.h"
#include "cat_feature_perfect_hash_helper.h"

#include <catboost/cuda/cuda_util/compression_helpers.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/data/pair.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/cuda/cuda_util/cpu_random.h>
#include <util/stream/file.h>
#include <util/system/spinlock.h>
#include <util/system/sem.h>
#include <util/random/shuffle.h>

namespace NCatboostCuda
{
    class TDataProviderBuilder: public IPoolBuilder
    {
    public:

        TDataProviderBuilder(TBinarizedFeaturesManager& featureManager,
                             TDataProvider& dst,
                             bool isTest = false,
                             const int buildThreads = 1)
                : FeaturesManager(featureManager)
                  , DataProvider(dst)
                  , IsTest(isTest)
                  , BuildThreads(buildThreads)
                  , CatFeaturesPerfectHashHelper(FeaturesManager)
        {
        }

        template<class TContainer>
        TDataProviderBuilder& AddIgnoredFeatures(const TContainer& container)
        {
            for (auto& f : container)
            {
                IgnoreFeatures.insert(f);
            }
            return *this;
        }

        void Start(const TPoolColumnsMetaInfo& metaInfo, int /*docCount*/) override
        {
            DataProvider.Features.clear();

            DataProvider.Baseline.clear();
            DataProvider.Baseline.resize(metaInfo.BaselineCount);

            Cursor = 0;
            IsDone = false;
            FeatureValues.clear();
            FeatureValues.resize(metaInfo.FactorCount);

            CatFeatureIds = yset<int>(metaInfo.CatFeatureIds.begin(), metaInfo.CatFeatureIds.end());
        }

        TDataProviderBuilder& SetShuffleFlag(bool shuffle)
        {
            ShuffleFlag = shuffle;
            return *this;
        }

        void StartNextBlock(ui32 blockSize) override
        {
            Cursor = DataProvider.Targets.size();
            const auto newDataSize = Cursor + blockSize;

            DataProvider.Targets.resize(newDataSize);
            DataProvider.Weights.resize(newDataSize, 1.0);
            DataProvider.QueryIds.resize(newDataSize);

            for (ui32 i = Cursor; i < DataProvider.QueryIds.size(); ++i)
            {
                DataProvider.QueryIds[i] = i;
            }

            for (auto& baseline : DataProvider.Baseline)
            {
                baseline.resize(newDataSize);
            }
            for (ui32 factor = 0; factor < FeatureValues.size(); ++factor)
            {
                if (IgnoreFeatures.count(factor) == 0)
                {
                    FeatureValues[factor].resize(newDataSize);
                }
            }

            DataProvider.DocIds.resize(newDataSize);
        }

        float GetCatFeatureValue(const TStringBuf& feature) override
        {
            return ConvertCatFeatureHashToFloat(StringToIntHash(feature));
        }

        void AddCatFeature(ui32 localIdx,
                           ui32 featureId,
                           const TStringBuf& feature) override
        {
            if (IgnoreFeatures.count(featureId) == 0)
            {
                Y_ASSERT(CatFeatureIds.has(featureId));
                int hash = StringToIntHash(feature);
                //dirty c++ hack to store everything in float-vector
                AddFloatFeature(localIdx, featureId, ConvertCatFeatureHashToFloat(hash));
            }
        }

        void AddFloatFeature(ui32 localIdx, ui32 featureId, float feature) override
        {
            if (IgnoreFeatures.count(featureId) == 0)
            {
                auto& featureColumn = FeatureValues[featureId];
                featureColumn[GetLineIdx(localIdx)] = feature;
            }
        }

        void AddAllFloatFeatures(ui32 localIdx, const yvector<float>& features) override
        {
            CB_ENSURE(features.ysize() == FeatureValues.ysize(),
                      "Error: number of features should be equal to factor count");
            for (int featureId = 0; featureId < FeatureValues.ysize(); ++featureId)
            {
                if (IgnoreFeatures.count(featureId) == 0)
                {
                    auto& featureColumn = FeatureValues[featureId];
                    featureColumn[GetLineIdx(localIdx)] = features[featureId];
                }
            }
        }

        void AddTarget(ui32 localIdx, float value) override
        {
            DataProvider.Targets[GetLineIdx(localIdx)] = value;
        }

        void AddWeight(ui32 localIdx, float value) override
        {
            DataProvider.Weights[GetLineIdx(localIdx)] = value;
        }

        void AddQueryId(ui32 localIdx, ui32 queryId) override
        {
            DataProvider.QueryIds[GetLineIdx(localIdx)] = queryId;
        }

        void AddBaseline(ui32 localIdx, ui32 baselineIdx, double value) override
        {
            DataProvider.Baseline[baselineIdx][GetLineIdx(localIdx)] = (float) value;
        }

        void AddDocId(ui32 localIdx, const TStringBuf& value) override
        {
            DataProvider.DocIds[GetLineIdx(localIdx)] = StringToIntHash(value);
        };

        void SetFeatureIds(const yvector<TString>& featureIds) override
        {
            FeatureNames = featureIds;
        }

        void SetPairs(const yvector<TPair>& /*pairs*/) override
        {
            CB_ENSURE(false, "This function is not implemented in cuda");
        }

        int GetDocCount() const override
        {
            CB_ENSURE(false, "This function is not implemented in cuda");
        }

        void Finish() override
        {
            CB_ENSURE(!IsDone, "Error: can't finish more than once");
            DataProvider.Features.reserve(FeatureValues.size());

            DataProvider.Order.resize(DataProvider.Targets.size());
            std::iota(DataProvider.Order.begin(),
                      DataProvider.Order.end(), 0);

            if (ShuffleFlag)
            {
                TRandom random(0);
                if (DataProvider.QueryIds.empty())
                {
                    MATRIXNET_INFO_LOG << "Warning: dataSet shuffle with query ids is not implemented yet";
                } else
                {
                    Shuffle(DataProvider.Order.begin(),
                            DataProvider.Order.end(),
                            random);
                }

                ApplyPermutation(DataProvider.Order, DataProvider.Weights);
                for (auto& baseline : DataProvider.Baseline)
                {
                    ApplyPermutation(DataProvider.Order, baseline);
                }
                ApplyPermutation(DataProvider.Order, DataProvider.Targets);
                ApplyPermutation(DataProvider.Order, DataProvider.QueryIds);
                ApplyPermutation(DataProvider.Order, DataProvider.DocIds);
            }
            yvector<TString> featureNames;
            featureNames.resize(FeatureValues.size());

            TAdaptiveLock lock;

            NPar::TLocalExecutor executor;
            executor.RunAdditionalThreads(BuildThreads - 1);

            yvector<TFeatureColumnPtr> featureColumns(FeatureValues.size());

            if (!IsTest)
            {
                RegisterFeaturesInFeatureManager(featureColumns);
            }

            yvector<yvector<float>> grid;
            grid.resize(FeatureValues.size());

            NPar::ParallelFor(executor, 0, FeatureValues.size(), [&](ui32 featureId)
            {
                auto featureName = GetFeatureName(featureId);
                featureNames[featureId] = featureName;

                if (FeatureValues[featureId].size() == 0)
                {
                    return;
                }

                yvector<float> line(DataProvider.Order.size());
                for (ui32 i = 0; i < DataProvider.Order.size(); ++i)
                {
                    line[i] = FeatureValues[featureId][DataProvider.Order[i]];
                }

                if (CatFeatureIds.has(featureId))
                {
                    static_assert(sizeof(float) == sizeof(ui32), "Error: float size should be equal to ui32 size");
                    const bool shouldSkip = IsTest && (CatFeaturesPerfectHashHelper.GetUniqueValues(featureId) == 0);
                    if (!shouldSkip)
                    {
                        auto data = CatFeaturesPerfectHashHelper.UpdatePerfectHashAndBinarize(featureId,
                                                                                              ~line,
                                                                                              line.size());

                        const ui32 uniqueValues = CatFeaturesPerfectHashHelper.GetUniqueValues(featureId);

                        if (uniqueValues > 1)
                        {
                            auto compressedData = CompressVector<ui64>(~data, line.size(), IntLog2(uniqueValues));
                            featureColumns[featureId] = MakeHolder<TCatFeatureValuesHolder>(featureId,
                                                                                            line.size(),
                                                                                            std::move(compressedData),
                                                                                            uniqueValues,
                                                                                            featureName);
                        }
                    }
                } else
                {
                    auto floatFeature = MakeHolder<TFloatValuesHolder>(featureId,
                                                                       std::move(line),
                                                                       featureName);

                    yvector<float>& borders = grid[featureId];

                    if (FeaturesManager.HasFloatFeatureBorders(*floatFeature))
                    {
                        borders = FeaturesManager.GetFloatFeatureBorders(*floatFeature);
                    }

                    if (borders.empty() && !IsTest)
                    {
                        const auto& floatValues = floatFeature->GetValues();
                        const auto& config = FeaturesManager.GetDefaultFloatFeatureBinarizationDescription();
                        borders = BuildBorders(floatValues, floatFeature->GetId(), config);
                    }
                    if (borders.ysize() == 0)
                    {
                        MATRIXNET_INFO_LOG << "Float Feature #" << featureId << " is empty" << Endl;
                        return;
                    }

                    auto binarizedData = BinarizeLine(floatFeature->GetValues().data(),
                                                      floatFeature->GetValues().size(), borders);
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

            for (ui32 featureId = 0; featureId < featureColumns.size(); ++featureId)
            {
                if (CatFeatureIds.has(featureId))
                {
                    if (featureColumns[featureId] == nullptr && (!IsTest))
                    {
                        MATRIXNET_INFO_LOG << "Cat Feature #" << featureId << " is empty" << Endl;
                    }
                } else
                {
                    if (!FeaturesManager.HasFloatFeatureBordersForDataProviderFeature(featureId))
                    {
                        FeaturesManager.SetFloatFeatureBordersForDataProviderId(featureId, std::move(grid[featureId]));
                    }
                }
                if (featureColumns[featureId] != nullptr)
                {
                    DataProvider.Features.push_back(std::move(featureColumns[featureId]));
                }
            }

            GroupQueries(DataProvider.QueryIds,
                         &DataProvider.Queries);

            DataProvider.BuildIndicesRemap();

            if (!IsTest)
            {
                TOnCpuGridBuilderFactory gridBuilderFactory;
                FeaturesManager.SetTargetBorders(TBordersBuilder(gridBuilderFactory,
                                                                 DataProvider.GetTargets())(
                        FeaturesManager.GetTargetBinarizationDescription()));
            }

            DataProvider.FeatureNames = featureNames;
            DataProvider.CatFeatureIds = CatFeatureIds;

            IsDone = true;
        }

        void RegisterFeaturesInFeatureManager(const yvector<TFeatureColumnPtr>& featureColumns) const
        {
            for (ui32 featureId = 0; featureId < featureColumns.size(); ++featureId)
            {
                if (!FeaturesManager.IsKnown(featureId))
                {
                    if (CatFeatureIds.has(featureId))
                    {
                        FeaturesManager.RegisterDataProviderCatFeature(featureId);
                    } else
                    {
                        FeaturesManager.RegisterDataProviderFloatFeature(featureId);
                    }
                }
            }
        }

    private:
        inline ui32 GetLineIdx(ui32 localIdx)
        {
            return Cursor + localIdx;
        }

        inline TString GetFeatureName(ui32 featureId)
        {
            return FeatureNames.size() ? FeatureNames[featureId] : ToString<ui32>(featureId);
        }

        TBinarizedFeaturesManager& FeaturesManager;
        TDataProvider& DataProvider;
        bool IsTest;
        ui32 BuildThreads;
        TCatFeaturesPerfectHashHelper CatFeaturesPerfectHashHelper;

        bool ShuffleFlag = true;
        ui32 Cursor = 0;
        bool IsDone = false;
        yvector<yvector<float>> FeatureValues;
        yset<ui32> IgnoreFeatures;
        yvector<TString> FeatureNames;
        yset<int> CatFeatureIds;
    };
}
