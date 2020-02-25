#pragma once

#include "compressed_index.h"
#include "ctr_helper.h"
#include "batch_binarized_ctr_calcer.h"
#include "compressed_index_builder.h"
#include "estimated_features_calcer.h"
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/private/libs/quantization/utils.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/data/data_utils.h>

#include <library/threading/local_executor/local_executor.h>
#include <util/generic/fwd.h>

namespace NCatboostCuda {
    TMirrorBuffer<ui8> BuildBinarizedTarget(const TBinarizedFeaturesManager& featuresManager,
                                            const TVector<float>& targets);

    template <class T>
    inline TVector<T> Join(TConstArrayRef<T> left,
                           TMaybe<TConstArrayRef<T>> right) {
        TVector<float> result(left.begin(), left.end());
        if (right) {
            for (const auto& element : *right) {
                result.push_back(element);
            }
        }
        return result;
    }

    void SplitByPermutationDependence(const TBinarizedFeaturesManager& featuresManager,
                                      const TVector<ui32>& features,
                                      const ui32 permutationCount,
                                      TVector<ui32>* permutationIndependent,
                                      TVector<ui32>* permutationDependent);

    THolder<TCtrTargets<NCudaLib::TMirrorMapping>> BuildCtrTarget(const TBinarizedFeaturesManager& featuresManager,
                                                                  const NCB::TTrainingDataProvider& dataProvider,
                                                                  const NCB::TTrainingDataProvider* test = nullptr);

    TVector<ui32> GetLearnFeatureIds(TBinarizedFeaturesManager& featuresManager);

    //filter features and write float and one-hot ones for selected dataSetId
    template <class TLayoutPolicy = TFeatureParallelLayout>
    class TFloatAndOneHotFeaturesWriter {
    public:
        TFloatAndOneHotFeaturesWriter(TBinarizedFeaturesManager& featuresManager,
                                      TSharedCompressedIndexBuilder<TLayoutPolicy>& indexBuilder,
                                      const NCB::TTrainingDataProvider& dataProvider,
                                      const ui32 dataSetId,
                                      NPar::TLocalExecutor* localExecutor)
            : FeaturesManager(featuresManager)
            , DataProvider(dataProvider)
            , DataSetId(dataSetId)
            , IndexBuilder(indexBuilder)
            , LocalExecutor(localExecutor)
        {
        }

        void Write(const TVector<ui32>& featureIds) {
            TVector<ui32> floatFeatureIds;
            for (auto feature : featureIds) {
                if (FeaturesManager.IsCtr(feature)) {
                    continue;
                } else if (FeaturesManager.IsFloat(feature)) {
                    floatFeatureIds.push_back(feature);
                } else if (FeaturesManager.IsCat(feature)) {
                    CB_ENSURE(FeaturesManager.UseForOneHotEncoding(feature));
                    WriteOneHotFeature(feature, DataProvider);
                    CheckInterrupted(); // check after long-lasting operation
                }
            }
            constexpr ui32 FeatureBlockSize = 16;
            const auto featureCount = floatFeatureIds.size();
            for (auto featureIdx : xrange<ui32>(0, featureCount, FeatureBlockSize)) {
                const auto begin = floatFeatureIds.begin() + featureIdx;
                const auto end = floatFeatureIds.begin() + Min<ui32>(featureCount, featureIdx + FeatureBlockSize);
                WriteFloatFeatures(MakeArrayRef(begin, end), DataProvider);
                CheckInterrupted(); // check after long-lasting operation
            }
        }

    private:
        void WriteFloatFeatures(TConstArrayRef<ui32> features,
                               const NCB::TTrainingDataProvider& dataProvider) {
            for (auto feature : features) {
                const auto featureId = FeaturesManager.GetDataProviderId(feature);
                const auto& featureMetaInfo = dataProvider.MetaInfo.FeaturesLayout->GetExternalFeaturesMetaInfo()[featureId];
                CB_ENSURE(featureMetaInfo.IsAvailable,
                        TStringBuilder() << "Feature #" << featureId << " is empty");
                CB_ENSURE(featureMetaInfo.Type == EFeatureType::Float,
                        TStringBuilder() << "Feature #" << featureId << " is not float");
            }
            const auto& objectsData = *dataProvider.ObjectsData;
            const auto featureCount = features.size();
            TVector<NCB::TMaybeOwningArrayHolder<ui8>> featureValues(featureCount);
            TVector<ui32> featureBinCounts(featureCount);
            LocalExecutor->ExecRangeWithThrow(
                [&] (int taskIdx) {
                    const auto feature = features[taskIdx];
                    const auto featureId = FeaturesManager.GetDataProviderId(feature);
                    const auto floatFeatureIdx = dataProvider.MetaInfo.FeaturesLayout->GetInternalFeatureIdx<EFeatureType::Float>(featureId);
                    featureBinCounts[taskIdx] = objectsData.GetQuantizedFeaturesInfo()->GetBinCount(floatFeatureIdx);
                    const auto& featuresHolder = **(objectsData.GetFloatFeature(*floatFeatureIdx));
                    featureValues[taskIdx] = featuresHolder.ExtractValues(LocalExecutor);
                },
                0, featureCount, NPar::TLocalExecutor::WAIT_COMPLETE
            );
            for (auto taskIdx : xrange(featureCount)) {
                IndexBuilder.template Write<ui8>(DataSetId, features[taskIdx], featureBinCounts[taskIdx], *featureValues[taskIdx]);
            }
        }

        void WriteOneHotFeature(const ui32 feature,
                                const NCB::TTrainingDataProvider& dataProvider) {
            const auto featureId = FeaturesManager.GetDataProviderId(feature);
            const auto& featureMetaInfo = dataProvider.MetaInfo.FeaturesLayout->GetExternalFeaturesMetaInfo()[featureId];
            CB_ENSURE(featureMetaInfo.IsAvailable,
                      TStringBuilder() << "Feature #" << featureId << " is empty");
            CB_ENSURE(featureMetaInfo.Type == EFeatureType::Categorical,
                      TStringBuilder() << "Feature #" << featureId << " is not categorical");

            auto catFeatureIdx = dataProvider.MetaInfo.FeaturesLayout->GetInternalFeatureIdx<EFeatureType::Categorical>(featureId);

            const auto& featuresHolder = **(dataProvider.ObjectsData->GetCatFeature(*catFeatureIdx));
            IndexBuilder.template Write<ui32>(DataSetId,
                                              feature,
                                              dataProvider.ObjectsData->GetQuantizedFeaturesInfo()->GetUniqueValuesCounts(catFeatureIdx).OnAll,
                                              *featuresHolder.ExtractValues(LocalExecutor));
        }

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const NCB::TTrainingDataProvider& DataProvider;
        ui32 DataSetId = -1;
        TSharedCompressedIndexBuilder<TLayoutPolicy>& IndexBuilder;
        NPar::TLocalExecutor* LocalExecutor;
    };

    template <class TLayoutPolicy = TFeatureParallelLayout>
    class TCtrsWriter {
    public:
        TCtrsWriter(TBinarizedFeaturesManager& featuresManager,
                    TSharedCompressedIndexBuilder<TLayoutPolicy>& indexBuilder,
                    TBatchedBinarizedCtrsCalcer& binarizedCtrCalcer,
                    ui32 dataSetId,
                    ui32 testSetId = -1)
            : FeaturesManager(featuresManager)
            , IndexBuilder(indexBuilder)
            , CtrCalcer(binarizedCtrCalcer)
            , DataSetId(dataSetId)
            , TestDataSetId(testSetId)
        {
        }

        void Write(const TVector<ui32>& featureIds) {
            TVector<ui32> ctrs = TakeCtrs(featureIds);

            if (ctrs.size()) {
                auto batchGroups = CreateBatchGroups(ctrs);

                for (const auto& group : batchGroups) {
                    TVector<TBatchedBinarizedCtrsCalcer::TBinarizedCtr> learnCtrs;
                    TVector<TBatchedBinarizedCtrsCalcer::TBinarizedCtr> testCtrs;
                    CtrCalcer.ComputeBinarizedCtrs(group, &learnCtrs, &testCtrs);
                    for (ui32 i = 0; i < group.size(); ++i) {
                        const ui32 featureId = group[i];
                        IndexBuilder.template Write<ui8>(DataSetId,
                                                         featureId,
                                                         learnCtrs[i].BinCount,
                                                         learnCtrs[i].BinarizedCtr);

                        if (testCtrs.size()) {
                            CB_ENSURE(TestDataSetId != (ui32)-1, "Error: set test dataset");
                            CB_ENSURE(testCtrs[i].BinCount == learnCtrs[i].BinCount);

                            IndexBuilder.template Write<ui8>(TestDataSetId,
                                                             featureId,
                                                             testCtrs[i].BinCount,
                                                             testCtrs[i].BinarizedCtr);
                        }
                    }
                    CheckInterrupted(); // check after long-lasting operation
                }
            }
        }

    private:
        TVector<ui32> TakeCtrs(const TVector<ui32>& featureIds) {
            TVector<ui32> ctrs;
            for (auto feature : featureIds) {
                if (FeaturesManager.IsCtr(feature)) {
                    ctrs.push_back(feature);
                } else {
                    continue;
                }
            }
            return ctrs;
        }

        TVector<TVector<ui32>> CreateBatchGroups(const TVector<ui32>& features) {
            TMap<TFeatureTensor, TVector<ui32>> byTensorGroups;
            ui32 deviceCount = NCudaLib::GetCudaManager().GetDevices(UseOnlyLocalDevices).size();
            for (auto& feature : features) {
                byTensorGroups[FeaturesManager.GetCtr(feature).FeatureTensor].push_back(feature);
            }

            TVector<TVector<ui32>> batchGroups;
            ui32 tensorCount = deviceCount;

            for (auto& group : byTensorGroups) {
                if (tensorCount >= deviceCount) {
                    batchGroups.push_back(TVector<ui32>());
                    tensorCount = 0;
                }
                auto& ids = group.second;
                for (auto id : ids) {
                    batchGroups.back().push_back(id);
                }
                ++tensorCount;
            }
            if (batchGroups.size() && batchGroups.back().size() == 0) {
                batchGroups.pop_back();
            }
            return batchGroups;
        }

    private:
        const bool UseOnlyLocalDevices = true;

        TBinarizedFeaturesManager& FeaturesManager;
        TSharedCompressedIndexBuilder<TLayoutPolicy>& IndexBuilder;
        TBatchedBinarizedCtrsCalcer& CtrCalcer;
        ui32 DataSetId = -1;
        ui32 TestDataSetId = -1;
    };



    template <class TLayoutPolicy = TFeatureParallelLayout>
    class TEstimatedFeaturesWriter {
    public:

        TEstimatedFeaturesWriter(TBinarizedFeaturesManager& featuresManager,
                                 TSharedCompressedIndexBuilder<TLayoutPolicy>& indexBuilder,
                                 TEstimatorsExecutor& featuresCalcer,
                                 ui32 dataSetId,
                                 TMaybe<ui32> testSetId = Nothing())
            : FeaturesManager(featuresManager)
              , IndexBuilder(indexBuilder)
              , EstimatorsExecutor(featuresCalcer)
              , DataSetId(dataSetId)
              , TestDataSetId(testSetId) {
        }

        void Write(const TVector<ui32>& featureIds) {
            using namespace std::placeholders;
            THashSet<ui32> genericFeaturesToEstimate = TakeFeaturesToEstimate(featureIds);
            Write(
                genericFeaturesToEstimate,
                std::bind(&TEstimatorsExecutor::ExecEstimators, EstimatorsExecutor, _1, _2, _3)
            );

            THashSet<ui32> binaryFeaturesToEstimate = TakeFeaturesToEstimate(featureIds, true);
            Write(
                binaryFeaturesToEstimate,
                std::bind(&TEstimatorsExecutor::ExecBinaryFeaturesEstimators, EstimatorsExecutor, _1, _2, _3)
            );
        }

    private:
        template <class ExecEstimatorsFunc>
        void Write(const THashSet<ui32>& featureIds, ExecEstimatorsFunc&& execEstimators) {
            if (featureIds.empty()) {
                return;
            }
            auto estimators = GetEstimators(featureIds);

            auto binarizedWriter = [&](
                ui32 dataSetId,
                TConstArrayRef<ui8> binarizedFeature,
                TEstimatedFeature feature,
                ui8 binCount
            ) {
                const auto featureId = FeaturesManager.GetId(feature);
                if (featureIds.contains(featureId)) {
                    IndexBuilder.template Write<ui8>(dataSetId,
                                                     featureId,
                                                     binCount,
                                                     binarizedFeature);
                }
                CheckInterrupted(); // check after long-lasting operation
            };

            TEstimatorsExecutor::TBinarizedFeatureVisitor learnWriter = std::bind(binarizedWriter,
                                                                                  DataSetId,
                                                                                  std::placeholders::_1,
                                                                                  std::placeholders::_2,
                                                                                  std::placeholders::_3);

            TMaybe<TEstimatorsExecutor::TBinarizedFeatureVisitor> testWriter;
            if (TestDataSetId) {
                testWriter = std::bind(binarizedWriter,
                                       *TestDataSetId,
                                       std::placeholders::_1,
                                       std::placeholders::_2,
                                       std::placeholders::_3);
            }
            execEstimators(estimators, learnWriter, testWriter);
        }

        THashSet<ui32> TakeFeaturesToEstimate(const TVector<ui32>& featureIds, bool takeBinaryFeatures = false) {
            THashSet<ui32> result;
            for (const auto& feature : featureIds) {
                if (FeaturesManager.IsEstimatedFeature(feature)) {
                    const ui32 featureBinCount = FeaturesManager.GetBinCount(feature);
                    if (
                        (takeBinaryFeatures && (featureBinCount == 2)) ||
                        (!takeBinaryFeatures && (featureBinCount > 2))
                    ) {
                        result.insert(feature);
                    }
                }
            }
            return result;
        }

        TVector<NCB::TEstimatorId> GetEstimators(const THashSet<ui32>& features) {
            using NCB::TEstimatorId;
            THashSet<TEstimatorId> estimators;
            for (const auto& feature : features) {
                TEstimatorId id = FeaturesManager.GetEstimatedFeature(feature).EstimatorId;
                estimators.insert(id);
            }
            TVector<TEstimatorId> result;
            result.insert(result.end(), estimators.begin(), estimators.end());
            return result;
        }

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        TSharedCompressedIndexBuilder<TLayoutPolicy>& IndexBuilder;
        TEstimatorsExecutor& EstimatorsExecutor;
        ui32 DataSetId = -1;
        TMaybe<ui32> TestDataSetId;
    };


    extern template class TFloatAndOneHotFeaturesWriter<TFeatureParallelLayout>;
    extern template class TFloatAndOneHotFeaturesWriter<TDocParallelLayout>;

    extern template class TCtrsWriter<TFeatureParallelLayout>;
    extern template class TCtrsWriter<TDocParallelLayout>;

    extern template class TEstimatedFeaturesWriter<TFeatureParallelLayout>;
    extern template class TEstimatedFeaturesWriter<TDocParallelLayout>;



}
