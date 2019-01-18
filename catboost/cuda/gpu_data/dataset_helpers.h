#pragma once

#include "compressed_index.h"
#include "ctr_helper.h"
#include "batch_binarized_ctr_calcer.h"
#include "compressed_index_builder.h"
#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/quantization/utils.h>
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
            for (auto feature : featureIds) {
                if (FeaturesManager.IsCtr(feature)) {
                    continue;
                } else if (FeaturesManager.IsFloat(feature)) {
                    WriteFloatFeature(feature,
                                      DataProvider);
                } else if (FeaturesManager.IsCat(feature)) {
                    CB_ENSURE(FeaturesManager.UseForOneHotEncoding(feature));
                    WriteOneHotFeature(feature, DataProvider);
                }
                CheckInterrupted(); // check after long-lasting operation
            }
        }

    private:
        void WriteFloatFeature(const ui32 feature,
                               const NCB::TTrainingDataProvider& dataProvider) {
            const auto featureId = FeaturesManager.GetDataProviderId(feature);
            const auto& featureMetaInfo = dataProvider.MetaInfo.FeaturesLayout->GetExternalFeaturesMetaInfo()[featureId];
            CB_ENSURE(featureMetaInfo.IsAvailable,
                      TStringBuilder() << "Feature #" << featureId << " is empty");
            CB_ENSURE(featureMetaInfo.Type == EFeatureType::Float,
                      TStringBuilder() << "Feature #" << featureId << " is not float");

            auto floatFeatureIdx = dataProvider.MetaInfo.FeaturesLayout->GetInternalFeatureIdx<EFeatureType::Float>(featureId);

            const auto& featuresHolder = **(dataProvider.ObjectsData->GetFloatFeature(*floatFeatureIdx));
            IndexBuilder.template Write<ui8>(DataSetId,
                                             feature,
                                             dataProvider.ObjectsData->GetQuantizedFeaturesInfo()->GetBinCount(floatFeatureIdx),
                                             *featuresHolder.ExtractValues(LocalExecutor));
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

    extern template class TFloatAndOneHotFeaturesWriter<TFeatureParallelLayout>;
    extern template class TFloatAndOneHotFeaturesWriter<TDocParallelLayout>;

    extern template class TCtrsWriter<TFeatureParallelLayout>;
    extern template class TCtrsWriter<TDocParallelLayout>;

}
