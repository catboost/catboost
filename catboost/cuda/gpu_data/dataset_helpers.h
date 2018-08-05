#pragma once

#include "compressed_index.h"
#include "ctr_helper.h"
#include "batch_binarized_ctr_calcer.h"
#include "compressed_index_builder.h"
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/cuda/data/binarizations_manager.h>

namespace NCatboostCuda {
    TMirrorBuffer<ui8> BuildBinarizedTarget(const TBinarizedFeaturesManager& featuresManager,
                                            const TVector<float>& targets);

    template <class T>
    inline TVector<T> Join(const TVector<T>& left,
                           const TVector<T>* right) {
        TVector<float> result(left.begin(), left.end());
        if (right) {
            for (auto& element : *right) {
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
                                                                  const TDataProvider& dataProvider,
                                                                  const TDataProvider* test = nullptr);

    TVector<ui32> GetLearnFeatureIds(TBinarizedFeaturesManager& featuresManager);

    //filter features and write float and one-hot ones for selected dataSetId
    template <class TLayoutPolicy = TFeatureParallelLayout>
    class TFloatAndOneHotFeaturesWriter {
    public:
        TFloatAndOneHotFeaturesWriter(TBinarizedFeaturesManager& featuresManager,
                                      TSharedCompressedIndexBuilder<TLayoutPolicy>& indexBuilder,
                                      const TDataProvider& dataProvider,
                                      const ui32 dataSetId)
            : FeaturesManager(featuresManager)
            , DataProvider(dataProvider)
            , DataSetId(dataSetId)
            , IndexBuilder(indexBuilder)
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
                               const TDataProvider& dataProvider) {
            const auto featureId = FeaturesManager.GetDataProviderId(feature);
            CB_ENSURE(dataProvider.HasFeatureId(featureId), TStringBuilder() << "Feature " << featureId << " (" << featureId << ")"
                                                                             << " is empty");
            const auto& featureStorage = dataProvider.GetFeatureById(featureId);

            if (featureStorage.GetType() == EFeatureValuesType::BinarizedFloat) {
                const auto& featuresHolder = dynamic_cast<const TBinarizedFloatValuesHolder&>(featureStorage);
                CB_ENSURE(featuresHolder.GetBorders() == FeaturesManager.GetBorders(feature),
                          "Error: inconsistent borders for feature #" << feature);

                const ui32 binCount = featuresHolder.BinCount();
                IndexBuilder.Write(DataSetId,
                                   feature,
                                   binCount,
                                   featuresHolder.ExtractValues<ui8>());

            } else {
                CB_ENSURE(featureStorage.GetType() == EFeatureValuesType::Float);

                const auto& holder = dynamic_cast<const TFloatValuesHolder&>(featureStorage);
                const auto& borders = FeaturesManager.GetBorders(feature);
                const ENanMode nanMode = FeaturesManager.GetOrComputeNanMode(holder);

                auto bins = BinarizeLine<ui32>(holder.GetValuesPtr(),
                                               holder.GetSize(),
                                               nanMode,
                                               borders);
                const ui32 binCount = borders.size() + 1 + (nanMode != ENanMode::Forbidden);

                IndexBuilder.Write(DataSetId,
                                   feature,
                                   binCount,
                                   bins);
            }
        }

        void WriteOneHotFeature(const ui32 feature,
                                const TDataProvider& dataProvider) {
            const auto& featureStorage = dataProvider.GetFeatureById(FeaturesManager.GetDataProviderId(feature));

            CB_ENSURE(featureStorage.GetType() == EFeatureValuesType::Categorical);

            const auto& featuresHolder = dynamic_cast<const ICatFeatureValuesHolder&>(featureStorage);
            const auto binCount = Min<ui32>(FeaturesManager.GetBinCount(feature), featuresHolder.GetUniqueValues());
            IndexBuilder.Write(DataSetId,
                               feature,
                               /*hack for using only learn catFeatures as one-hot splits. test set storage should know about unseen catFeatures from learn. binCount is totalUniqueValues (learn + test)*/
                               binCount,
                               featuresHolder.ExtractValues());
        }

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const TDataProvider& DataProvider;
        ui32 DataSetId = -1;
        TSharedCompressedIndexBuilder<TLayoutPolicy>& IndexBuilder;
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
                        IndexBuilder.Write(DataSetId,
                                           featureId,
                                           learnCtrs[i].BinCount,
                                           learnCtrs[i].BinarizedCtr);

                        if (testCtrs.size()) {
                            CB_ENSURE(TestDataSetId != (ui32)-1, "Error: set test dataset");
                            CB_ENSURE(testCtrs[i].BinCount == learnCtrs[i].BinCount);

                            IndexBuilder.Write(TestDataSetId,
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
