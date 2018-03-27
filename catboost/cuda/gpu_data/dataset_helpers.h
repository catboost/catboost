#pragma once

#include "compressed_index.h"
#include "ctr_helper.h"
#include "batch_binarized_ctr_calcer.h"
#include "compressed_index_builder.h"
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/cuda/data/binarizations_manager.h>

namespace NCatboostCuda {
    inline TMirrorBuffer<ui8> BuildBinarizedTarget(const TBinarizedFeaturesManager& featuresManager,
                                                   const TVector<float>& targets) {
        CB_ENSURE(featuresManager.HasTargetBinarization(),
                  "Error: target binarization should be set beforedataSet build");
        auto& borders = featuresManager.GetTargetBorders();

        auto binarizedTarget = BinarizeLine<ui8>(~targets,
                                                 targets.size(),
                                                 ENanMode::Forbidden,
                                                 borders);

        TMirrorBuffer<ui8> binarizedTargetGpu = TMirrorBuffer<ui8>::Create(
            NCudaLib::TMirrorMapping(binarizedTarget.size()));
        binarizedTargetGpu.Write(binarizedTarget);
        return binarizedTargetGpu;
    }

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

    inline void SplitByPermutationDependence(const TBinarizedFeaturesManager& featuresManager,
                                             const TVector<ui32>& features,
                                             const ui32 permutationCount,
                                             TVector<ui32>* permutationIndependent,
                                             TVector<ui32>* permutationDependent) {
        if (permutationCount == 1) {
            //            shortcut
            (*permutationIndependent) = features;
            return;
        }
        permutationDependent->clear();
        permutationIndependent->clear();
        for (const auto& feature : features) {
            const bool needPermutationFlag = featuresManager.IsCtr(feature) && featuresManager.IsPermutationDependent(featuresManager.GetCtr(feature));
            if (needPermutationFlag) {
                permutationDependent->push_back(feature);
            } else {
                permutationIndependent->push_back(feature);
            }
        }
    }

    inline THolder<TCtrTargets<NCudaLib::TMirrorMapping>> BuildCtrTarget(const TBinarizedFeaturesManager& featuresManager,
                                                                         const TDataProvider& dataProvider,
                                                                         const TDataProvider* test = nullptr) {
        TVector<float> joinedTarget = Join(dataProvider.GetTargets(),
                                           test ? &test->GetTargets() : nullptr);

        THolder<TCtrTargets<NCudaLib::TMirrorMapping>> ctrsTargetPtr;
        ctrsTargetPtr = new TCtrTargets<NCudaLib::TMirrorMapping>;
        auto& ctrsTarget = *ctrsTargetPtr;
        ctrsTarget.BinarizedTarget = BuildBinarizedTarget(featuresManager,
                                                          joinedTarget);

        ctrsTarget.WeightedTarget.Reset(NCudaLib::TMirrorMapping(joinedTarget.size()));
        ctrsTarget.Weights.Reset(NCudaLib::TMirrorMapping(joinedTarget.size()));

        ctrsTarget.LearnSlice = TSlice(0, dataProvider.GetSampleCount());
        ctrsTarget.TestSlice = TSlice(dataProvider.GetSampleCount(), joinedTarget.size());

        TVector<float> ctrWeights;
        ctrWeights.resize(joinedTarget.size(), 1.0f);

        TVector<float> ctrWeightedTargets(joinedTarget.begin(), joinedTarget.end());

        double totalWeight = 0;
        for (ui32 i = (ui32)ctrsTarget.LearnSlice.Right; i < ctrWeights.size(); ++i) {
            ctrWeights[i] = 0;
        }

        for (ui32 i = 0; i < ctrWeightedTargets.size(); ++i) {
            ctrWeightedTargets[i] *= ctrWeights[i];
            totalWeight += ctrWeights[i];
        }

        ctrsTarget.TotalWeight = (float)totalWeight;
        ctrsTarget.WeightedTarget.Write(ctrWeightedTargets);
        ctrsTarget.Weights.Write(ctrWeights);

        CB_ENSURE(ctrsTarget.IsTrivialWeights());
        return ctrsTargetPtr;
    }

    inline TVector<ui32> GetLearnFeatureIds(TBinarizedFeaturesManager& featuresManager) {
        TSet<ui32> featureIdsSet;
        auto ctrTypes = featuresManager.GetKnownSimpleCtrTypes();

        for (auto floatFeature : featuresManager.GetFloatFeatureIds()) {
            if (featuresManager.GetBinCount(floatFeature) > 1) {
                featureIdsSet.insert(floatFeature);
            }
        }
        for (auto catFeature : featuresManager.GetCatFeatureIds()) {
            if (featuresManager.UseForOneHotEncoding(catFeature)) {
                if (featuresManager.GetBinCount(catFeature) > 1) {
                    featureIdsSet.insert(catFeature);
                }
            }

            if (featuresManager.UseForCtr(catFeature)) {
                for (auto& ctr : ctrTypes) {
                    const auto simpleCtrsForType = featuresManager.CreateSimpleCtrsForType(catFeature,
                                                                                           ctr);
                    for (auto ctrFeatureId : simpleCtrsForType) {
                        featureIdsSet.insert(ctrFeatureId);
                    }
                }
            }
        }
        TSet<ui32> combinationCtrIds;

        for (auto& ctr : ctrTypes) {
            auto combinationCtrs = featuresManager.CreateCombinationCtrForType(ctr);
            for (auto ctrFeatureId : combinationCtrs) {
                TFeatureTensor tensor = featuresManager.GetCtr(ctrFeatureId).FeatureTensor;
                bool hasUnknownFeatures = false;
                CB_ENSURE(tensor.GetSplits().size() == 0);

                for (auto featureId : tensor.GetCatFeatures()) {
                    if (!featureIdsSet.has(featureId)) {
                        hasUnknownFeatures = true;
                        break;
                    }
                }
                for (auto binarySplit : tensor.GetSplits()) {
                    if (!featureIdsSet.has(binarySplit.FeatureId)) {
                        hasUnknownFeatures = true;
                        break;
                    }
                }
                if (!hasUnknownFeatures) {
                    combinationCtrIds.insert(ctrFeatureId);
                }
            }
        }
        featureIdsSet.insert(combinationCtrIds.begin(), combinationCtrIds.end());
        return TVector<ui32>(featureIdsSet.begin(), featureIdsSet.end());
    }

    //filter features and write float and one-hotones for selected dataSetId
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
                          "Error: unconsistent borders for feature #" << feature);

                const ui32 binCount = featuresHolder.BinCount();
                IndexBuilder.Write(DataSetId,
                                   feature,
                                   binCount,
                                   featuresHolder.ExtractValues<ui8>());

            } else {
                CB_ENSURE(featureStorage.GetType() == EFeatureValuesType::Float);

                const auto& holder = dynamic_cast<const TFloatValuesHolder&>(featureStorage);
                const auto& borders = FeaturesManager.GetBorders(feature);
                const ENanMode nanMode = FeaturesManager.GetOrCreateNanMode(holder);

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
                               /*hack for using only learn catFeatures as one-hot splits. test set storage should now about unseen catFeatures from learn. binCount is totalUniqueValues (learn + test)*/
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

}
