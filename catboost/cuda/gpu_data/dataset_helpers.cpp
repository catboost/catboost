#include "dataset_helpers.h"
#include "feature_layout_doc_parallel.h"
#include "feature_layout_feature_parallel.h"
#include <util/generic/maybe.h>

THolder<NCatboostCuda::TCtrTargets<NCudaLib::TMirrorMapping>> NCatboostCuda::BuildCtrTarget(const NCatboostCuda::TBinarizedFeaturesManager& featuresManager,
                                                                                            const NCB::TTrainingDataProvider& dataProvider,
                                                                                            const NCB::TTrainingDataProvider* test) {
    TVector<float> joinedTarget = Join((*dataProvider.TargetData->GetTarget())[0],
                                       test ? MakeMaybe((*test->TargetData->GetTarget())[0]) : Nothing()); // espetrov: fix for multi-target + cat features

    THolder<TCtrTargets<NCudaLib::TMirrorMapping>> ctrsTargetPtr;
    ctrsTargetPtr = MakeHolder<TCtrTargets<NCudaLib::TMirrorMapping>>();
    auto& ctrsTarget = *ctrsTargetPtr;
    ctrsTarget.BinarizedTarget = BuildBinarizedTarget(featuresManager,
                                                      joinedTarget);

    ctrsTarget.WeightedTarget.Reset(NCudaLib::TMirrorMapping(joinedTarget.size()));
    ctrsTarget.Weights.Reset(NCudaLib::TMirrorMapping(joinedTarget.size()));

    ctrsTarget.LearnSlice = TSlice(0, dataProvider.GetObjectCount());
    ctrsTarget.TestSlice = TSlice(dataProvider.GetObjectCount(), joinedTarget.size());

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

    if (!dataProvider.ObjectsGrouping->IsTrivial() && featuresManager.GetCatFeatureOptions().CtrHistoryUnit == ECtrHistoryUnit::Group) {
        const ui64 groupCountLearn = dataProvider.ObjectsGrouping->GetGroupCount();
        TVector<ui32> groupIds;
        groupIds.reserve(joinedTarget.size());

        for (ui32 groupId = 0; groupId < groupCountLearn; ++groupId) {
            ui32 groupSize = dataProvider.ObjectsGrouping->GetGroup(groupId).GetSize();
            for (ui32 j  = 0; j < groupSize; ++j) {
                groupIds.push_back(groupId);
            }
        }
        const ui64 groupCountTest = test ? test->ObjectsGrouping->GetGroupCount() : 0;

        for (ui32 groupId = 0; groupId < groupCountTest; ++groupId) {
            ui32 groupSize = test->ObjectsGrouping->GetGroup(groupId).GetSize();
            for (ui32 j = 0; j < groupSize; ++j) {
                groupIds.push_back(groupId + groupCountLearn);
            }
        }


        auto tmp = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(groupIds.size()));
        tmp.Write(groupIds);
        ctrsTarget.GroupIds = tmp.ConstCopyView();
    }
    return ctrsTargetPtr;
}

TVector<ui32> NCatboostCuda::GetLearnFeatureIds(NCatboostCuda::TBinarizedFeaturesManager& featuresManager) {
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
                if (!featureIdsSet.contains(featureId)) {
                    hasUnknownFeatures = true;
                    break;
                }
            }
            for (auto binarySplit : tensor.GetSplits()) {
                if (!featureIdsSet.contains(binarySplit.FeatureId)) {
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

    auto estimatedFeatures = featuresManager.GetEstimatedFeatureIds();
    featureIdsSet.insert(estimatedFeatures.begin(), estimatedFeatures.end());

    auto featureBundleIds = featuresManager.GetExclusiveFeatureBundleIds();
    featureIdsSet.insert(featureBundleIds.begin(), featureBundleIds.end());

    return TVector<ui32>(featureIdsSet.begin(), featureIdsSet.end());
}

namespace NCatboostCuda {
    TMirrorBuffer<ui8> BuildBinarizedTarget(const TBinarizedFeaturesManager& featuresManager, const TVector<float>& targets) {
        TVector<ui8> binarizedTarget;
        if (featuresManager.HasTargetBinarization()) {
            auto& borders = featuresManager.GetTargetBorders();
            binarizedTarget = NCB::BinarizeLine<ui8>(targets,
                                                     ENanMode::Forbidden,
                                                     borders);
        } else {
            binarizedTarget.resize(targets.size(), 0);
        }

        TMirrorBuffer<ui8> binarizedTargetGpu = TMirrorBuffer<ui8>::Create(NCudaLib::TMirrorMapping(binarizedTarget.size()));
        binarizedTargetGpu.Write(binarizedTarget);
        return binarizedTargetGpu;
    }

    void SplitByPermutationDependence(const TBinarizedFeaturesManager& featuresManager, const TVector<ui32>& features,
                                      const ui32 permutationCount, TVector<ui32>* permutationIndependent,
                                      TVector<ui32>* permutationDependent) {
        if (permutationCount == 1) {
            //            shortcut
            (*permutationIndependent) = features;
            return;
        }
        permutationDependent->clear();
        permutationIndependent->clear();
        for (const auto& feature : features) {
            const bool permutationDependentCtr = featuresManager.IsCtr(feature) && featuresManager.IsPermutationDependent(featuresManager.GetCtr(feature));
            const bool onlineEstimatedFeature = featuresManager.IsEstimatedFeature(feature) && featuresManager.GetEstimatedFeature(feature).EstimatorId.IsOnline;

            const bool needPermutationFlag = permutationDependentCtr || onlineEstimatedFeature;
            if (needPermutationFlag) {
                permutationDependent->push_back(feature);
            } else {
                permutationIndependent->push_back(feature);
            }
        }
    }

    template class TFloatAndOneHotFeaturesWriter<TFeatureParallelLayout>;
    template class TFloatAndOneHotFeaturesWriter<TDocParallelLayout>;

    template class TCtrsWriter<TFeatureParallelLayout>;
    template class TCtrsWriter<TDocParallelLayout>;

    template class TEstimatedFeaturesWriter<TFeatureParallelLayout>;
    template class TEstimatedFeaturesWriter<TDocParallelLayout>;


}
