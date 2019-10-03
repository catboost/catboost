#include "ctr_helper.h"

#include "target_classifier.h"

#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/model/target_classifier.h>
#include <catboost/private/libs/options/cat_feature_options.h>
#include <catboost/private/libs/options/defaults_helper.h>


inline TCtrInfo MakeCtrInfo(
    const NCatboostOptions::TCtrDescription& description,
    THashMap<NCatboostOptions::TBinarizationOptions, ui32>* targetClassifiers) {

    TCtrInfo ctrInfo;
    ctrInfo.Type = description.Type;
    ctrInfo.BorderCount = description.GetCtrBinarization().BorderCount;
    CB_ENSURE(
        description.GetCtrBinarization().BorderSelectionType.Get() == EBorderSelectionType::Uniform,
        "Error: CPU supports only uniform binarization for CTRS");

    if (NeedTargetClassifier(ctrInfo.Type)) {
        const auto& targetBinarization = description.TargetBinarization.Get();
        if (!targetClassifiers->contains(targetBinarization)) {
            ui32 targetClassifierId = targetClassifiers->size();
            (*targetClassifiers)[targetBinarization] = targetClassifierId;
        }
        ctrInfo.TargetClassifierIdx = (*targetClassifiers)[targetBinarization];
    } else {
        ctrInfo.TargetClassifierIdx = 0;
    }
    for (const auto& prior : description.GetPriors()) {
        CB_ENSURE(
            prior.size() <= 2,
            "Error: too many prior parameters. Expect 1 or 2, got " << prior.size() << " for ctr type " << description.Type);
        CB_ENSURE(prior.size() != 0, "Error: no prior parameter found for ctr type " << description.Type);
        const auto num = prior[0];
        const auto denom = prior.size() > 1 ? prior[1] : 1.0;
        CB_ENSURE(denom == 1.0, "Error: CPU could use only 1 as denom for ctrs currently");
        ctrInfo.Priors.push_back(num);
    }
    return ctrInfo;
}

void TCtrHelper::InitCtrHelper(
    const NCatboostOptions::TCatFeatureParams& catFeatureParams,
    const NCB::TFeaturesLayout& layout,
    NCB::TMaybeData<TConstArrayRef<float>> target,
    ELossFunction loss,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    bool allowConstLabel) {

    using TCtrsDescription = TVector<NCatboostOptions::TCtrDescription>;
    const TCtrsDescription& treeCtrs = catFeatureParams.CombinationCtrs;
    const TCtrsDescription& simpleCtrs = catFeatureParams.SimpleCtrs;
    const TMap<ui32, TCtrsDescription>& perFeatureCtrs = catFeatureParams.PerFeatureCtrs;

    THashMap<NCatboostOptions::TBinarizationOptions, ui32> targetClassifierIds;
    {
        NCatboostOptions::TBinarizationOptions fakeCounterClassifier;
        fakeCounterClassifier.BorderCount = 0;
        targetClassifierIds[fakeCounterClassifier] = 0;
    }

    for (const auto& perFeatureCtr : perFeatureCtrs) {
        int feature = perFeatureCtr.first;
        const auto& descriptions = perFeatureCtr.second;

        CB_ENSURE(
            layout.IsCorrectExternalFeatureIdx(feature),
            "Feature " + ToString(feature) + " in per-feature-priors does not exist");
        CB_ENSURE(
            layout.GetExternalFeatureType(feature) == EFeatureType::Categorical,
            "Feature " + ToString(feature) + " in per-feature-priors is not categorical");
        int featureIdx = layout.GetInternalFeatureIdx(feature);
        CB_ENSURE(
            !PerFeatureCtrs.contains(featureIdx),
            "Error: duplicate per feature ctr descriptions (feature #" << feature << ")");
        for (auto description : descriptions) {
            PerFeatureCtrs[featureIdx].push_back(MakeCtrInfo(description, &targetClassifierIds));
        }
    }

    for (const auto& simpleCtr : simpleCtrs) {
        SimpleCtrs.push_back(MakeCtrInfo(simpleCtr, &targetClassifierIds));
    }

    for (const auto& treeCtr : treeCtrs) {
        TreeCtrs.push_back(MakeCtrInfo(treeCtr, &targetClassifierIds));
    }

    TargetClassifiers.resize(targetClassifierIds.size());

    for (const auto& targetClassifier : targetClassifierIds) {
        ui32 id = targetClassifier.second;
        const NCatboostOptions::TBinarizationOptions& binarizationOption = targetClassifier.first;
        if (binarizationOption.BorderCount.Get() == 0) {
            TargetClassifiers[id] = TTargetClassifier();
        } else {
            TargetClassifiers[id] = BuildTargetClassifier(
                *target,
                loss,
                objectiveDescriptor,
                binarizationOption.BorderCount,
                binarizationOption.BorderSelectionType,
                allowConstLabel);
        }
    }
}
