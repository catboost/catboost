#include "ctr_helper.h"

#include "target_classifier.h"

#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/model/target_classifier.h>
#include <catboost/private/libs/options/cat_feature_options.h>
#include <catboost/private/libs/options/defaults_helper.h>

using BinarizationOptionAndTarget = std::pair<NCatboostOptions::TBinarizationOptions, ui32>;

template <>
struct THash<BinarizationOptionAndTarget> {
    inline size_t operator()(const BinarizationOptionAndTarget& value) const {
        return static_cast<ui64>(value.first.GetHash() << 32 | value.second);
    }
};

inline TVector<TCtrInfo> MakeCtrInfo(
    const NCatboostOptions::TCtrDescription& description,
    THashMap<BinarizationOptionAndTarget, ui32>* targetClassifiers,
    size_t targetCount) {

    TVector<TCtrInfo> ctrsInfo;
    for (size_t target = 0; target < targetCount; ++target) {
        TCtrInfo ctrInfo;
        ctrInfo.Type = description.Type;
        ctrInfo.BorderCount = description.GetCtrBinarization().BorderCount;
        CB_ENSURE(
            description.GetCtrBinarization().BorderSelectionType.Get() == EBorderSelectionType::Uniform,
            "Error: CPU supports only uniform binarization for CTRS");

        if (NeedTargetClassifier(ctrInfo.Type)) {
            const auto& targetBinarization = description.TargetBinarization.Get();
            if (!targetClassifiers->contains({targetBinarization, target})) {
                ui32 targetClassifierId = targetClassifiers->size();
                (*targetClassifiers)[BinarizationOptionAndTarget(targetBinarization, target)] = targetClassifierId;
            }
            ctrInfo.TargetClassifierIdx = (*targetClassifiers)[BinarizationOptionAndTarget(targetBinarization, target)];
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

        ctrsInfo.push_back(ctrInfo);
    }
    return ctrsInfo;
}

void TCtrHelper::InitCtrHelper(
    const NCatboostOptions::TCatFeatureParams& catFeatureParams,
    const NCB::TFeaturesLayout& layout,
    NCB::TMaybeData<TConstArrayRef<TConstArrayRef<float>>> targets,
    ELossFunction loss,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    bool allowConstLabel) {

    using TCtrsDescription = TVector<NCatboostOptions::TCtrDescription>;
    const TCtrsDescription& treeCtrs = catFeatureParams.CombinationCtrs;
    const TCtrsDescription& simpleCtrs = catFeatureParams.SimpleCtrs;
    const TMap<ui32, TCtrsDescription>& perFeatureCtrs = catFeatureParams.PerFeatureCtrs;

    THashMap<BinarizationOptionAndTarget, ui32> targetClassifierIds;
    {
        NCatboostOptions::TBinarizationOptions fakeCounterClassifier;
        fakeCounterClassifier.BorderCount = 0;
        targetClassifierIds[BinarizationOptionAndTarget(fakeCounterClassifier, 0)] = 0;
    }

    size_t targetCount = targets ? targets->size() : 0;
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
            auto ctrsInfo = MakeCtrInfo(description, &targetClassifierIds, targetCount);
            for (auto& ctrInfo: ctrsInfo) {
                PerFeatureCtrs[featureIdx].push_back(ctrInfo);
            }
        }
    }

    for (const auto& simpleCtr : simpleCtrs) {
        auto ctrsInfo = MakeCtrInfo(simpleCtr, &targetClassifierIds, targetCount);
        for (auto& ctrInfo: ctrsInfo) {
            SimpleCtrs.push_back(ctrInfo);
        }
    }

    for (const auto& treeCtr : treeCtrs) {
        auto ctrsInfo = MakeCtrInfo(treeCtr, &targetClassifierIds, targetCount);
        for (auto& ctrInfo: ctrsInfo) {
            TreeCtrs.push_back(ctrInfo);
        }
    }

    TargetClassifiers.resize(targetClassifierIds.size());

    for (const auto& targetClassifier : targetClassifierIds) {
        ui32 id = targetClassifier.second;
        const NCatboostOptions::TBinarizationOptions& binarizationOption = targetClassifier.first.first;
        auto targetId = targetClassifier.first.second;

        if (binarizationOption.BorderCount.Get() == 0) {
            TargetClassifiers[id] = TTargetClassifier();
        } else {
            TargetClassifiers[id] = BuildTargetClassifier(
                (*targets)[targetId],
                loss,
                objectiveDescriptor,
                binarizationOption.BorderCount,
                binarizationOption.BorderSelectionType,
                allowConstLabel,
                targetId);
        }
    }
}
