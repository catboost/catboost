#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/split.h>

#include <util/generic/array_ref.h>
#include <util/generic/set.h>

#include <algorithm>
#include <cmath>

namespace NCB {

struct TMetalTreeCtrTensorPack {
    TFeatureCombination Base;
    TSet<TFeatureCombination> Tensors;
};

// Model-index counterpart of CUDA TTreeCtrDataSetsHelper's tensor scheduler.
// The caller supplies only category model indices eligible for tree CTRs (the
// same OnAll/one_hot_max_size policy as the categorical dataset builder).
// Simple CTRs remain in the caller's static feature dataset; this class emits
// only tensors introduced by selected splits in the current tree.
// Source rules: cuda/methods/tree_ctrs.cpp AddSplit, UpdatePureTreeCtrTensor,
// and AddDataSetPacks; cuda/data/feature.h TFeatureTensor::GetComplexity;
// cuda/data/binarizations_manager.{h,cpp} UseAsBaseTensorForTreeCtr/UseForTreeCtr.
class TMetalTreeCtrTensorScheduler {
public:
    TMetalTreeCtrTensorScheduler(TConstArrayRef<int> eligibleCatFeatures,
                                ui32 maxTensorComplexity,
                                TConstArrayRef<int> estimatedFloatFeatures = {})
        : MaxTensorComplexity(maxTensorComplexity)
    {
        CB_ENSURE(MaxTensorComplexity > 0, "Tree CTR max tensor complexity must be positive");
        for (int feature : eligibleCatFeatures) {
            CB_ENSURE(feature >= 0, "Tree CTR category model indices must be nonnegative");
            EligibleCatFeatures.insert(feature);
        }
        for (int feature : estimatedFloatFeatures) {
            CB_ENSURE(feature >= 0, "Estimated float model indices must be nonnegative");
            EstimatedFloatFeatures.insert(feature);
        }
    }

    void BeginTree() {
        CurrentTensor.Clear();
        PureTreeTensor.Clear();
        PureTreeHasEstimatedFeature = false;
        PersistentTensors.clear();
        PersistentBases.clear();
        PureTreeTensors.clear();
        ActiveTensors.clear();
        HasPureTreeTensor = false;
    }

    void AddSplit(const TModelSplit& split) {
        if (EligibleCatFeatures.empty() || MaxTensorComplexity <= 1) {
            return;
        }
        if (split.Type == ESplitType::OnlineCtr) {
            // A chosen CTR contributes its underlying category/predicate
            // projection, not a binary predicate on the CTR value itself.
            TFeatureCombination merged = CurrentTensor;
            Merge(split.OnlineCtr.Ctr.Base.Projection, &merged);
            // CUDA commits CurrentTensor only when it remains a legal *base*:
            // its complexity must leave room for one additional category.
            if (merged == CurrentTensor || Complexity(merged) >= MaxTensorComplexity) {
                return;
            }
            CurrentTensor = std::move(merged);
            PersistentBases.insert(CurrentTensor);
            AddCrossedTensors(CurrentTensor, &PersistentTensors);
        } else {
            TFeatureCombination next = PureTreeTensor;
            bool hasEstimated = PureTreeHasEstimatedFeature;
            if (split.Type == ESplitType::FloatFeature) {
                next.BinFeatures.push_back(split.FloatFeature);
            } else if (split.Type == ESplitType::OneHotFeature) {
                next.OneHotFeatures.push_back(split.OneHotFeature);
            } else if (split.Type == ESplitType::EstimatedFeature) {
                // TFeatureCombination has no estimated-predicate component.
                // CUDA still accumulates it in the pure-tree tensor, making
                // every later pure-tree candidate ineligible. Remember that
                // state rather than silently retaining the preceding packs.
                hasEstimated = true;
            } else {
                CB_ENSURE(false, "Unsupported split type in the tree CTR tensor scheduler");
            }
            Normalize(&next);
            PureTreeTensor = std::move(next);
            PureTreeHasEstimatedFeature = hasEstimated;
            HasPureTreeTensor = true;
            // Ordinary splits replace only the previous pure-tree packs.
            // Previously admitted merged-CTR packs remain available this tree.
            PureTreeTensors.clear();
            if (!PureTreeHasEstimatedFeature) {
                AddCrossedTensors(PureTreeTensor, &PureTreeTensors);
            }
        }
        ActiveTensors = PersistentTensors;
        ActiveTensors.insert(PureTreeTensors.begin(), PureTreeTensors.end());
    }

    const TSet<TFeatureCombination>& GetActiveTensors() const {
        return ActiveTensors;
    }

    // Preserve the base that seeded CUDA's dataset visitor. Reconstructing
    // it by removing a category from a full projection is ambiguous. Several
    // physical CUDA packs of a base differ only in crossed categories; each
    // category carries the full common configuration set and policy mask.
    TVector<TMetalTreeCtrTensorPack> GetActiveTensorPacks() const {
        TVector<TMetalTreeCtrTensorPack> result;
        for (const auto& base : PersistentBases) {
            TMetalTreeCtrTensorPack pack;
            pack.Base = base;
            AddCrossedTensors(base, &pack.Tensors);
            if (!pack.Tensors.empty()) result.push_back(std::move(pack));
        }
        if (HasPureTreeTensor && !PureTreeHasEstimatedFeature) {
            TMetalTreeCtrTensorPack pack;
            pack.Base = PureTreeTensor;
            AddCrossedTensors(PureTreeTensor, &pack.Tensors);
            if (!pack.Tensors.empty()) result.push_back(std::move(pack));
        }
        return result;
    }

    static size_t Complexity(const TFeatureCombination& tensor) {
        // CUDA counts the entire binary-predicate part as one component,
        // regardless of how many float and one-hot splits it contains.
        return tensor.CatFeatures.size() +
            static_cast<size_t>(!tensor.BinFeatures.empty() || !tensor.OneHotFeatures.empty());
    }

private:
    template <class T>
    static void SortUnique(TVector<T>* values) {
        std::sort(values->begin(), values->end());
        values->erase(std::unique(values->begin(), values->end()), values->end());
    }

    static void Normalize(TFeatureCombination* tensor) {
        for (int feature : tensor->CatFeatures) {
            CB_ENSURE(feature >= 0, "Tree CTR category model indices must be nonnegative");
        }
        for (auto& split : tensor->BinFeatures) {
            CB_ENSURE(split.FloatFeature >= 0 && std::isfinite(split.Split),
                      "Tree CTR float predicates require a valid model index and finite border");
            split.Canonize();
        }
        for (const auto& split : tensor->OneHotFeatures) {
            CB_ENSURE(split.CatFeatureIdx >= 0, "Tree CTR one-hot model indices must be nonnegative");
        }
        SortUnique(&tensor->CatFeatures);
        SortUnique(&tensor->BinFeatures);
        SortUnique(&tensor->OneHotFeatures);
    }

    static void Merge(const TFeatureCombination& source, TFeatureCombination* destination) {
        destination->CatFeatures.insert(destination->CatFeatures.end(), source.CatFeatures.begin(), source.CatFeatures.end());
        destination->BinFeatures.insert(destination->BinFeatures.end(), source.BinFeatures.begin(), source.BinFeatures.end());
        destination->OneHotFeatures.insert(destination->OneHotFeatures.end(), source.OneHotFeatures.begin(), source.OneHotFeatures.end());
        Normalize(destination);
    }

    bool ContainsEstimatedPredicate(const TFeatureCombination& tensor) const {
        for (const auto& split : tensor.BinFeatures) {
            if (EstimatedFloatFeatures.contains(split.FloatFeature)) {
                return true;
            }
        }
        return false;
    }

    void AddCrossedTensors(const TFeatureCombination& base,
                           TSet<TFeatureCombination>* destination) const {
        if (ContainsEstimatedPredicate(base)) {
            return;
        }
        for (int feature : EligibleCatFeatures) {
            if (std::binary_search(base.CatFeatures.begin(), base.CatFeatures.end(), feature)) {
                continue;
            }
            TFeatureCombination candidate = base;
            candidate.CatFeatures.push_back(feature);
            SortUnique(&candidate.CatFeatures);
            if (Complexity(candidate) <= MaxTensorComplexity) {
                destination->insert(std::move(candidate));
            }
        }
    }

    ui32 MaxTensorComplexity;
    TSet<int> EligibleCatFeatures;
    TSet<int> EstimatedFloatFeatures;
    TFeatureCombination CurrentTensor;
    TFeatureCombination PureTreeTensor;
    bool PureTreeHasEstimatedFeature = false;
    bool HasPureTreeTensor = false;
    TSet<TFeatureCombination> PersistentBases;
    TSet<TFeatureCombination> PersistentTensors;
    TSet<TFeatureCombination> PureTreeTensors;
    TSet<TFeatureCombination> ActiveTensors;
};

} // namespace NCB
