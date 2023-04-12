#include "model_converter.h"

#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/algo/projection.h>
#include <catboost/private/libs/algo/split.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model_build_helper.h>

#include <util/generic/cast.h>

#include <limits>

using namespace NCB;

TVector<TTargetClassifier> NCatboostCuda::CreateTargetClassifiers(const NCatboostCuda::TBinarizedFeaturesManager& featuresManager) {
    TTargetClassifier targetClassifier(featuresManager.GetTargetBorders(), 0);
    TVector<TTargetClassifier> classifiers;
    classifiers.resize(1, targetClassifier);
    return classifiers;
}

namespace NCatboostCuda {

TModelConverter::TModelConverter(
    const TBinarizedFeaturesManager& manager,
    const TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    const TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap,
    const TClassificationTargetHelper& targetHelper)
    : FeaturesManager(manager)
      , QuantizedFeaturesInfo(quantizedFeaturesInfo)
      , FeaturesLayout(*quantizedFeaturesInfo->GetFeaturesLayout())
      , CatFeatureBinToHashIndex(perfectHashedToHashedCatValuesMap)
      , TargetHelper(targetHelper) {
    FloatFeaturesNanMode.resize(FeaturesLayout.GetFloatFeatureCount(), ENanMode::Forbidden);

    FeaturesLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
        [&](const TFloatFeatureIdx floatFeatureIdx) {
            FloatFeaturesNanMode[*floatFeatureIdx] = QuantizedFeaturesInfo->GetNanMode(floatFeatureIdx);
        });
}

inline bool HasEstimatedFeatures(const TBinarizedFeaturesManager& manager, TConstArrayRef<TBinarySplit> splits) {
    for (const auto& split : splits) {
        if (manager.IsEstimatedFeature(split.FeatureId)) {
            return true;
        }
    }
    return false;
}

inline bool HasEstimatedFeature(const TBinarizedFeaturesManager& manager, const TObliviousTreeModel& ot) {
    return HasEstimatedFeatures(manager, ot.GetStructure().Splits);
}

inline bool HasEstimatedFeature(const TBinarizedFeaturesManager& manager, const TNonSymmetricTree& tree) {
    for (const auto& split : tree.GetStructure().GetNodes()) {
        if (manager.IsEstimatedFeature(split.FeatureId)) {
            return true;
        }
    }
    return false;
}

template <class TModel>
inline bool HasEstimatedFeature(const TBinarizedFeaturesManager& manager, const TAdditiveModel<TModel>& model) {
    for (const auto& tree : model.WeakModels) {
        if (HasEstimatedFeature(manager, tree)) {
            return true;
        }
    }
    return false;
}

TFullModel TModelConverter::Convert(
    const TAdditiveModel<TObliviousTreeModel>& src,
    THashMap<TFeatureCombination, TProjection>* featureCombinationToProjection) const {
    TFullModel coreModel;
    coreModel.ModelInfo["params"] = "{}"; //will be overriden with correct params later

    ui32 cpuApproxDim = 1;

    if (TargetHelper.IsInitialized()) {
        coreModel.ModelInfo["class_params"] = TargetHelper.Serialize();
        if (TargetHelper.IsMultiClass()) {
            cpuApproxDim = SafeIntegerCast<ui32>(TargetHelper.GetApproxDimension());
        }
    }

    if (src.Bias) {
        cpuApproxDim = src.Bias->size();
    }

    TVector<TFloatFeature> floatFeatures = CreateFloatFeatures(
        *QuantizedFeaturesInfo->GetFeaturesLayout(), // it's ok to get from QuantizedFeaturesInfo
        *QuantizedFeaturesInfo);
    TVector<TCatFeature> catFeatures = CreateCatFeatures(*QuantizedFeaturesInfo->GetFeaturesLayout());
    TVector<TTextFeature> textFeatures = CreateTextFeatures(*QuantizedFeaturesInfo->GetFeaturesLayout());
    TVector<TEmbeddingFeature> embeddingFeatures = CreateEmbeddingFeatures(*QuantizedFeaturesInfo->GetFeaturesLayout());

    TObliviousTreeBuilder obliviousTreeBuilder(
        floatFeatures,
        catFeatures,
        textFeatures,
        embeddingFeatures,
        cpuApproxDim);

    for (ui32 i = 0; i < src.Size(); ++i) {
        const TObliviousTreeModel& model = src.GetWeakModel(i);
        const ui32 outputDim = model.OutputDim();
        TVector<TVector<double>> leafValues(cpuApproxDim);
        TVector<double> leafWeights;

        const auto& values = model.GetValues();
        const auto& weights = model.GetWeights();

        leafWeights.resize(weights.size());
        for (ui32 leaf = 0; leaf < weights.size(); ++leaf) {
            leafWeights[leaf] = weights[leaf];
        }

        for (ui32 dim = 0; dim < cpuApproxDim; ++dim) {
            leafValues[dim].resize(model.BinCount());
            if (dim < outputDim) {
                for (ui32 leaf = 0; leaf < model.BinCount(); ++leaf) {
                    const double val = values[outputDim * leaf + dim];
                    leafValues[dim][leaf] = val;
                }
            }
        }

        const auto& structure = model.GetStructure();
        auto treeStructure = ConvertSplits(structure.Splits, featureCombinationToProjection);
        obliviousTreeBuilder.AddTree(treeStructure, leafValues, leafWeights);
    }

    obliviousTreeBuilder.Build(coreModel.ModelTrees.GetMutable());
    TVector<double> bias;
    if (src.Bias) {
        bias = *src.Bias;
    }
    coreModel.SetScaleAndBias({1.0, bias});
    coreModel.UpdateDynamicData();
    return coreModel;
}

    TFullModel TModelConverter::Convert(
        const TAdditiveModel<TNonSymmetricTree>& src,
        THashMap<TFeatureCombination, TProjection>* featureCombinationToProjection) const {
        TFullModel coreModel;
        coreModel.ModelInfo["params"] = "{}"; //will be overriden with correct params later
        ui32 cpuApproxDim = 1;

        if (TargetHelper.IsInitialized()) {
            coreModel.ModelInfo["class_params"] = TargetHelper.Serialize();
            cpuApproxDim = SafeIntegerCast<ui32>(TargetHelper.GetApproxDimension());
        }

        if (src.Bias) {
            cpuApproxDim = src.Bias->size();
        }

        TVector<TFloatFeature> floatFeatures = CreateFloatFeatures(
            *QuantizedFeaturesInfo->GetFeaturesLayout(), // it's ok to get from QuantizedFeaturesInfo
            *QuantizedFeaturesInfo);
        TVector<TCatFeature> catFeatures = CreateCatFeatures(*QuantizedFeaturesInfo->GetFeaturesLayout());
        TVector<TTextFeature> textFeatures = CreateTextFeatures(*QuantizedFeaturesInfo->GetFeaturesLayout());
        TVector<TEmbeddingFeature> embeddingFeatures = CreateEmbeddingFeatures(*QuantizedFeaturesInfo->GetFeaturesLayout());


        TNonSymmetricTreeModelBuilder treeBuilder(
            floatFeatures,
            catFeatures,
            textFeatures,
            embeddingFeatures,
            cpuApproxDim);

        for (ui32 treeId = 0; treeId < src.Size(); ++treeId) {
            const TNonSymmetricTree& tree = src.GetWeakModel(treeId);
            THolder<TNonSymmetricTreeNode> treeHeadHolder = MakeHolder<TNonSymmetricTreeNode>();
            tree.VisitLeavesAndWeights([&](
                const TLeafPath& leafPath,
                TConstArrayRef<float> pathValues,
                double weight) {
                auto pathStructure = ConvertSplits(leafPath.Splits, featureCombinationToProjection);
                TNonSymmetricTreeNode* currentNode = treeHeadHolder.Get();
                for (size_t i = 0; i < pathStructure.size(); ++i) {
                    if (!currentNode->SplitCondition) {
                        currentNode->SplitCondition = pathStructure[i];
                    } else {
                        CB_ENSURE_INTERNAL(
                            currentNode->SplitCondition == pathStructure[i],
                            "Error while converting non symmetric trees from GPU format to CPU"
                        );
                    }
                    if (leafPath.Directions[i] == ESplitValue::Zero) {
                        if (!currentNode->Left) {
                            currentNode->Left = MakeHolder<TNonSymmetricTreeNode>();
                        }
                        currentNode = currentNode->Left.Get();
                    } else {
                        if (!currentNode->Right) {
                            currentNode->Right = MakeHolder<TNonSymmetricTreeNode>();
                        }
                        currentNode = currentNode->Right.Get();
                    }
                }
                auto vals = TVector<double>(pathValues.begin(), pathValues.end());
                CB_ENSURE(vals.size() <= cpuApproxDim, "Error: this is a bug with dimensions, contact catboost team");
                //GPU in multiclass use classCount - 1 dimensions for learning
                vals.resize(cpuApproxDim);

                currentNode->Value = vals;

                currentNode->NodeWeight = weight;
            });

            treeBuilder.AddTree(std::move(treeHeadHolder));
        }

        treeBuilder.Build(coreModel.ModelTrees.GetMutable());
        TVector<double> bias;
        if (src.Bias) {
            bias = *src.Bias;
        }
        coreModel.SetScaleAndBias({1.0, bias});
        coreModel.UpdateDynamicData();
        return coreModel;
    }

    TModelSplit TModelConverter::CreateFloatSplit(const TBinarySplit& split) const {
        CB_ENSURE(
            FeaturesManager.IsFloat(split.FeatureId),
            "Feature with id " << split.FeatureId << " is not FloatFeature"
        );

        TModelSplit modelSplit;
        modelSplit.Type = ESplitType::FloatFeature;
        auto dataProviderId = FeaturesManager.GetDataProviderId(split.FeatureId);
        auto remapId = FeaturesLayout.GetInternalFeatureIdx<EFeatureType::Float>(dataProviderId);

        float border = FeaturesManager.GetBorders(split.FeatureId).at(split.BinIdx);
        modelSplit.FloatFeature = TFloatSplit{(int) *remapId, border};
        return modelSplit;
    }

    TModelSplit TModelConverter::CreateEstimatedFeatureSplit(const TBinarySplit& split) const {
        CB_ENSURE(
            FeaturesManager.IsEstimatedFeature(split.FeatureId),
            "Feature with id " << split.FeatureId << " is not EstimatedFeature"
        );

        TModelSplit modelSplit;
        modelSplit.Type = ESplitType::EstimatedFeature;

        const auto& featureBorders = FeaturesManager.GetBorders(split.FeatureId);
        const float border = featureBorders.at(split.BinIdx);

        NCB::TEstimatedFeatureId estimatedFeature = FeaturesManager.GetEstimatedFeature(split.FeatureId);
        TFeatureEstimatorsPtr featureEstimators = FeaturesManager.GetFeatureEstimators();
        const TGuid& estimatorGuid = featureEstimators->GetEstimatorGuid(estimatedFeature.EstimatorId);

        modelSplit.EstimatedFeature = TEstimatedFeatureSplit(TModelEstimatedFeature{
            SafeIntegerCast<int>(featureEstimators->GetEstimatorSourceFeatureIdx(estimatorGuid).TextFeatureId),
            estimatorGuid,
            SafeIntegerCast<int>(estimatedFeature.LocalFeatureId),
            FeatureTypeToEstimatedSourceFeatureType(featureEstimators->GetEstimatorSourceType(estimatorGuid))},
            border
        );
        return modelSplit;
    }

    TModelSplit TModelConverter::CreateOneHotSplit(const TBinarySplit& split) const {
        CB_ENSURE(
            FeaturesManager.IsCat(split.FeatureId),
            "Feature with id " << split.FeatureId << " is not CategoricalFeature"
        );

        TModelSplit modelSplit;
        modelSplit.Type = ESplitType::OneHotFeature;
        auto dataProviderId = FeaturesManager.GetDataProviderId(split.FeatureId);
        auto remapId = FeaturesLayout.GetInternalFeatureIdx<EFeatureType::Categorical>(dataProviderId);

        CB_ENSURE(CatFeatureBinToHashIndex[*remapId].size(),
                  TStringBuilder() << "Error: no catFeature perfect hash for feature " << dataProviderId);
        CB_ENSURE(split.BinIdx < CatFeatureBinToHashIndex[*remapId].size(),
                  TStringBuilder() << "Error: no hash for feature " << split.FeatureId << " " << split.BinIdx);
        const int hash = CatFeatureBinToHashIndex[*remapId][split.BinIdx];
        modelSplit.OneHotFeature = TOneHotSplit(*remapId,
                                                hash);
        return modelSplit;
    }

    ui32 TModelConverter::GetRemappedIndex(ui32 featureId) const {
        CB_ENSURE(FeaturesManager.IsCat(featureId) || FeaturesManager.IsFloat(featureId));
        ui32 dataProviderId = FeaturesManager.GetDataProviderId(featureId);
        return FeaturesLayout.GetInternalFeatureIdx(dataProviderId);
    }

    void TModelConverter::ExtractProjection(
        const TCtr& ctr,
        TFeatureCombination* dstFeatureCombination,
        TProjection* dstProjection) const {
        TFeatureCombination featureCombination;

        TVector<TBinFeature> binFeatures;
        TVector<TOneHotSplit> oneHotFeatures;

        for (auto split : ctr.FeatureTensor.GetSplits()) {
            if (FeaturesManager.IsFloat(split.FeatureId)) {
                auto floatSplit = CreateFloatSplit(split);
                featureCombination.BinFeatures.push_back(floatSplit.FloatFeature);
                binFeatures.push_back(TBinFeature(floatSplit.FloatFeature.FloatFeature, split.BinIdx));
            } else if (FeaturesManager.IsCat(split.FeatureId)) {
                auto oneHotSplit = CreateOneHotSplit(split);
                featureCombination.OneHotFeatures.push_back(oneHotSplit.OneHotFeature);
                oneHotFeatures.push_back(TOneHotSplit(oneHotSplit.OneHotFeature.CatFeatureIdx, split.BinIdx));
            } else {
                CB_ENSURE(false, "Error: unknown split type");
            }
        }
        for (auto catFeature : ctr.FeatureTensor.GetCatFeatures()) {
            featureCombination.CatFeatures.push_back(GetRemappedIndex(catFeature));
        }

        //just for more more safety
        Sort(featureCombination.BinFeatures);
        Sort(featureCombination.CatFeatures);
        Sort(featureCombination.OneHotFeatures);

        Sort(binFeatures);
        Sort(oneHotFeatures);
        dstProjection->BinFeatures = std::move(binFeatures);
        dstProjection->OneHotFeatures = std::move(oneHotFeatures);
        dstProjection->CatFeatures = featureCombination.CatFeatures;

        *dstFeatureCombination = std::move(featureCombination);
    }

    TModelSplit TModelConverter::CreateCtrSplit(
        const TBinarySplit& split,
        THashMap<TFeatureCombination, TProjection>* featureCombinationToProjection) const {
        TModelSplit modelSplit;
        CB_ENSURE(
            FeaturesManager.IsCtr(split.FeatureId),
            "Feature with id " << split.FeatureId << " is not CtrFeature"
        );
        const auto& ctr = FeaturesManager.GetCtr(split.FeatureId);
        auto& borders = FeaturesManager.GetBorders(split.FeatureId);
        CB_ENSURE(split.BinIdx < borders.size(), "Split " << split.BinIdx << ", borders: " << borders.size());

        modelSplit.Type = ESplitType::OnlineCtr;
        modelSplit.OnlineCtr.Border = borders[split.BinIdx];

        TModelCtr& modelCtr = modelSplit.OnlineCtr.Ctr;

        TProjection projection;
        ExtractProjection(ctr, &modelCtr.Base.Projection, &projection);
        featureCombinationToProjection->insert({modelCtr.Base.Projection, std::move(projection)});

        modelCtr.Base.CtrType = ctr.Configuration.Type;
        modelCtr.Base.TargetBorderClassifierIdx = 0;

        const auto& config = ctr.Configuration;
        modelCtr.TargetBorderIdx = config.ParamId;
        modelCtr.PriorNum = GetNumeratorShift(config);
        modelCtr.PriorDenom = GetDenumeratorShift(config);

        modelSplit.OnlineCtr.Canonize();

        return modelSplit;
    }

    TVector<TModelSplit> TModelConverter::ConvertSplits(
        const TVector<TBinarySplit>& splits,
        THashMap<TFeatureCombination, TProjection>* featureCombinationToProjection
    ) const {
        TVector<TModelSplit> structure3;
        for (auto split : splits) {
            TModelSplit modelSplit;
            if (FeaturesManager.IsFloat(split.FeatureId)) {
                modelSplit = CreateFloatSplit(split);
            } else if (FeaturesManager.IsCat(split.FeatureId)) {
                modelSplit = CreateOneHotSplit(split);
            } else if (FeaturesManager.IsCtr(split.FeatureId)) {
                modelSplit = CreateCtrSplit(split, featureCombinationToProjection);
            } else if (FeaturesManager.IsEstimatedFeature(split.FeatureId)) {
                modelSplit = CreateEstimatedFeatureSplit(split);
            } else {
                CB_ENSURE_INTERNAL(false, "Unknown split type");
            }
            structure3.push_back(modelSplit);
        }
        return structure3;
    }
}

