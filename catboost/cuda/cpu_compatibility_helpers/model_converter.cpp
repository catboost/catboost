#include "model_converter.h"

#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/algo/projection.h>
#include <catboost/libs/algo/split.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model_build_helper.h>

#include <util/generic/cast.h>

#include <limits>

using namespace NCB;

TVector<TTargetClassifier> NCatboostCuda::CreateTargetClassifiers(const NCatboostCuda::TBinarizedFeaturesManager& featuresManager) {
    TTargetClassifier targetClassifier(featuresManager.GetTargetBorders());
    TVector<TTargetClassifier> classifiers;
    classifiers.resize(1, targetClassifier);
    return classifiers;
}

namespace NCatboostCuda {
    TModelConverter::TModelConverter(const TBinarizedFeaturesManager& manager,
                                     const TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
                                     const TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap,
                                     const TClassificationTargetHelper& targetHelper)
        : FeaturesManager(manager)
        , QuantizedFeaturesInfo(quantizedFeaturesInfo)
        , FeaturesLayout(*quantizedFeaturesInfo->GetFeaturesLayout())
        , CatFeatureBinToHashIndex(perfectHashedToHashedCatValuesMap)
        , TargetHelper(targetHelper)
    {
        Borders.resize(FeaturesLayout.GetFloatFeatureCount());
        FloatFeaturesNanMode.resize(FeaturesLayout.GetFloatFeatureCount(), ENanMode::Forbidden);

        FeaturesLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
            [&](const TFloatFeatureIdx floatFeatureIdx) {
                Borders[*floatFeatureIdx] = QuantizedFeaturesInfo->GetBorders(floatFeatureIdx);
                FloatFeaturesNanMode[*floatFeatureIdx] = QuantizedFeaturesInfo->GetNanMode(floatFeatureIdx);
            });
    }

    TFullModel TModelConverter::Convert(const TAdditiveModel<TObliviousTreeModel>& src,
                                        THashMap<TFeatureCombination, TProjection>* featureCombinationToProjection) const {
        TFullModel coreModel;
        coreModel.ModelInfo["params"] = "{}"; //will be overriden with correct params later

        ui32 cpuApproxDim = 1;

        if (TargetHelper.IsMultiClass()) {
            coreModel.ModelInfo["multiclass_params"] = TargetHelper.Serialize();
            cpuApproxDim = SafeIntegerCast<ui32>(TargetHelper.GetNumClasses());
        }

        TVector<TFloatFeature> floatFeatures = CreateFloatFeatures(*QuantizedFeaturesInfo);
        TVector<TCatFeature> catFeatures = CreateCatFeatures(*QuantizedFeaturesInfo);

        TObliviousTreeBuilder obliviousTreeBuilder(
            floatFeatures,
            catFeatures,
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
            auto treeStructure = ConvertStructure(structure, featureCombinationToProjection);
            obliviousTreeBuilder.AddTree(treeStructure, leafValues, leafWeights);
        }
        coreModel.ObliviousTrees = obliviousTreeBuilder.Build();
        return coreModel;
    }

    TModelSplit TModelConverter::CreateFloatSplit(const TBinarySplit& split) const {
        CB_ENSURE(FeaturesManager.IsFloat(split.FeatureId));

        TModelSplit modelSplit;
        modelSplit.Type = ESplitType::FloatFeature;
        auto dataProviderId = FeaturesManager.GetDataProviderId(split.FeatureId);
        auto remapId = FeaturesLayout.GetInternalFeatureIdx<EFeatureType::Float>(dataProviderId);

        float border = Borders.at(*remapId).at(split.BinIdx);
        modelSplit.FloatFeature = TFloatSplit{(int)*remapId, border};
        return modelSplit;
    }

    TModelSplit TModelConverter::CreateOneHotSplit(const TBinarySplit& split) const {
        CB_ENSURE(FeaturesManager.IsCat(split.FeatureId));

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

    void TModelConverter::ExtractProjection(const TCtr& ctr,
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

    TModelSplit TModelConverter::CreateCtrSplit(const TBinarySplit& split,
                                                THashMap<TFeatureCombination, TProjection>* featureCombinationToProjection) const {
        TModelSplit modelSplit;
        CB_ENSURE(FeaturesManager.IsCtr(split.FeatureId));
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
        modelCtr.Base.TargetBorderClassifierIdx = 0; // TODO(kirillovs): remove me

        const auto& config = ctr.Configuration;
        modelCtr.TargetBorderIdx = config.ParamId;
        modelCtr.PriorNum = GetNumeratorShift(config);
        modelCtr.PriorDenom = GetDenumeratorShift(config);

        return modelSplit;
    }

    TVector<TModelSplit> TModelConverter::ConvertStructure(const TObliviousTreeStructure& structure,
                                                           THashMap<TFeatureCombination, TProjection>* featureCombinationToProjection) const {
        TVector<TModelSplit> structure3;
        for (auto split : structure.Splits) {
            TModelSplit modelSplit;
            if (FeaturesManager.IsFloat(split.FeatureId)) {
                modelSplit = CreateFloatSplit(split);
            } else if (FeaturesManager.IsCat(split.FeatureId)) {
                modelSplit = CreateOneHotSplit(split);
            } else {
                modelSplit = CreateCtrSplit(split, featureCombinationToProjection);
            }
            structure3.push_back(modelSplit);
        }
        return structure3;
    }
}
