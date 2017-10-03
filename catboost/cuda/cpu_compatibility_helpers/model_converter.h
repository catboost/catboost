#pragma once

#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/data/cat_feature_binarization_helpers.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/target_classifier.h>

class TModelConverter {
public:
    TModelConverter(const TBinarizedFeaturesManager& manager,
                    const TDataProvider& dataProvider,
                    const TString& catFeatureBinarizationFilename)
        : FeaturesManager(manager)
        , DataProvider(dataProvider)
    {
        auto& allFeatures = dataProvider.GetFeatureNames();
        auto& catFeatureIds = dataProvider.GetCatFeatureIds();

        {
            for (ui32 featureId = 0; featureId < allFeatures.size(); ++featureId) {
                if (catFeatureIds.has(featureId)) {
                    CatFeaturesRemap[featureId] = static_cast<ui32>(CatFeaturesRemap.size());
                } else {
                    if (dataProvider.HasFeatureId(featureId)) {
                        Borders.push_back(dataProvider.GetBinarizedFloatFeatureById(featureId).GetBorders());
                    } else {
                        Borders.push_back(yvector<float>());
                    }
                    FloatFeaturesRemap[featureId] = static_cast<ui32>(FloatFeaturesRemap.size());
                }
            }
        }
        {
            yvector<ui32> catFeatureVec(catFeatureIds.begin(), catFeatureIds.end());
            CatFeatureBinToHashIndex = TCatFeatureBinarizationHelpers::MakeInverseCatFeatureIndex(catFeatureVec,
                                                                                                  catFeatureBinarizationFilename);
        }
    }

    TCoreModel Convert(const TAdditiveModel<TObliviousTreeModel>& src) const {
        const auto& featureNames = DataProvider.GetFeatureNames();

        TCoreModel coreModel;
        coreModel.Borders = Borders;
        coreModel.ModelInfo["params"] = "{}"; //TODO(noxoomo): something meaningful here
        coreModel.FeatureCount = featureNames.size();
        coreModel.CatFeatures = yvector<int>(DataProvider.GetCatFeatureIds().begin(),
                                             DataProvider.GetCatFeatureIds().end());
        coreModel.FeatureIds = featureNames;
        coreModel.TargetClassifiers = CreateTargetClassifiers();

        coreModel.LeafValues.resize(src.Size());
        for (ui32 i = 0; i < src.Size(); ++i) {
            coreModel.LeafValues[i].resize(1);
            const TObliviousTreeModel& model = src.GetWeakModel(i);
            auto& values = model.GetValues();

            coreModel.LeafValues[i][0].resize(values.size());
            for (ui32 leaf = 0; leaf < values.size(); ++leaf) {
                coreModel.LeafValues[i][0][leaf] = values[leaf];
            }

            const auto& structure = model.GetStructure();
            TTensorStructure3 treeStructure = ConvertStructure(structure);
            coreModel.TreeStruct.push_back(treeStructure);
        }
        return coreModel;
    }

private:
    inline TModelSplit CreateFloatSplit(const TBinarySplit& split) const {
        CB_ENSURE(FeaturesManager.IsFloat(split.FeatureId));

        TModelSplit modelSplit;
        modelSplit.Type = ESplitType::FloatFeature;
        auto dataProviderId = FeaturesManager.GetDataProviderId(split.FeatureId);
        auto remapId = FloatFeaturesRemap.at(dataProviderId);
        modelSplit.BinFeature = TBinFeature(remapId,
                                            split.BinIdx);
        return modelSplit;
    }

    inline TModelSplit CreateOneHotSplit(const TBinarySplit& split) const {
        CB_ENSURE(FeaturesManager.IsCat(split.FeatureId));

        TModelSplit modelSplit;
        modelSplit.Type = ESplitType::OneHotFeature;
        auto dataProviderId = FeaturesManager.GetDataProviderId(split.FeatureId);
        auto remapId = CatFeaturesRemap.at(dataProviderId);
        CB_ENSURE(CatFeatureBinToHashIndex[remapId].size(), TStringBuilder() << "Error: no catFeature perferct hash for feature " << dataProviderId);
        CB_ENSURE(split.BinIdx < CatFeatureBinToHashIndex[remapId].size(), TStringBuilder() << "Error: no gasg fir feature " << split.FeatureId << " " << split.BinIdx);
        const int hash = CatFeatureBinToHashIndex[remapId][split.BinIdx];
        modelSplit.OneHotFeature = TOneHotFeature(remapId,
                                                  hash);
        return modelSplit;
    }

    inline ui32 GetRemappedIndex(ui32 featureId) const {
        CB_ENSURE(FeaturesManager.IsCat(featureId) || FeaturesManager.IsFloat(featureId));
        ui32 dataProviderId = FeaturesManager.GetDataProviderId(featureId);
        if (FeaturesManager.IsFloat(featureId)) {
            return FloatFeaturesRemap.at(dataProviderId);
        } else {
            return CatFeaturesRemap.at(dataProviderId);
        }
    }

    TProjection ExtractProjection(const TCtr& ctr) const {
        TProjection projection;
        for (auto split : ctr.FeatureTensor.GetSplits()) {
            if (FeaturesManager.IsFloat(split.FeatureId)) {
                projection.AddBinFeature(CreateFloatSplit(split).BinFeature);
            } else if (FeaturesManager.IsCat(split.FeatureId)) {
                projection.AddOneHotFeature(CreateOneHotSplit(split).OneHotFeature);
            } else {
                CB_ENSURE(false, "Error: unknown split type");
            }
        }
        for (auto catFeature : ctr.FeatureTensor.GetCatFeatures()) {
            projection.AddCatFeature(GetRemappedIndex(catFeature));
        }
        return projection;
    }

    inline TModelSplit CreateCtrSplit(const TBinarySplit& split) const {
        TModelSplit modelSplit;
        CB_ENSURE(FeaturesManager.IsCtr(split.FeatureId));
        const auto& ctr = FeaturesManager.GetCtr(split.FeatureId);
        auto& borders = FeaturesManager.GetBorders(split.FeatureId);
        CB_ENSURE(split.BinIdx < borders.size());

        modelSplit.Type = ESplitType::OnlineCtr;
        modelSplit.OnlineCtr.Border = borders[split.BinIdx];

        TModelCtr& modelCtr = modelSplit.OnlineCtr.Ctr;
        modelCtr.Projection = ExtractProjection(ctr);
        modelCtr.CtrType = ctr.Configuration.Type;

        const auto& config = ctr.Configuration;
        modelCtr.TargetBorderIdx = config.ParamId;
        modelCtr.TargetBorderClassifierIdx = 0;
        modelCtr.PriorNum = GetNumeratorShift(config);
        modelCtr.PriorDenom = GetDenumeratorShift(config);

        return modelSplit;
    }

    inline TTensorStructure3 ConvertStructure(const TObliviousTreeStructure& structure) const {
        TTensorStructure3 structure3;
        for (auto split : structure.Splits) {
            TModelSplit modelSplit;
            if (FeaturesManager.IsFloat(split.FeatureId)) {
                modelSplit = CreateFloatSplit(split);
            } else if (FeaturesManager.IsCat(split.FeatureId)) {
                modelSplit = CreateOneHotSplit(split);
            } else {
                modelSplit = CreateCtrSplit(split);
            }
            structure3.Add(modelSplit);
        }
        return structure3;
    }

    yvector<TTargetClassifier> CreateTargetClassifiers() const {
        TTargetClassifier targetClassifier(FeaturesManager.GetTargetBorders());
        yvector<TTargetClassifier> classifiers;
        classifiers.resize(1, targetClassifier);
        return classifiers;
    }

private:
    const TBinarizedFeaturesManager& FeaturesManager;
    const TDataProvider& DataProvider;
    yvector<yvector<int>> CatFeatureBinToHashIndex;
    ymap<ui32, ui32> CatFeaturesRemap;
    ymap<ui32, ui32> FloatFeaturesRemap;
    yvector<yvector<float>> Borders;
};

TCoreModel ConvertToCoreModel(const TBinarizedFeaturesManager& manager,
                              const TDataProvider& dataProvider,
                              const TString& catFeatureBinarizationTempName,
                              const TAdditiveModel<TObliviousTreeModel>& treeModel) {
    TModelConverter converter(manager, dataProvider, catFeatureBinarizationTempName);
    return converter.Convert(treeModel);
}
