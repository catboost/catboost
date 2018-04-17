#pragma once

#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/data/cat_feature_perfect_hash.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/target_classifier.h>
#include <catboost/libs/algo/projection.h>
#include <catboost/libs/algo/split.h>
#include <catboost/libs/model/model_build_helper.h>
#include <limits>

namespace NCatboostCuda {
    //store hash = result[i][bin] is catFeatureHash for feature catFeatures[i]
    inline TVector<TVector<int>>
    MakeInverseCatFeatureIndexForDataProviderIds(const TBinarizedFeaturesManager& featuresManager,
                                                 const TVector<ui32>& catFeaturesDataProviderIds,
                                                 bool clearFeatureManagerRamCache = true) {
        TVector<TVector<int>> result(catFeaturesDataProviderIds.size());
        for (ui32 i = 0; i < catFeaturesDataProviderIds.size(); ++i) {
            const ui32 featureManagerId = featuresManager.GetFeatureManagerIdForCatFeature(
                catFeaturesDataProviderIds[i]);
            const auto& perfectHash = featuresManager.GetCategoricalFeaturesPerfectHash(featureManagerId);

            if (!perfectHash.empty()) {
                result[i].resize(perfectHash.size());
                for (const auto& entry : perfectHash) {
                    result[i][entry.second] = entry.first;
                }
            }
        }
        if (clearFeatureManagerRamCache) {
            featuresManager.UnloadCatFeaturePerfectHashFromRam();
        }
        return result;
    }

    class TModelConverter {
    public:
        TModelConverter(const TBinarizedFeaturesManager& manager,
                        const TDataProvider& dataProvider)
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
                            const auto featureIdInFeaturesManager = manager.GetFeatureManagerIdForFloatFeature(featureId);
                            TVector<float> borders = manager.GetBorders(featureIdInFeaturesManager);
                            Borders.push_back(std::move(borders));
                            FloatFeaturesNanMode.push_back(manager.GetNanMode(featureIdInFeaturesManager));
                        } else {
                            Borders.push_back(TVector<float>());
                            FloatFeaturesNanMode.push_back(ENanMode::Forbidden);
                        }
                        FloatFeaturesRemap[featureId] = static_cast<ui32>(FloatFeaturesRemap.size());
                    }
                }
            }
            {
                TVector<ui32> catFeatureVec(catFeatureIds.begin(), catFeatureIds.end());
                CatFeatureBinToHashIndex = MakeInverseCatFeatureIndexForDataProviderIds(manager, catFeatureVec);
            }
        }

        TFullModel Convert(const TAdditiveModel<TObliviousTreeModel>& src) const {
            const auto& featureNames = DataProvider.GetFeatureNames();
            const auto& catFeatureIds = DataProvider.GetCatFeatureIds();
            TFullModel coreModel;
            coreModel.ModelInfo["params"] = "{}"; //will be overriden with correct params later
            auto featureCount = featureNames.ysize();
            TVector<TFloatFeature> floatFeatures;
            TVector<TCatFeature> catFeatures;

            for (int i = 0; i < featureCount; ++i) {
                if (catFeatureIds.has(i)) {
                    auto catFeatureIdx = catFeatures.size();
                    auto& catFeature = catFeatures.emplace_back();
                    catFeature.FeatureIndex = catFeatureIdx;
                    catFeature.FlatFeatureIndex = i;
                    catFeature.FeatureId = featureNames[catFeature.FlatFeatureIndex];
                    Y_ASSERT((ui32)catFeature.FeatureIndex == CatFeaturesRemap.at(i));
                } else {
                    auto floatFeatureIdx = floatFeatures.size();
                    auto& floatFeature = floatFeatures.emplace_back();
                    const bool hasNans = FloatFeaturesNanMode.at(floatFeatureIdx) != ENanMode::Forbidden;
                    floatFeature.FeatureIndex = floatFeatureIdx;
                    floatFeature.FlatFeatureIndex = i;
                    floatFeature.Borders = Borders[floatFeatureIdx];
                    floatFeature.FeatureId = featureNames[i];
                    floatFeature.HasNans = hasNans;
                    if (hasNans) {
                        if (FloatFeaturesNanMode.at(floatFeatureIdx) == ENanMode::Min) {
                            floatFeature.NanValueTreatment = NCatBoostFbs::ENanValueTreatment_AsFalse;
                        } else {
                            floatFeature.NanValueTreatment = NCatBoostFbs::ENanValueTreatment_AsTrue;
                        }
                    }
                    Y_ASSERT((ui32)floatFeature.FeatureIndex == FloatFeaturesRemap.at(i));
                }
            }

            TObliviousTreeBuilder obliviousTreeBuilder(
                floatFeatures,
                catFeatures,
                1
            );

            for (ui32 i = 0; i < src.Size(); ++i) {
                TVector<TVector<double>> leafValues(1);
                const TObliviousTreeModel& model = src.GetWeakModel(i);

                auto& values = model.GetValues();
                leafValues[0].resize(values.size());
                for (ui32 leaf = 0; leaf < values.size(); ++leaf) {
                    leafValues[0][leaf] = values[leaf];
                }

                const auto& structure = model.GetStructure();
                auto treeStructure = ConvertStructure(structure);
                obliviousTreeBuilder.AddTree(treeStructure, leafValues);
            }
            coreModel.ObliviousTrees = obliviousTreeBuilder.Build();
            return coreModel;
        }

    private:
        inline TModelSplit CreateFloatSplit(const TBinarySplit& split) const {
            CB_ENSURE(FeaturesManager.IsFloat(split.FeatureId));

            TModelSplit modelSplit;
            modelSplit.Type = ESplitType::FloatFeature;
            auto dataProviderId = FeaturesManager.GetDataProviderId(split.FeatureId);
            CB_ENSURE(FloatFeaturesRemap.has(dataProviderId));
            auto remapId = FloatFeaturesRemap.at(dataProviderId);

            float border = 0;
            const auto nanMode = FloatFeaturesNanMode.at(remapId);
            switch (nanMode) {
                case ENanMode::Forbidden: {
                    border = Borders.at(remapId).at(split.BinIdx);
                    break;
                }
                case ENanMode::Min: {
                    border = split.BinIdx != 0 ? Borders.at(remapId).at(split.BinIdx - 1) : -std::numeric_limits<float>::lowest();
                    break;
                }
                case ENanMode::Max: {
                    border = split.BinIdx != Borders.at(remapId).size() ? Borders.at(remapId).at(split.BinIdx) : std::numeric_limits<float>::max();
                    break;
                }
                default: {
                    ythrow TCatboostException() << "Unknown NaN mode " << nanMode;
                };
            }
            modelSplit.FloatFeature = TFloatSplit{(int)remapId, border};
            return modelSplit;
        }

        inline TModelSplit CreateOneHotSplit(const TBinarySplit& split) const {
            CB_ENSURE(FeaturesManager.IsCat(split.FeatureId));

            TModelSplit modelSplit;
            modelSplit.Type = ESplitType::OneHotFeature;
            auto dataProviderId = FeaturesManager.GetDataProviderId(split.FeatureId);
            CB_ENSURE(CatFeaturesRemap.has(dataProviderId));
            auto remapId = CatFeaturesRemap.at(dataProviderId);
            CB_ENSURE(CatFeatureBinToHashIndex[remapId].size(),
                      TStringBuilder() << "Error: no catFeature perferct hash for feature " << dataProviderId);
            CB_ENSURE(split.BinIdx < CatFeatureBinToHashIndex[remapId].size(),
                      TStringBuilder() << "Error: no gasg fir feature " << split.FeatureId << " " << split.BinIdx);
            const int hash = CatFeatureBinToHashIndex[remapId][split.BinIdx];
            modelSplit.OneHotFeature = TOneHotSplit(remapId,
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

        TFeatureCombination ExtractProjection(const TCtr& ctr) const {
            TFeatureCombination projection;
            for (auto split : ctr.FeatureTensor.GetSplits()) {
                if (FeaturesManager.IsFloat(split.FeatureId)) {
                    auto floatSplit = CreateFloatSplit(split);
                    projection.BinFeatures.push_back(floatSplit.FloatFeature);
                } else if (FeaturesManager.IsCat(split.FeatureId)) {
                    projection.OneHotFeatures.push_back(CreateOneHotSplit(split).OneHotFeature);
                } else {
                    CB_ENSURE(false, "Error: unknown split type");
                }
            }
            for (auto catFeature : ctr.FeatureTensor.GetCatFeatures()) {
                projection.CatFeatures.push_back(GetRemappedIndex(catFeature));
            }
            //just for more more safety
            Sort(projection.BinFeatures.begin(), projection.BinFeatures.end());
            Sort(projection.CatFeatures.begin(), projection.CatFeatures.end());
            Sort(projection.OneHotFeatures.begin(), projection.OneHotFeatures.end());
            return projection;
        }

        inline TModelSplit CreateCtrSplit(const TBinarySplit& split) const {
            TModelSplit modelSplit;
            CB_ENSURE(FeaturesManager.IsCtr(split.FeatureId));
            const auto& ctr = FeaturesManager.GetCtr(split.FeatureId);
            auto& borders = FeaturesManager.GetBorders(split.FeatureId);
            CB_ENSURE(split.BinIdx < borders.size(), "Split " << split.BinIdx << ", borders: " << borders.size());

            modelSplit.Type = ESplitType::OnlineCtr;
            modelSplit.OnlineCtr.Border = borders[split.BinIdx];

            TModelCtr& modelCtr = modelSplit.OnlineCtr.Ctr;
            modelCtr.Base.Projection = ExtractProjection(ctr);
            modelCtr.Base.CtrType = ctr.Configuration.Type;
            modelCtr.Base.TargetBorderClassifierIdx = 0; // TODO(kirillovs): remove me

            const auto& config = ctr.Configuration;
            modelCtr.TargetBorderIdx = config.ParamId;
            modelCtr.PriorNum = GetNumeratorShift(config);
            modelCtr.PriorDenom = GetDenumeratorShift(config);

            return modelSplit;
        }

        inline TVector<TModelSplit> ConvertStructure(const TObliviousTreeStructure& structure) const {
            TVector<TModelSplit> structure3;
            for (auto split : structure.Splits) {
                TModelSplit modelSplit;
                if (FeaturesManager.IsFloat(split.FeatureId)) {
                    modelSplit = CreateFloatSplit(split);
                } else if (FeaturesManager.IsCat(split.FeatureId)) {
                    modelSplit = CreateOneHotSplit(split);
                } else {
                    modelSplit = CreateCtrSplit(split);
                }
                structure3.push_back(modelSplit);
            }
            return structure3;
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const TDataProvider& DataProvider;
        TVector<TVector<int>> CatFeatureBinToHashIndex;
        TMap<ui32, ui32> CatFeaturesRemap;
        TMap<ui32, ui32> FloatFeaturesRemap;
        TVector<ENanMode> FloatFeaturesNanMode;
        TVector<TVector<float>> Borders;
    };

    inline TVector<TTargetClassifier> CreateTargetClassifiers(const TBinarizedFeaturesManager& featuresManager) {
        TTargetClassifier targetClassifier(featuresManager.GetTargetBorders());
        TVector<TTargetClassifier> classifiers;
        classifiers.resize(1, targetClassifier);
        return classifiers;
    }

    inline TFullModel ConvertToCoreModel(const TBinarizedFeaturesManager& manager,
                                         const TDataProvider& dataProvider,
                                         const TAdditiveModel<TObliviousTreeModel>& treeModel) {
        TModelConverter converter(manager, dataProvider);
        return converter.Convert(treeModel);
    }
}
