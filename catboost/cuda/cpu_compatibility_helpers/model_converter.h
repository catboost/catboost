#pragma once

#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/target_classifier.h>

namespace NCatboostCuda {
    //store hash = result[i][bin] is catFeatureHash for feature catFeatures[i]
    TVector<TVector<int>> MakeInverseCatFeatureIndexForDataProviderIds(const TBinarizedFeaturesManager& featuresManager,
                                                                      const TVector<ui32>& catFeaturesDataProviderIds,
                                                                      bool clearFeatureManagerRamCache = true);

    class TModelConverter {
    public:
        TModelConverter(const TBinarizedFeaturesManager& manager,
                        const TDataProvider& dataProvider);

        TFullModel Convert(const TAdditiveModel<TObliviousTreeModel>& src) const;

    private:
        TModelSplit CreateFloatSplit(const TBinarySplit& split) const;

        TModelSplit CreateOneHotSplit(const TBinarySplit& split) const;

        ui32 GetRemappedIndex(ui32 featureId) const;

        TFeatureCombination ExtractProjection(const TCtr& ctr) const;

        TModelSplit CreateCtrSplit(const TBinarySplit& split) const;

        TVector<TModelSplit> ConvertStructure(const TObliviousTreeStructure& structure) const;

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const TDataProvider& DataProvider;
        TVector<TVector<int>> CatFeatureBinToHashIndex;
        TMap<ui32, ui32> CatFeaturesRemap;
        TMap<ui32, ui32> FloatFeaturesRemap;
        TVector<ENanMode> FloatFeaturesNanMode;
        TVector<TVector<float>> Borders;
    };

    TVector<TTargetClassifier> CreateTargetClassifiers(const TBinarizedFeaturesManager& featuresManager);

    inline TFullModel ConvertToCoreModel(const TBinarizedFeaturesManager& manager,
                                         const TDataProvider& dataProvider,
                                         const TAdditiveModel<TObliviousTreeModel>& treeModel) {
        TModelConverter converter(manager, dataProvider);
        return converter.Convert(treeModel);
    }
}
