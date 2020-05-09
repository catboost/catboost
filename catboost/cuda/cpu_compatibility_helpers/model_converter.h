#pragma once

#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/models/non_symmetric_tree.h>
#include <catboost/cuda/models/oblivious_model.h>

#include <catboost/private/libs/algo/projection.h>
#include <catboost/libs/data/quantized_features_info.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/target_classifier.h>
#include <catboost/private/libs/target/classification_target_helper.h>

namespace NCatboostCuda {
    class TModelConverter {
    public:
        TModelConverter(const TBinarizedFeaturesManager& manager,
                        const NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
                        const NCB::TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap,
                        const NCB::TClassificationTargetHelper& targetHelper);

        TFullModel Convert(const TAdditiveModel<TObliviousTreeModel>& src,
                           THashMap<TFeatureCombination, TProjection>* featureCombinationToProjection) const;
        TFullModel Convert(const TAdditiveModel<TNonSymmetricTree>& src,
                           THashMap<TFeatureCombination, TProjection>* featureCombinationToProjection) const;

    private:
        TModelSplit CreateFloatSplit(const TBinarySplit& split) const;

        TModelSplit CreateEstimatedFeatureSplit(const TBinarySplit& split) const;

        TModelSplit CreateOneHotSplit(const TBinarySplit& split) const;

        ui32 GetRemappedIndex(ui32 featureId) const;

        void ExtractProjection(const TCtr& ctr,
                               TFeatureCombination* dstFeatureCombination,
                               TProjection* dstProjection) const;

        TModelSplit CreateCtrSplit(const TBinarySplit& split,
                                   THashMap<TFeatureCombination, TProjection>* featureCombinationToProjection) const;

        TVector<TModelSplit> ConvertSplits(const TVector<TBinarySplit>& splits,
                                           THashMap<TFeatureCombination, TProjection>* featureCombinationToProjection) const;

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCB::TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
        const NCB::TFeaturesLayout& FeaturesLayout;
        const NCB::TPerfectHashedToHashedCatValuesMap& CatFeatureBinToHashIndex;
        TVector<ENanMode> FloatFeaturesNanMode;

        const NCB::TClassificationTargetHelper& TargetHelper;
    };

    TVector<TTargetClassifier> CreateTargetClassifiers(const TBinarizedFeaturesManager& featuresManager);

    template <typename TModel>
    inline TFullModel ConvertToCoreModel(const TBinarizedFeaturesManager& manager,
                                         const NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
                                         const NCB::TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap,
                                         const NCB::TClassificationTargetHelper& targetHelper,
                                         const TAdditiveModel<TModel>& treeModel,
                                         THashMap<TFeatureCombination, TProjection>* featureCombinationToProjection) {
        TModelConverter converter(manager,
                                  quantizedFeaturesInfo,
                                  perfectHashedToHashedCatValuesMap,
                                  targetHelper);
        return converter.Convert(treeModel, featureCombinationToProjection);
    }
}
