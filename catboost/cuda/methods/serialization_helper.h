#pragma once

#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/ctrs/ctr.h>
#include <catboost/cuda/data/binarizations_manager.h>

#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <utility>
#include <catboost/libs/logging/logging.h>

namespace NCatboostCuda {
    inline TString GpuProgressLabel() {
        return ToString<ETaskType>(ETaskType::GPU);
    }

    //TODO(noxoomo): vector of external serializers/deserializers instead of task options. we should correctly restore overffitting-detector state, best iteration, etc
    struct TSnapshotMeta {
        TString Path;
        TString TaskOptions;
        ui64 SaveIntervalSeconds;
    };

    template <class TModel>
    class TFeatureIdsRemaper;

    struct TModelFeaturesMap {
        struct TCtrFeature {
            TCtr Ctr;
            TVector<float> Borders;

            TCtrFeature() = default;

            TCtrFeature(const TCtr& ctr,
                        TVector<float> borders)
                : Ctr(ctr)
                , Borders(std::move(borders))
            {
            }

            Y_SAVELOAD_DEFINE(Ctr, Borders);
        };

        struct TFloatFeature {
            ui32 DataProviderId;
            TVector<float> Borders;

            TFloatFeature() = default;

            TFloatFeature(const ui32& dataProviderId,
                          TVector<float> borders)
                : DataProviderId(dataProviderId)
                , Borders(std::move(borders))
            {
            }

            Y_SAVELOAD_DEFINE(DataProviderId, Borders);
        };

        TMap<ui32, TCtrFeature> Ctrs;
        TMap<ui32, TFloatFeature> FloatFeatures;
        TMap<ui32, ui32> CatFeaturesMap;

        Y_SAVELOAD_DEFINE(Ctrs, FloatFeatures, CatFeaturesMap);
    };

    TCtr MigrateCtr(TBinarizedFeaturesManager& featuresManager,
                    const TModelFeaturesMap& map,
                    const TCtr& oldCtr);

    ui32 UpdateFeatureId(TBinarizedFeaturesManager& featuresManager,
                         const TModelFeaturesMap& map,
                         const ui32 featureId);

    template <>
    class TFeatureIdsRemaper<TObliviousTreeModel> {
    public:
        static TObliviousTreeModel Remap(TBinarizedFeaturesManager& featuresManager,
                                         const TModelFeaturesMap& map,
                                         const TObliviousTreeModel& src) {
            TObliviousTreeStructure structure = src.GetStructure();
            for (ui32 i = 0; i < structure.Splits.size(); ++i) {
                structure.Splits[i].FeatureId = UpdateFeatureId(featuresManager, map, structure.Splits[i].FeatureId);
            }
            return TObliviousTreeModel(std::move(structure), src.GetValues());
        }
    };

    template <class TModel>
    class TModelFeaturesBuilder;

    class TModelFeaturesMapUpdater {
    public:
        TModelFeaturesMapUpdater(const TBinarizedFeaturesManager& featuresManager,
                                 TModelFeaturesMap& featuresMap)
            : FeaturesManager(featuresManager)
            , FeaturesMap(featuresMap)
        {
        }

        void AddFeature(ui32 featureId) {
            if (FeaturesManager.IsFloat(featureId)) {
                AddFloatFeature(featureId);
            } else if (FeaturesManager.IsCat(featureId)) {
                AddCatFeature(featureId);
            } else {
                CB_ENSURE(FeaturesManager.IsCtr(featureId), "Unknown feature id #" << featureId);
                AddCtr(featureId);
            }
        }

    private:
        void AddFloatFeature(ui32 featureId) {
            Y_ASSERT(FeaturesManager.IsFloat(featureId));
            if (FeaturesMap.FloatFeatures.has(featureId)) {
                return;
            }
            FeaturesMap.FloatFeatures[featureId] = TModelFeaturesMap::TFloatFeature(FeaturesManager.GetDataProviderId(featureId),
                                                                                    FeaturesManager.GetBorders(featureId));
        }

        void AddCatFeature(ui32 featureId) {
            Y_ASSERT(FeaturesManager.IsCat(featureId));
            FeaturesMap.CatFeaturesMap[featureId] = FeaturesManager.GetDataProviderId(featureId);
        }

        void AddCtr(ui32 featureId) {
            Y_ASSERT(FeaturesManager.IsCtr(featureId));
            if (FeaturesMap.Ctrs.has(featureId)) {
                return;
            }
            TCtr ctr = FeaturesManager.GetCtr(featureId);
            TVector<float> borders = FeaturesManager.GetBorders(featureId);
            FeaturesMap.Ctrs[featureId] = TModelFeaturesMap::TCtrFeature(ctr, std::move(borders));

            for (auto split : ctr.FeatureTensor.GetSplits()) {
                AddFeature(split.FeatureId);
            }
            for (ui32 catId : ctr.FeatureTensor.GetCatFeatures()) {
                AddFeature(catId);
            }
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        TModelFeaturesMap& FeaturesMap;
    };

    template <>
    class TModelFeaturesBuilder<TObliviousTreeModel> {
    public:
        static void Write(const TBinarizedFeaturesManager& manager,
                          const TObliviousTreeModel& model,
                          TModelFeaturesMap& modelFeatures) {
            for (const auto& split : model.GetStructure().Splits) {
                TModelFeaturesMapUpdater(manager, modelFeatures).AddFeature(split.FeatureId);
            }
        }
    };

    template <class TModel>
    class TModelFeaturesBuilder<TAdditiveModel<TModel>> {
    public:
        static void Write(const TBinarizedFeaturesManager& manager,
                          const TAdditiveModel<TModel>& model,
                          TModelFeaturesMap& modelFeatures) {
            for (auto& weak : model.WeakModels) {
                TModelFeaturesBuilder<TModel>::Write(manager, weak, modelFeatures);
            }
        }
    };

    template <class TModel>
    class TFeatureIdsRemaper<TAdditiveModel<TModel>> {
    public:
        static TAdditiveModel<TModel> Remap(TBinarizedFeaturesManager& featuresManager,
                                            const TModelFeaturesMap& map,
                                            const TAdditiveModel<TModel>& src) {
            TAdditiveModel<TModel> result;
            for (ui32 i = 0; i < src.WeakModels.size(); ++i) {
                result.WeakModels.push_back(TFeatureIdsRemaper<TModel>::Remap(featuresManager,
                                                                              map,
                                                                              src.WeakModels[i]));
            }
            return result;
        }
    };

}
