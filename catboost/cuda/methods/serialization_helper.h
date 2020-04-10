#pragma once

#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/ctrs/ctr.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/models/region_model.h>
#include <catboost/cuda/models/non_symmetric_tree.h>
#include <utility>
#include <catboost/libs/data/feature_estimators.h>
#include <catboost/libs/logging/logging.h>

namespace NCatboostCuda {

    inline TString GpuProgressLabel() {
        return ToString<ETaskType>(ETaskType::GPU);
    }

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
            ui32 Feature;
            TVector<float> Borders;

            TFloatFeature() = default;

            TFloatFeature(const ui32& dataProviderId,
                          TVector<float> borders)
                : Feature(dataProviderId)
                , Borders(std::move(borders))
            {
            }

            Y_SAVELOAD_DEFINE(Feature, Borders);
        };

        struct TCalculatedFeature {
            NCB::TEstimatedFeatureId Feature;
            TVector<float> Borders;

            TCalculatedFeature() = default;

            TCalculatedFeature(const NCB::TEstimatedFeatureId& id,
                              TVector<float> borders)
                : Feature(id)
                , Borders(std::move(borders))
            {
            }

            Y_SAVELOAD_DEFINE(Feature, Borders);
        };



        TMap<ui32, TCtrFeature> Ctrs;
        TMap<ui32, TFloatFeature> FloatFeatures;
        TMap<ui32, ui32> CatFeaturesMap;
        TMap<ui32, TCalculatedFeature> CalculatedFeaturesMap;

        Y_SAVELOAD_DEFINE(Ctrs, FloatFeatures, CatFeaturesMap, CalculatedFeaturesMap);
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
            return TObliviousTreeModel(std::move(structure),
                                       src.GetValues(),
                                       src.GetWeights(),
                                       src.OutputDim());
        }
    };

    template <>
    class TFeatureIdsRemaper<TRegionModel> {
    public:
        static TRegionModel Remap(TBinarizedFeaturesManager& featuresManager,
                                  const TModelFeaturesMap& map,
                                  const TRegionModel& src) {
            TRegionStructure structure = src.GetStructure();
            for (ui32 i = 0; i < structure.Splits.size(); ++i) {
                structure.Splits[i].FeatureId = UpdateFeatureId(featuresManager, map, structure.Splits[i].FeatureId);
            }
            return TRegionModel(std::move(structure),
                                src.GetValues(),
                                src.GetWeights(),
                                src.OutputDim());
        }
    };

    template <>
    class TFeatureIdsRemaper<TNonSymmetricTree> {
    public:
        static TNonSymmetricTree Remap(TBinarizedFeaturesManager& featuresManager,
                                       const TModelFeaturesMap& map,
                                       const TNonSymmetricTree& src) {
            auto structure = src.GetStructure();
            for (ui32 i = 0; i < structure.GetNodes().size(); ++i) {
                structure.GetNodes()[i].FeatureId = UpdateFeatureId(featuresManager, map, structure.GetNodes()[i].FeatureId);
            }
            return TNonSymmetricTree(std::move(structure),
                                     src.GetValues(),
                                     src.GetWeights(),
                                     src.OutputDim());
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
            } else if (FeaturesManager.IsEstimatedFeature(featureId)) {
                AddEstimatedFeature(featureId);
            } else {
                CB_ENSURE(FeaturesManager.IsCtr(featureId), "Unknown feature id #" << featureId);
                AddCtr(featureId);
            }
        }

    private:
        void AddFloatFeature(ui32 featureId) {
            Y_ASSERT(FeaturesManager.IsFloat(featureId));
            if (FeaturesMap.FloatFeatures.contains(featureId)) {
                return;
            }
            // we store here featureManager id to store info about wide features splitted into N subfeatures
            FeaturesMap.FloatFeatures[featureId] = TModelFeaturesMap::TFloatFeature(featureId,
                                                                                    FeaturesManager.GetBorders(featureId));
        }

        void AddCatFeature(ui32 featureId) {
            Y_ASSERT(FeaturesManager.IsCat(featureId));
            FeaturesMap.CatFeaturesMap[featureId] = FeaturesManager.GetDataProviderId(featureId);
        }

        void AddEstimatedFeature(ui32 featureId) {
            Y_ASSERT(FeaturesManager.IsEstimatedFeature(featureId));
            FeaturesMap.CalculatedFeaturesMap[featureId] = TModelFeaturesMap::TCalculatedFeature(FeaturesManager.GetEstimatedFeature(featureId),
                                                                                                 FeaturesManager.GetBorders(featureId));
        }
        void AddCtr(ui32 featureId) {
            Y_ASSERT(FeaturesManager.IsCtr(featureId));
            if (FeaturesMap.Ctrs.contains(featureId)) {
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

    template <>
    class TModelFeaturesBuilder<TRegionModel> {
    public:
        static void Write(const TBinarizedFeaturesManager& manager,
                          const TRegionModel& model,
                          TModelFeaturesMap& modelFeatures) {
            for (const auto& split : model.GetStructure().Splits) {
                TModelFeaturesMapUpdater(manager, modelFeatures).AddFeature(split.FeatureId);
            }
        }
    };

    template <>
    class TModelFeaturesBuilder<TNonSymmetricTree> {
    public:
        static void Write(const TBinarizedFeaturesManager& manager,
                          const TNonSymmetricTree& model,
                          TModelFeaturesMap& modelFeatures) {
            for (const auto& split : model.GetStructure().GetNodes()) {
                TModelFeaturesMapUpdater(manager, modelFeatures).AddFeature((ui32)split.FeatureId);
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

    template <class T, class TMapping>
    inline void SaveCudaBuffer(const NCudaLib::TCudaBuffer<T, TMapping>& data, IOutputStream* out) {
        ui64 size = data.GetMapping().GetObjectsSlice().Size();
        if (size == 0) {
            return;
        }
        TVector<T> cpuData;
        data.Read(cpuData);
        ::Save(out, cpuData);
    }

    template <class T, class TMapping>
    inline void LoadCudaBuffer(IInputStream* in, NCudaLib::TCudaBuffer<T, TMapping>* data) {
        ui64 size = data->GetMapping().GetObjectsSlice().Size();
        if (size == 0) {
            return;
        }
        TVector<T> cpuData;
        ::Load(in, cpuData);
        const auto expectedBufferSize = size * data->GetColumnCount();
        CB_ENSURE(cpuData.size() == expectedBufferSize, "Inconsistent data: expected " << expectedBufferSize << ", got " << cpuData.size());
        data->Write(cpuData);
    }

}
