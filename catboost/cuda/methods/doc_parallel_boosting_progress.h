#pragma once

#include "serialization_helper.h"
#include <util/ysaveload.h>
#include <util/folder/path.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/cuda/cuda_lib/read_and_write_helpers.h>

namespace NCatboostCuda {
    template <class TModel>
    struct TBoostingProgress {
        TVector<TModel> Models;
        TModelFeaturesMap ModelFeaturesMap;

        Y_SAVELOAD_DEFINE(Models, ModelFeaturesMap);
    };

    template <class TModel>
    TBoostingProgress<TModel> MakeProgress(const TBinarizedFeaturesManager& featuresManager,
                                           const TVector<TModel>& models) {
        TBoostingProgress<TModel> progress;
        progress.Models = models;
        for (auto& model : models) {
            TModelFeaturesBuilder<TModel>::Write(featuresManager, model, progress.ModelFeaturesMap);
        }
        return progress;
    };

    template <class TModel>
    TVector<TModel> RestoreFromProgress(TBinarizedFeaturesManager& featuresManager,
                                        const TBoostingProgress<TModel>& progress) {
        TVector<TModel> models;
        models.reserve(progress.Models.size());
        for (const auto& progressModel : progress.Models) {
            models.push_back(TFeatureIdsRemaper<TModel>::Remap(featuresManager, progress.ModelFeaturesMap, progressModel));
        }
        return models;
    }

}
