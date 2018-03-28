#pragma once

#include <catboost/libs/options/load_options.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/cuda/cpu_compatibility_helpers/full_model_saver.h>

namespace NCatboostCuda {
    inline void MakeFullModel(TFullModel&& coreModel,
                              const TPool& pool,
                              const TVector<TTargetClassifier>& targetClassifiers,
                              ui32 numThreads,
                              const TString& fullModelPath) {
        CB_ENSURE(numThreads);
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(numThreads - 1);
        TCoreModelToFullModelConverter converter(std::move(coreModel), pool, targetClassifiers, localExecutor);
        converter.SaveToFile(fullModelPath);
    }

    inline void MakeFullModel(TFullModel&& coreModel,
                              const TPool& pool,
                              const TVector<TTargetClassifier>& targetClassifiers,
                              ui32 numThreads,
                              TFullModel* model) {
        CB_ENSURE(numThreads);
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(numThreads - 1);
        TCoreModelToFullModelConverter converter(std::move(coreModel), pool, targetClassifiers, localExecutor);
        converter.SaveToModel(model);
    }

    inline void MakeFullModel(const TString& coreModelPath,
                              const NCatboostOptions::TPoolLoadParams& poolLoadOptions,
                              const TVector<TString>& classNames,
                              const TVector<TTargetClassifier>& targetClassifiers,
                              const ui32 numThreads,
                              const TString& fullModelPath) {
        TPool pool;
        ReadPool(poolLoadOptions.CdFile,
                 poolLoadOptions.LearnFile,
                 "",
                 poolLoadOptions.IgnoredFeatures,
                 numThreads,
                 false,
                 poolLoadOptions.Delimiter,
                 poolLoadOptions.HasHeader,
                 classNames,
                 &pool);

        TFullModel coreModel;
        {
            TIFStream modelInput(coreModelPath);
            coreModel.Load(&modelInput);
        }
        MakeFullModel(std::move(coreModel), pool, targetClassifiers, numThreads, fullModelPath);
    }
}
