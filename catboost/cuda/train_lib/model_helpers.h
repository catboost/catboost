#pragma once

#include <catboost/libs/options/load_options.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>
#include <catboost/cuda/cpu_compatibility_helpers/full_model_saver.h>
#include <catboost/libs/options/enums.h>

namespace NCatboostCuda {
    void MakeFullModel(TFullModel&& coreModel,
                       const TPool& pool,
                       const TVector<TTargetClassifier>& targetClassifiers,
                       ui32 numThreads,
                       const TString& fullModelPath,
                       EFinalCtrComputationMode finalCtrComputationMode);

    void MakeFullModel(TFullModel&& coreModel,
                       const TPool& pool,
                       const TVector<TTargetClassifier>& targetClassifiers,
                       ui32 numThreads,
                       TFullModel* model,
                       EFinalCtrComputationMode finalCtrComputationMode);

    void MakeFullModel(const TString& coreModelPath,
                       const NCatboostOptions::TPoolLoadParams& poolLoadOptions,
                       const TVector<TString>& classNames,
                       const TVector<TTargetClassifier>& targetClassifiers,
                       const ui32 numThreads,
                       const TString& fullModelPath,
                       EFinalCtrComputationMode finalCtrComputationMode);
}
