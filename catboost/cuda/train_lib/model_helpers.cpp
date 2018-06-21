#include "model_helpers.h"
#include <catboost/libs/data/load_data.h>

namespace NCatboostCuda {
   void MakeFullModel(TFullModel&& coreModel,
                      const TPool& pool,
                      const TVector<TTargetClassifier>& targetClassifiers,
                      ui32 numThreads,
                      const TString& fullModelPath,
                      EFinalCtrComputationMode finalCtrComputationMode) {
        CB_ENSURE(numThreads);
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(numThreads - 1);
        TCoreModelToFullModelConverter converter(std::move(coreModel), pool, targetClassifiers, localExecutor);
        converter.SaveToFile(finalCtrComputationMode, fullModelPath);
    }

    void MakeFullModel(TFullModel&& coreModel,
                       const TPool& pool,
                       const TVector<TTargetClassifier>& targetClassifiers,
                       ui32 numThreads,
                       TFullModel* model,
                       EFinalCtrComputationMode finalCtrComputationMode) {
        CB_ENSURE(numThreads);
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(numThreads - 1);
        TCoreModelToFullModelConverter converter(std::move(coreModel), pool, targetClassifiers, localExecutor);
        converter.SaveToModel(finalCtrComputationMode, model);
    }

    void MakeFullModel(const TString& coreModelPath,
                       const NCatboostOptions::TPoolLoadParams& poolLoadOptions,
                       const TVector<TString>& classNames,
                       const TVector<TTargetClassifier>& targetClassifiers,
                       const ui32 numThreads,
                       const TString& fullModelPath,
                       EFinalCtrComputationMode finalCtrComputationMode) {
        TPool pool;
        if (finalCtrComputationMode == EFinalCtrComputationMode::Default) {
            NCB::ReadPool(poolLoadOptions.LearnSetPath,
                          NCB::TPathWithScheme(),
                          poolLoadOptions.DsvPoolFormatParams,
                          poolLoadOptions.IgnoredFeatures,
                          numThreads,
                          false,
                          classNames,
                          &pool);
        }
        TFullModel coreModel;
        {
            TIFStream modelInput(coreModelPath);
            coreModel.Load(&modelInput);
        }
        MakeFullModel(std::move(coreModel), pool, targetClassifiers, numThreads, fullModelPath, finalCtrComputationMode);
    }
}
