#pragma once

#include <catboost/cuda/data/load_config.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/cuda/cpu_compatibility_helpers/full_model_saver.h>

namespace NCatboostCuda
{
    inline void MakeFullModel(const TCoreModel& coreModel,
                              const TPool& pool,
                              ui32 numThreads,
                              const TString& fullModelPath)
    {
        CB_ENSURE(numThreads);
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(numThreads - 1);
        TCoreModelToFullModelConverter converter(coreModel, pool, localExecutor);
        converter.SaveToFile(fullModelPath);
    }

    inline void MakeFullModel(const TCoreModel& coreModel,
                              const TPool& pool,
                              ui32 numThreads,
                              TFullModel* model)
    {
        CB_ENSURE(numThreads);
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(numThreads - 1);
        TCoreModelToFullModelConverter converter(coreModel, pool, localExecutor);
        converter.SaveToModel(model);
    }

    inline void MakeFullModel(const TString& coreModelPath,
                              const TPoolLoadOptions& poolLoadOptions,
                              const ui32 numThreads,
                              const TString& fullModelPath)
    {
        TPool pool;
        ReadPool(poolLoadOptions.GetColumnDescriptionName(),
                 poolLoadOptions.GetFeaturesFilename(),
                 "",
                 numThreads,
                 false,
                 poolLoadOptions.GetDelimiter(),
                 poolLoadOptions.HasHeader(),
                 poolLoadOptions.GetClassNames(),
                 &pool);

        TCoreModel coreModel;
        {
            TIFStream modelInput(coreModelPath);
            coreModel.Load(&modelInput);
        }
        MakeFullModel(coreModel, pool, numThreads, fullModelPath);
    }
}
