#pragma once

#include "cmd_line.h"

#include <catboost/libs/data/load_data.h>

#include <library/threading/local_executor/local_executor.h>

template <class TConsumer>
inline void ReadAndProceedPoolInBlocks(const TAnalyticalModeCommonParams& params,
                                       ui32 blockSize,
                                       TConsumer&& poolConsumer) {
    TPool pool;
    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(params.ThreadCount - 1);

    THolder<IPoolBuilder> poolBuilder = InitBuilder(&pool, &localExecutor);
    TPoolReader poolReader(params.CdFile, params.InputPath, params.PairsFile,
                           /*ignoredFeatures*/ {},
                           params.Delimiter, params.HasHeader, params.ClassNames,
                           blockSize, poolBuilder.Get(), &localExecutor);
    int offset = 0;
    while (poolReader.ReadBlock()) {
        StartBuilder(poolReader.FeatureIds, poolReader.PoolMetaInfo, poolReader.GetBlockSize(), false, offset, poolBuilder.Get());
        poolReader.ProcessBlock();
        FinalizeBuilder(poolReader.ColumnsDescription, poolReader.PairsFile, poolBuilder.Get());
        poolConsumer(pool);
        offset += blockSize;
    }
}
