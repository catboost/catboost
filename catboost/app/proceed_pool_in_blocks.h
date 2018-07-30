#pragma once

#include "cmd_line.h"

#include <catboost/libs/data/doc_pool_data_provider.h>
#include <catboost/libs/data/load_data.h>

#include <library/threading/local_executor/local_executor.h>

template <class TConsumer>
inline void ReadAndProceedPoolInBlocks(const TAnalyticalModeCommonParams& params,
                                       ui32 blockSize,
                                       TConsumer&& poolConsumer,
                                       NPar::TLocalExecutor* localExecutor) {
    TPool pool;
    THolder<NCB::IPoolBuilder> poolBuilder = NCB::InitBuilder(params.InputPath, *localExecutor, &pool);

    auto docPoolDataProvider = NCB::GetProcessor<NCB::IDocPoolDataProvider>(
        params.InputPath, // for choosing processor

        // processor args
        NCB::TDocPoolDataProviderArgs {
            params.InputPath,
            params.PairsFilePath,
            params.DsvPoolFormatParams,
            /*ignoredFeatures*/ {},
            params.ClassNames,
            blockSize,
            localExecutor
        }
    );

    while (docPoolDataProvider->DoBlock(poolBuilder.Get())) {
        poolConsumer(pool);
    }
}
