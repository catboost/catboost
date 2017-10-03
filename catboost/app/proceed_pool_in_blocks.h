#pragma once

#include "cmd_line.h"
#include <catboost/libs/data/load_data.h>

template <class TConsumer>
inline void ReadAndProceedPoolInBlocks(const TAnalyticalModeCommonParams& params,
                                       ui32 blockSize,
                                       TConsumer&& poolConsumer) {
    TPool pool;
    THolder<IPoolBuilder> poolBuilder = InitBuilder(&pool);
    TPoolReader poolReader(params.CdFile, params.InputPath, params.PairsFile, params.ThreadCount,
                           params.Delimiter, params.HasHeader, params.ClassNames,
                           poolBuilder.Get(),
                           blockSize);

    while (poolReader.ReadBlock()) {
        StartBuilder(poolReader.FeatureIds, poolReader.PoolMetaInfo, false, poolBuilder.Get());
        poolReader.ProcessBlock();
        FinalizeBuilder(poolReader.ColumnsDescription, poolReader.PairsFile, poolBuilder.Get());
        const TPool& poolConstRef = pool;
        poolConsumer(poolConstRef);
    }
}
