#pragma once

#include <catboost/libs/data/meta_info.h>

#include <catboost/private/libs/options/enums.h>


namespace NPar {
    class ILocalExecutor;
}


namespace NCB {
    class TRawTargetDataProvider;


    TTargetStats ComputeTargetStatsForYetiRank(
        const TRawTargetDataProvider& rawTargetData,
        ELossFunction lossFunction,
        NPar::ILocalExecutor* localExecutor
    );

}
