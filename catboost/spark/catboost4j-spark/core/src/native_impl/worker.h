#pragma once

#include <catboost/libs/data/data_provider.h>

#include <util/system/types.h>


void RunWorker(
    i32 hostId,
    i32 nodePort,
    i32 numThreads,
    const TString& plainJsonParamsAsString,
    NCB::TDataProviderPtr trainDataProvider,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo
) throw (yexception);
