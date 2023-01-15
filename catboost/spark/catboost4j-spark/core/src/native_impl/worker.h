#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/quantized_features_info.h>

#include <util/generic/fwd.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>


void CreateTrainingDataForWorker(
    i32 hostId,
    i32 numThreads,
    const TString& plainJsonParamsAsString,
    NCB::TDataProviderPtr trainDataProvider,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    NCB::TDataProviderPtr trainEstimatedDataProvider, // can be nullptr
    const TString& precomputedOnlineCtrMetaDataAsJsonString
) throw (yexception);

// needed for forwarding exceptions from C++ to JVM
void RunWorkerWrapper(i32 numThreads, i32 nodePort) throw (yexception);
