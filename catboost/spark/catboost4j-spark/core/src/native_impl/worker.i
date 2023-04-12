%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/worker.h>
#include <catboost/private/libs/distributed/worker.h>
%}

%include <bindings/swiglib/stroka.swg>

%include "defaults.i"
%include "data_provider.i"
%include "quantized_features_info.i"


%catches(yexception) RunWorker(ui32 numThreads, ui32 nodePort);

void RunWorker(ui32 numThreads, ui32 nodePort);


%catches(std::exception) GetPartitionTotalObjectCount(
    const TVector<NCB::TDataProviderPtr>& trainDataProviders
);

%catches(yexception) CreateTrainingDataForWorker(
    i32 hostId,
    i32 numThreads,
    const TString& plainJsonParamsAsString,
    const TVector<i8>& serializedLabelConverter,
    const TVector<NCB::TDataProviderPtr>& trainDataProviders,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    const TVector<NCB::TDataProviderPtr>& trainEstimatedDataProviders, // can be empty
    const TString& precomputedOnlineCtrMetaDataAsJsonString
);

%include "worker.h"
