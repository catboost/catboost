#include "worker.h"

#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/distributed/data_types.h>
#include <catboost/private/libs/distributed/worker.h>
#include <catboost/private/libs/options/plain_options_helper.h>

#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/logging/logging.h>

#include <library/cpp/json/json_reader.h>

#include <util/generic/cast.h>
#include <util/generic/ptr.h>

using namespace NCB;


void CreateTrainingDataForWorker(
    i32 hostId,
    i32 numThreads,
    const TString& plainJsonParamsAsString,
    NCB::TDataProviderPtr trainDataProvider,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    NCB::TDataProviderPtr trainEstimatedDataProvider,
    const TString& precomputedOnlineCtrMetaDataAsJsonString
) throw (yexception) {
    CB_ENSURE(numThreads >= 1, "Non-positive number of threads specified");

    auto& localData = NCatboostDistributed::TLocalTensorSearchData::GetRef();
    localData = NCatboostDistributed::TLocalTensorSearchData();

    TDataProviders pools;
    pools.Learn = trainDataProvider;

    NJson::TJsonValue plainJsonParams;
    NJson::ReadJsonTree(plainJsonParamsAsString, &plainJsonParams, /*throwOnError*/ true);
    ConvertIgnoredFeaturesFromStringToIndices(trainDataProvider->MetaInfo, &plainJsonParams);

    NJson::TJsonValue catBoostJsonOptions;
    NJson::TJsonValue outputJsonOptions;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &catBoostJsonOptions, &outputJsonOptions);
    ConvertParamsToCanonicalFormat(trainDataProvider->MetaInfo, &catBoostJsonOptions);

    NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::CPU);
    catBoostOptions.Load(catBoostJsonOptions);
    catBoostOptions.SystemOptions->FileWithHosts->clear();
    catBoostOptions.DataProcessingOptions->AllowConstLabel = true;
    TLabelConverter labelConverter;

    NPar::TLocalExecutor* localExecutor = &NPar::LocalExecutor();
    if ((localExecutor->GetThreadCount() + 1) < numThreads) {
        localExecutor->RunAdditionalThreads(numThreads - 1);
    }

    if (localData.Rand == nullptr) {
        localData.Rand = MakeHolder<TRestorableFastRng64>(catBoostOptions.RandomSeed.Get() + hostId);
    }

    CATBOOST_DEBUG_LOG << "Create train data for worker " << hostId << "..." << Endl;
    localData.TrainData = GetTrainingData(
        std::move(pools),
        /*borders*/ Nothing(), // borders are already loaded to quantizedFeaturesInfo
        /*ensureConsecutiveIfDenseLearnFeaturesDataForCpu*/ true,
        /*allowWriteFiles*/ false,
        /*tmpDir*/ TString(), // does not matter, because allowWritingFiles == false
        quantizedFeaturesInfo,
        &catBoostOptions,
        &labelConverter,
        localExecutor,
        localData.Rand.Get()
    );

    if (trainEstimatedDataProvider) {
        CATBOOST_DEBUG_LOG << "Create precomputed train data for worker " << hostId << "..." << Endl;

        localData.PrecomputedSingleOnlineCtrDataForSingleFold.ConstructInPlace();
        localData.PrecomputedSingleOnlineCtrDataForSingleFold->Meta
            = TPrecomputedOnlineCtrMetaData::DeserializeFromJson(
                precomputedOnlineCtrMetaDataAsJsonString
              );
        localData.PrecomputedSingleOnlineCtrDataForSingleFold->DataProviders.Learn
            = dynamic_cast<TQuantizedObjectsDataProvider*>(
                trainEstimatedDataProvider->ObjectsData.Get()
              );
        CB_ENSURE_INTERNAL(
            localData.PrecomputedSingleOnlineCtrDataForSingleFold->DataProviders.Learn,
            "Precomputed data: Non-quantized objects data specified"
        );
    }
}


void RunWorkerWrapper(i32 numThreads, i32 nodePort) throw (yexception) {
    RunWorker(SafeIntegerCast<ui32>(numThreads), SafeIntegerCast<ui32>(nodePort));
}

