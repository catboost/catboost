#include "worker.h"

#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/distributed/data_types.h>
#include <catboost/private/libs/options/plain_options_helper.h>

#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/logging/logging.h>

#include <library/cpp/binsaver/util_stream_io.h>
#include <library/cpp/json/json_reader.h>

#include <util/generic/cast.h>
#include <util/generic/ptr.h>
#include <util/generic/xrange.h>

using namespace NCB;


i64 GetPartitionTotalObjectCount(const TVector<TDataProviderPtr>& trainDataProviders) {
    i64 result = 0;
    for (const auto& trainDataProvider : trainDataProviders) {
        result += SafeIntegerCast<i64>(trainDataProvider->GetObjectCount());
    }
    return result;
}


void CreateTrainingDataForWorker(
    i32 hostId,
    i32 numThreads,
    const TString& plainJsonParamsAsString,
    const TVector<i8>& serializedLabelConverter,
    const TVector<TDataProviderPtr>& trainDataProviders,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    const TVector<TDataProviderPtr>& trainEstimatedDataProviders, // can be empty
    const TString& precomputedOnlineCtrMetaDataAsJsonString
) {
    CB_ENSURE(numThreads >= 1, "Non-positive number of threads specified");
    CB_ENSURE_INTERNAL(
        trainDataProviders.size() >= 1,
        "trainDataProviders must have at least learn data provider"
    );

    auto& localData = NCatboostDistributed::TLocalTensorSearchData::GetRef();
    localData = NCatboostDistributed::TLocalTensorSearchData();

    TDataProviders pools;
    pools.Learn = trainDataProviders[0];
    pools.Test.assign(trainDataProviders.begin() + 1, trainDataProviders.end());

    NJson::TJsonValue plainJsonParams;
    NJson::ReadJsonTree(plainJsonParamsAsString, &plainJsonParams, /*throwOnError*/ true);
    ConvertIgnoredFeaturesFromStringToIndices(trainDataProviders[0]->MetaInfo, &plainJsonParams);

    NJson::TJsonValue catBoostJsonOptions;
    NJson::TJsonValue outputJsonOptions;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &catBoostJsonOptions, &outputJsonOptions);
    ConvertParamsToCanonicalFormat(trainDataProviders[0]->MetaInfo, &catBoostJsonOptions);

    NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::CPU);
    catBoostOptions.Load(catBoostJsonOptions);
    catBoostOptions.SystemOptions->FileWithHosts->clear();
    catBoostOptions.DataProcessingOptions->AllowConstLabel = true;

    TLabelConverter labelConverter;
    if (!serializedLabelConverter.empty()) {
        TMemoryInput in(serializedLabelConverter.data(), serializedLabelConverter.size());
        SerializeFromStream(in, labelConverter);
    }

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
        /*trainingDataCanBeEmpty*/ true,
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

    if (!trainEstimatedDataProviders.empty()) {
        CATBOOST_DEBUG_LOG << "Create precomputed data for worker " << hostId << "..." << Endl;

        localData.PrecomputedSingleOnlineCtrDataForSingleFold.ConstructInPlace();
        localData.PrecomputedSingleOnlineCtrDataForSingleFold->Meta
            = TPrecomputedOnlineCtrMetaData::DeserializeFromJson(
                precomputedOnlineCtrMetaDataAsJsonString
              );
        localData.PrecomputedSingleOnlineCtrDataForSingleFold->DataProviders.Learn
            = dynamic_cast<TQuantizedObjectsDataProvider*>(
                trainEstimatedDataProviders[0]->ObjectsData.Get()
              );
        CB_ENSURE_INTERNAL(
            localData.PrecomputedSingleOnlineCtrDataForSingleFold->DataProviders.Learn,
            "Precomputed learn data: Non-quantized objects data specified"
        );
        for (auto testIdx : xrange(trainEstimatedDataProviders.size() - 1)) {
            localData.PrecomputedSingleOnlineCtrDataForSingleFold->DataProviders.Test.push_back(
                dynamic_cast<TQuantizedObjectsDataProvider*>(
                    trainEstimatedDataProviders[testIdx + 1]->ObjectsData.Get()
                )
            );
            CB_ENSURE_INTERNAL(
                localData.PrecomputedSingleOnlineCtrDataForSingleFold->DataProviders.Test.back(),
                "Precomputed test data # " << testIdx << ": Non-quantized objects data specified"
            );
        }
    }
}

