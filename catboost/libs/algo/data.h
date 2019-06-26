#pragma once

#include <catboost/libs/data_new/data_provider.h>

#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>


class TLabelConverter;
struct TRestorableFastRng64;

namespace NCatboostOptions {
    class TCatBoostOptions;
}

namespace NPar {
    class TLocalExecutor;
}


namespace NCB {

    TTrainingDataProviderPtr GetTrainingData(
        TDataProviderPtr srcData,
        bool isLearnData,
        TStringBuf datasetName,
        const TMaybe<TString>& bordersFile,
        bool unloadCatFeaturePerfectHashFromRamIfPossible,
        bool ensureConsecutiveFeaturesDataForCpu,
        bool allowWriteFiles,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo, // can be nullptr, then create it
        NCatboostOptions::TCatBoostOptions* params,
        TLabelConverter* labelConverter,
        TMaybe<float>* targetBorder,
        NPar::TLocalExecutor* localExecutor,
        TRestorableFastRng64* rand);

    TTrainingDataProviders GetTrainingData(
        TDataProviders srcData,
        const TMaybe<TString>& bordersFile, // load borders from it if specified
        bool ensureConsecutiveLearnFeaturesDataForCpu,
        bool allowWriteFiles,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo, // can be nullptr, then create it
        NCatboostOptions::TCatBoostOptions* params,
        TLabelConverter* labelConverter,
        NPar::TLocalExecutor* localExecutor,
        TRestorableFastRng64* rand);

}
