#pragma once

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/options/enums.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/file.h>

#include <functional>


struct TRestorableFastRng64;

namespace NCatboostOptions {
    class TCatBoostOptions;
    class TOutputFilesOptions;
}

namespace NJson {
    class TJsonValue;
}

namespace NPar {
    class TLocalExecutor;
}


void CheckConsistency(const NCB::TTrainingDataProviders& data);

void UpdateUndefinedRandomSeed(
    ETaskType taskType,
    const NCatboostOptions::TOutputFilesOptions& outputOptions,
    NJson::TJsonValue* updatedJsonParams,
    std::function<void(TIFStream*, TString&)> paramsLoader
);

void UpdateUndefinedClassNames(
    const TVector<TString>& classNames,
    NJson::TJsonValue* updatedJsonParams
);


NCB::TDataProviderPtr ReorderByTimestampLearnDataIfNeeded(
    const NCatboostOptions::TCatBoostOptions& catBoostOptions,
    NCB::TDataProviderPtr learnData,
    NPar::TLocalExecutor* localExecutor
);

NCB::TDataProviderPtr ShuffleLearnDataIfNeeded(
    const NCatboostOptions::TCatBoostOptions& catBoostOptions,
    NCB::TDataProviderPtr learnData,
    NPar::TLocalExecutor* localExecutor,
    TRestorableFastRng64* rand
);
