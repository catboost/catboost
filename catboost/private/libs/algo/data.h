#pragma once

#include <catboost/private/libs/options/loss_description.h>

#include <catboost/libs/data/data_provider.h>

#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


class TLabelConverter;
struct TRestorableFastRng64;

namespace NCatboostOptions {
    class TCatBoostOptions;
    struct TPoolLoadParams;
}

namespace NPar {
    class ILocalExecutor;
}


namespace NCB {
    struct TPathWithScheme;

    TVector<NCatboostOptions::TLossDescription> GetMetricDescriptions(
        const NCatboostOptions::TCatBoostOptions& params);


    TTrainingDataProviderPtr GetTrainingData(
        TDataProviderPtr srcData,
        bool dataCanBeEmpty,
        bool isLearnData,
        TStringBuf datasetName,
        const TMaybe<TString>& bordersFile,
        bool unloadCatFeaturePerfectHashFromRam,
        bool ensureConsecutiveIfDenseFeaturesDataForCpu,
        const TString& tmpDir, // unused if unloadCatFeaturePerfectHashFromRam == false
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo, // can be nullptr, then create it
        NCatboostOptions::TCatBoostOptions* params,
        TLabelConverter* labelConverter,
        TMaybe<float>* targetBorder,
        NPar::ILocalExecutor* localExecutor,
        TRestorableFastRng64* rand,
        TMaybe<TFullModel*> initModel = Nothing());

    TTrainingDataProviders GetTrainingData(
        TDataProviders srcData,
        bool dataCanBeEmpty,
        const TMaybe<TString>& bordersFile, // load borders from it if specified
        bool ensureConsecutiveIfDenseLearnFeaturesDataForCpu,
        bool allowWriteFiles,
        const TString& tmpDir, // unused if allowWriteFiles == false
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo, // can be nullptr, then create it
        NCatboostOptions::TCatBoostOptions* params,
        TLabelConverter* labelConverter,
        NPar::ILocalExecutor* localExecutor,
        TRestorableFastRng64* rand,
        TMaybe<TFullModel*> initModel = Nothing());

    TTrainingDataProviders MakeFeatureSubsetTrainingData(
        const TVector<ui32>& ignoredFeatures,
        const NCB::TTrainingDataProviders& trainingData
    );

    bool HaveFeaturesInMemory(
        const NCatboostOptions::TCatBoostOptions& catBoostOptions,
        const TMaybe<TPathWithScheme>& maybePathWithScheme
    );

    void EnsureObjectsDataIsConsecutiveIfQuantized(
        ui64 cpuUsedRamLimit,
        NPar::ILocalExecutor* localExecutor,
        TDataProviderPtr* dataProvider
    );
}
