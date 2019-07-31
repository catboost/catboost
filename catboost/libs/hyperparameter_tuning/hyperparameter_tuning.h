#pragma once

#include <catboost/libs/algo/custom_objective_descriptor.h>
#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/options/split_params.h>
#include <catboost/libs/train_lib/cross_validation.h>

#include <library/json/json_value.h>

#include <util/generic/hash.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

#include <util/system/types.h>

using ::TCrossValidationParams;

namespace NCB {

    struct TCustomRandomDistributionGenerator {
        using TEvalFuncPtr = double (*)(void* customData);

        // Any custom data required by EvalFunc
        void* CustomData = nullptr;

        // Pointer to function that generate random values
        TEvalFuncPtr EvalFunc = nullptr;
    };

    struct TBestOptionValuesWithCvResult {
    public:
        TVector<TCVResult> CvResult;
        THashMap<TString, bool> BoolOptions;
        THashMap<TString, int> IntOptions;
        THashMap<TString, ui32> UIntOptions;
        THashMap<TString, double> DoubleOptions;
        THashMap<TString, TString> StringOptions;
    public:
        void SetOptionsFromJson(
            const THashMap<TString, NJson::TJsonValue>& options,
            const TVector<TString>& optionsNames);
    };

    void GridSearch(
        const NJson::TJsonValue& gridJsonValues,
        const NJson::TJsonValue& modelJsonParams,
        const TTrainTestSplitParams& trainTestSplitParams,
        const TCrossValidationParams& cvParams,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        TDataProviderPtr data,
        TBestOptionValuesWithCvResult* bestOptionValuesWithCvResult,
        bool isSearchUsingTrainTestSplit = true,
        bool returnCvStat = true,
        int verbose = 1);

    void RandomizedSearch(
        ui32 numberOfTries,
        const THashMap<TString, TCustomRandomDistributionGenerator>& randDistGenerators,
        const NJson::TJsonValue& gridJsonValues,
        const NJson::TJsonValue& modelJsonParams,
        const TTrainTestSplitParams& trainTestSplitParams,
        const TCrossValidationParams& cvParams,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        TDataProviderPtr data,
        TBestOptionValuesWithCvResult* bestOptionValuesWithCvResult,
        bool isSearchUsingTrainTestSplit = true,
        bool returnCvStat = true,
        int verbose = 1);
}
