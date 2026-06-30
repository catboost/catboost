#pragma once

#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/split_params.h>
#include <catboost/libs/train_lib/cross_validation.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/fwd.h>
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
        NJson::TJsonValue BestParams;
    public:
        TBestOptionValuesWithCvResult()
            : BestParams(NJson::JSON_MAP)
        {}

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
        TMetricsAndTimeLeftHistory* trainTestResult,
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
        TMetricsAndTimeLeftHistory* trainTestResult,
        bool isSearchUsingTrainTestSplit = true,
        bool returnCvStat = true,
        int verbose = 1);
}
