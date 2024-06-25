#pragma once

#include "implementation_type_enum.h"

#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/analytical_mode_params.h>

#include <library/cpp/getopt/small/last_getopt_opts.h>
#include <library/cpp/object_factory/object_factory.h>


namespace NCB {
    class IModeCalcImplementation {
    public:
        virtual int mode_calc(int argc, const char** argv) const = 0;
        virtual ~IModeCalcImplementation() = default;
    };

    using TModeCalcImplementationFactory = NObjectFactory::TParametrizedObjectFactory<IModeCalcImplementation, EImplementationType>;

    void PrepareCalcModeParamsParser(
        NCB::TAnalyticalModeCommonParams* paramsPtr,
        size_t* iterationsLimitPtr,
        size_t* evalPeriodPtr,
        size_t * virtualEnsemblesCountPtr,
        NLastGetopt::TOpts* parserPtr);

    void ReadModelAndUpdateParams(
        NCB::TAnalyticalModeCommonParams* paramsPtr,
        size_t* iterationsLimitPtr,
        size_t* evalPeriodPtr,
        TVector<TFullModel>* allModelsPtr);

    void CalcModelSingleHost(
        const NCB::TAnalyticalModeCommonParams& params,
        size_t iterationsLimit,
        size_t evalPeriod,
        size_t virtualEnsemblesCount,
        TVector<TFullModel>&& model);

    TEvalResult Apply(
        const TFullModel& model,
        const NCB::TDataProvider& dataset,
        size_t begin,
        size_t end,
        size_t evalPeriod,
        size_t virtualEnsemblesCount,
        bool isUncertaintyPrediction,
        NPar::ILocalExecutor* executor);

    TVector<TEvalResult> ApplyAllModels(
        const TVector<TFullModel>& allModels,
        const NCB::TDataProvider& dataset,
        size_t begin,
        size_t end,
        size_t evalPeriod,
        size_t virtualEnsemblesCount,
        bool isUncertaintyPrediction,
        NPar::ILocalExecutor* executor);

    TEvalColumnsInfo CreateEvalColumnsInfo(
        const TVector<TFullModel>& allModels,
        const TDataProviderPtr datasetPart,
        ui32 iterationsLimit,
        ui32 evalPeriod,
        ui32 virtualEnsemblesCount,
        bool isUncertaintyPrediction,
        NPar::TLocalExecutor* localExecutor);

    void AddBlendedApprox(
        const TString& blendingExpression,
        NCB::TEvalColumnsInfo* evalColumnsInfo);
}
