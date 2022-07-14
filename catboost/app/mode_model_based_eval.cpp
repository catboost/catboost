#include <catboost/private/libs/app_helpers/bind_options.h>
#include "modes.h"

#include <catboost/private/libs/algo/helpers.h>
#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/plain_options_helper.h>
#include <catboost/private/libs/options/pool_metainfo_options.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/cpp/json/json_reader.h>

#include <util/generic/ptr.h>


using namespace NCB;


int mode_model_based_eval(int argc, const char* argv[]) {
    ConfigureMalloc();

    NCatboostOptions::TPoolLoadParams poolLoadParams;
    TString paramsFile;
    NJson::TJsonValue catBoostFlatJsonOptions;
    ParseModelBasedEvalCommandLine(argc, argv, &catBoostFlatJsonOptions, &paramsFile, &poolLoadParams);
    NJson::TJsonValue catBoostJsonOptions;
    NJson::TJsonValue outputOptionsJson;
    InitOptions(paramsFile, &catBoostJsonOptions, &outputOptionsJson);
    NCatboostOptions::LoadPoolMetaInfoOptions(poolLoadParams.PoolMetaInfoPath, &catBoostJsonOptions);
    ConvertIgnoredFeaturesFromStringToIndices(poolLoadParams, &catBoostFlatJsonOptions);
    ConvertFeaturesToEvaluateFromStringToIndices(poolLoadParams, &catBoostFlatJsonOptions);
    NCatboostOptions::PlainJsonToOptions(catBoostFlatJsonOptions, &catBoostJsonOptions, &outputOptionsJson);
    ConvertParamsToCanonicalFormat(poolLoadParams, &catBoostJsonOptions);
    CopyIgnoredFeaturesToPoolParams(catBoostJsonOptions, &poolLoadParams);
    NCatboostOptions::TOutputFilesOptions outputOptions;
    outputOptions.Load(outputOptionsJson);

    //check model based eval restrictions
    CB_ENSURE(NCatboostOptions::GetTaskType(catBoostJsonOptions) == ETaskType::GPU);
    const auto featuresToEvaluate = GetOptionFeaturesToEvaluate(catBoostJsonOptions);
    CB_ENSURE(!featuresToEvaluate.empty(), "Error: no features to evaluate");
    for (ui32 feature : featuresToEvaluate) {
        CB_ENSURE(Count(poolLoadParams.IgnoredFeatures, feature) == 0, "Error: feature " + ToString(feature) + " is ignored");
    }
    if (outputOptions.IsMetricPeriodSet() && outputOptions.GetMetricPeriod() > 1) {
        CATBOOST_WARNING_LOG << "Warning: metric_period is ignored because "
            "model-based feature evaluation needs metric values on each iteration" << Endl;
    }
    outputOptions.SetMetricPeriod(1);

    ModelBasedEval(poolLoadParams, outputOptions, catBoostJsonOptions);

    return 0;
}
