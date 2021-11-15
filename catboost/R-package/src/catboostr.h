#pragma once
#include <Rinternals.h>

#if defined(_WIN32)
#define EXPORT_FUNCTION __declspec(dllexport) SEXP
#else
#define EXPORT_FUNCTION SEXP
#endif

#if defined(__cplusplus)
extern "C" {
#endif


EXPORT_FUNCTION CatBoostCreateFromFile_R(
    SEXP poolFileParam,
    SEXP cdFileParam,
    SEXP pairsFileParam,
    SEXP featureNamesFileParam,
    SEXP delimiterParam,
    SEXP numVectorDelimiterParam,
    SEXP hasHeaderParam,
    SEXP threadCountParam,
    SEXP verboseParam
);

EXPORT_FUNCTION CatBoostCreateFromMatrix_R(
    SEXP floatAndCatMatrixParam,
    SEXP targetParam,
    SEXP catFeaturesIndicesParam,
    SEXP textMatrixParam,
    SEXP textFeaturesIndicesParam,
    SEXP pairsParam,
    SEXP weightParam,
    SEXP groupIdParam,
    SEXP groupWeightParam,
    SEXP subgroupIdParam,
    SEXP pairsWeightParam,
    SEXP baselineParam,
    SEXP featureNamesParam
);

EXPORT_FUNCTION CatBoostHashStrings_R(SEXP stringsParam);

EXPORT_FUNCTION CatBoostPoolNumRow_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostPoolNumCol_R(SEXP poolParam);

EXPORT_FUNCTION CatBoostGetNumTrees_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostPoolNumTrees_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostIsOblivious_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostIsGroupwiseMetric_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostPoolSlice_R(
    SEXP poolParam,
    SEXP sizeParam,
    SEXP offsetParam
);

EXPORT_FUNCTION CatBoostFit_R(
    SEXP learnPoolParam,
    SEXP testPoolParam,
    SEXP fitParamsAsJsonParam
);

EXPORT_FUNCTION CatBoostSumModels_R(
    SEXP modelsParam,
    SEXP weightsParam,
    SEXP ctrMergePolicyParam
);

EXPORT_FUNCTION CatBoostCV_R(
    SEXP fitParamsAsJsonParam,
    SEXP poolParam,
    SEXP foldCountParam,
    SEXP typeParam,
    SEXP partitionRandomSeedParam,
    SEXP shuffleParam,
    SEXP stratifiedParam
);

EXPORT_FUNCTION CatBoostOutputModel_R(
    SEXP modelParam,
    SEXP fileParam,
    SEXP formatParam,
    SEXP exportParametersParam,
    SEXP poolParam
);

EXPORT_FUNCTION CatBoostReadModel_R(SEXP fileParam, SEXP formatParam);

EXPORT_FUNCTION CatBoostSerializeModel_R(SEXP handleParam);

EXPORT_FUNCTION CatBoostDeserializeModel_R(SEXP rawParam);

EXPORT_FUNCTION CatBoostPredictMulti_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP verboseParam,
    SEXP typeParam,
    SEXP treeCountStartParam,
    SEXP treeCountEndParam,
    SEXP threadCountParam
);

EXPORT_FUNCTION CatBoostPredictMulti_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP verboseParam,
    SEXP typeParam,
    SEXP treeCountStartParam,
    SEXP treeCountEndParam,
    SEXP threadCountParam
);

EXPORT_FUNCTION CatBoostPrepareEval_R(
    SEXP approxParam,
    SEXP typeParam,
    SEXP lossFunctionName,
    SEXP columnCountParam,
    SEXP threadCountParam
);

EXPORT_FUNCTION CatBoostPredictVirtualEnsembles_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP verboseParam,
    SEXP typeParam,
    SEXP treeCountEndParam,
    SEXP virtualEnsemblesCountParam,
    SEXP threadCountParam
);

EXPORT_FUNCTION CatBoostShrinkModel_R(
    SEXP modelParam,
    SEXP treeCountStartParam,
    SEXP treeCountEndParam
);

EXPORT_FUNCTION CatBoostDropUnusedFeaturesFromModel_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostGetModelParams_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostGetPlainParams_R(SEXP modelParam);

EXPORT_FUNCTION CatBoostCalcRegularFeatureEffect_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP fstrTypeParam,
    SEXP threadCountParam
);

EXPORT_FUNCTION CatBoostEvaluateObjectImportances_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP trainPoolParam,
    SEXP topSizeParam,
    SEXP ostrTypeParam,
    SEXP updateMethodParam,
    SEXP threadCountParam
);

EXPORT_FUNCTION CatBoostIsNullHandle_R(SEXP handleParam);


EXPORT_FUNCTION CatBoostEvalMetrics_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP metricsParam,
    SEXP treeCountStartParam,
    SEXP treeCountEndParam,
    SEXP evalPeriodParam,
    SEXP threadCountParam,
    SEXP tmpDirParam,
    SEXP resultDirParam
);



#if defined(__cplusplus)
}
#endif
