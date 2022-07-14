#include <R.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>
#include "catboostr.h"

static const R_CallMethodDef CallEntries[] = {
    {"CatBoostCalcRegularFeatureEffect_R",    (DL_FUNC) &CatBoostCalcRegularFeatureEffect_R,     4},
    {"CatBoostCreateFromFile_R",              (DL_FUNC) &CatBoostCreateFromFile_R,               9},
    {"CatBoostCreateFromMatrix_R",            (DL_FUNC) &CatBoostCreateFromMatrix_R,            13},
    {"CatBoostCV_R",                          (DL_FUNC) &CatBoostCV_R,                           7},
    {"CatBoostDeserializeModel_R",            (DL_FUNC) &CatBoostDeserializeModel_R,             1},
    {"CatBoostDropUnusedFeaturesFromModel_R", (DL_FUNC) &CatBoostDropUnusedFeaturesFromModel_R,  1},
    {"CatBoostEvalMetrics_R",                 (DL_FUNC) &CatBoostEvalMetrics_R,                  9},
    {"CatBoostEvaluateObjectImportances_R",   (DL_FUNC) &CatBoostEvaluateObjectImportances_R,    7},
    {"CatBoostFit_R",                         (DL_FUNC) &CatBoostFit_R,                          3},
    {"CatBoostGetModelParams_R",              (DL_FUNC) &CatBoostGetModelParams_R,               1},
    {"CatBoostGetNumTrees_R",                 (DL_FUNC) &CatBoostGetNumTrees_R,                  1},
    {"CatBoostGetPlainParams_R",              (DL_FUNC) &CatBoostGetPlainParams_R,               1},
    {"CatBoostHashStrings_R",                 (DL_FUNC) &CatBoostHashStrings_R,                  1},
    {"CatBoostIsGroupwiseMetric_R",           (DL_FUNC) &CatBoostIsGroupwiseMetric_R,            1},
    {"CatBoostIsNullHandle_R",                (DL_FUNC) &CatBoostIsNullHandle_R,                 1},
    {"CatBoostIsOblivious_R",                 (DL_FUNC) &CatBoostIsOblivious_R,                  1},
    {"CatBoostOutputModel_R",                 (DL_FUNC) &CatBoostOutputModel_R,                  5},
    {"CatBoostPoolNumCol_R",                  (DL_FUNC) &CatBoostPoolNumCol_R,                   1},
    {"CatBoostPoolNumRow_R",                  (DL_FUNC) &CatBoostPoolNumRow_R,                   1},
    {"CatBoostPoolSlice_R",                   (DL_FUNC) &CatBoostPoolSlice_R,                    3},
    {"CatBoostPredictMulti_R",                (DL_FUNC) &CatBoostPredictMulti_R,                 7},
    {"CatBoostPredictVirtualEnsembles_R",     (DL_FUNC) &CatBoostPredictVirtualEnsembles_R,      7},
    {"CatBoostPrepareEval_R",                 (DL_FUNC) &CatBoostPrepareEval_R,                  5},
    {"CatBoostReadModel_R",                   (DL_FUNC) &CatBoostReadModel_R,                    2},
    {"CatBoostSerializeModel_R",              (DL_FUNC) &CatBoostSerializeModel_R,               1},
    {"CatBoostShrinkModel_R",                 (DL_FUNC) &CatBoostShrinkModel_R,                  3},
    {"CatBoostSumModels_R",                   (DL_FUNC) &CatBoostSumModels_R,                    3},
    {NULL, NULL, 0}
};

#if defined(_WIN32)
__declspec(dllexport)
#endif
void R_init_libcatboostr(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
