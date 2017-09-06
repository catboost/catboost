#include <catboost/libs/data/pool.h>
#include <catboost/libs/algo/apply.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/algo/train_model.h>
#include <catboost/libs/algo/calc_fstr.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/logging/logging.h>

#if defined(SIZEOF_SIZE_T)
#undef SIZEOF_SIZE_T
#endif

#include <Rinternals.h>


#define R_API_BEGIN()                                               \
    SetCustomLoggingFunction([](const char* str, size_t len) {      \
        TString slicedStr(str, 0, len);                             \
        Rprintf("%s", slicedStr.c_str());                           \
    });                                                             \
    try {                                                           \


#define R_API_END()                                                 \
    } catch (std::exception& e) {                                   \
        error(e.what());                                            \
    }                                                               \
    RestoreOriginalLogger();                                        \


typedef TPool* TPoolHandle;
typedef std::unique_ptr<TPool> TPoolPtr;

typedef TFullModel* TFullModelHandle;
typedef std::unique_ptr<TFullModel> TFullModelPtr;

template<typename T>
void _Finalizer(SEXP ext) {
    if (R_ExternalPtrAddr(ext) == NULL) return;
    delete reinterpret_cast<T>(R_ExternalPtrAddr(ext)); // delete allocated memory
    R_ClearExternalPtr(ext);
}

template<typename T>
static yvector<T> GetVectorFromSEXP(SEXP arg) {
    yvector<T> result(length(arg));
    for (size_t i = 0; i < result.size(); ++i) {
        switch (TYPEOF(arg)) {
            case INTSXP:
                result[i] = static_cast<T>(INTEGER(arg)[i]);
                break;
            case REALSXP:
                result[i] = static_cast<T>(REAL(arg)[i]);
                break;
            case LGLSXP:
                result[i] = static_cast<T>(LOGICAL(arg)[i]);
                break;
            default:
                Y_ENSURE(false, "unsupported vector type: int, real or logical is required");
        }
    }
    return result;
}

static NJson::TJsonValue LoadFitParams(SEXP fitParamsAsJson) {
    TString paramsStr(CHAR(asChar(fitParamsAsJson)));
    TStringInput is(paramsStr);
    NJson::TJsonValue result;
    NJson::ReadJsonTree(&is, &result);
    return result;
}

extern "C" {

SEXP CatBoostCreateFromFile_R(SEXP poolFileParam,
                              SEXP cdFileParam,
                              SEXP threadCountParam,
                              SEXP verboseParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolPtr poolPtr = std::make_unique<TPool>();
    ReadPool(CHAR(asChar(cdFileParam)),
             CHAR(asChar(poolFileParam)),
             asInteger(threadCountParam),
             asLogical(verboseParam),
             '\t',
             false,
             yvector<TString>(),
             poolPtr.get());
    result = PROTECT(R_MakeExternalPtr(poolPtr.get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TPoolHandle>, TRUE);
    poolPtr.release();
    R_API_END();
    UNPROTECT(1);
    return result;
}

SEXP CatBoostCreateFromMatrix_R(SEXP matrixParam,
                                SEXP targetParam,
                                SEXP weightParam,
                                SEXP baselineParam,
                                SEXP catFeaturesParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    SEXP dataDim = getAttrib(matrixParam, R_DimSymbol);
    size_t dataRows = static_cast<size_t>(INTEGER(dataDim)[0]);
    size_t dataColumns = static_cast<size_t>(INTEGER(dataDim)[1]);
    SEXP baselineDim = getAttrib(baselineParam, R_DimSymbol);
    size_t baselineRows = 0;
    size_t baselineColumns = 0;
    if (baselineParam != R_NilValue) {
        baselineRows = static_cast<size_t>(INTEGER(baselineDim)[0]);
        baselineColumns = static_cast<size_t>(INTEGER(baselineDim)[1]);
    }
    TPoolPtr poolPtr = std::make_unique<TPool>();
    poolPtr->CatFeatures = GetVectorFromSEXP<int>(catFeaturesParam);
    poolPtr->Docs.resize(dataRows);
    for (size_t i = 0; i < dataRows; ++i) {
        if (targetParam != R_NilValue) {
            poolPtr->Docs[i].Target = static_cast<float>(REAL(targetParam)[i]);
        }
        if (weightParam != R_NilValue) {
            poolPtr->Docs[i].Weight = static_cast<float>(REAL(weightParam)[i]);
        }
        if (baselineParam != R_NilValue) {
            poolPtr->Docs[i].Baseline.resize(baselineColumns);
            for (size_t j = 0; j < baselineColumns; ++j) {
                poolPtr->Docs[i].Baseline[j] = static_cast<float>(REAL(baselineParam)[i + baselineRows * j]);
            }
        }
        poolPtr->Docs[i].Factors.resize(dataColumns);
        for (size_t j = 0; j < dataColumns; ++j) {
            poolPtr->Docs[i].Factors[j] = static_cast<float>(REAL(matrixParam)[i + dataRows * j]);  // casting double to float
        }
    }
    result = PROTECT(R_MakeExternalPtr(poolPtr.get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TPoolHandle>, TRUE);
    poolPtr.release();
    R_API_END();
    UNPROTECT(1);
    return result;
}

SEXP CatBoostHashStrings_R(SEXP stringsParam) {
   SEXP result = PROTECT(allocVector(REALSXP, length(stringsParam)));
   for (int i = 0; i < length(stringsParam); ++i) {
       REAL(result)[i] = static_cast<double>(ConvertCatFeatureHashToFloat(CalcCatFeatureHash(TString(CHAR(STRING_ELT(stringsParam, i))))));
   }
   UNPROTECT(1);
   return result;
}

SEXP CatBoostPoolNumRow_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    result = ScalarInteger(static_cast<int>(pool->Docs.size()));
    R_API_END();
    return result;
}

SEXP CatBoostPoolNumCol_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    result = ScalarInteger(0);
    if (!pool->Docs.empty()) {
        result = ScalarInteger(static_cast<int>(pool->Docs[0].Factors.size()));
    }
    R_API_END();
    return result;
}

SEXP CatBoostPoolNumTrees_R(SEXP modelParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    result = ScalarInteger(static_cast<int>(model->LeafValues.size()));
    R_API_END();
    return result;
}

SEXP CatBoostPoolHead_R(SEXP poolParam, SEXP n) {
    SEXP result = NULL;
    size_t resultSize = 0;
    R_API_BEGIN();
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    resultSize = std::min(static_cast<size_t>(asInteger(n)), pool->Docs.size());
    result = PROTECT(allocVector(VECSXP, resultSize));
    for (size_t i = 0; i < resultSize; ++i) {
        SEXP row = PROTECT(allocVector(REALSXP, pool->Docs[i].Factors.size() + 2));
        REAL(row)[0] = pool->Docs[i].Target;
        REAL(row)[1] = pool->Docs[i].Weight;
        for (size_t j = 0; j < pool->Docs[i].Factors.size(); ++j) {
            REAL(row)[j + 2] = pool->Docs[i].Factors[j];
        }
        SET_VECTOR_ELT(result, i, row);
    }
    R_API_END();
    UNPROTECT(resultSize + 1);
    return result;
}

SEXP CatBoostFit_R(SEXP learnPoolParam, SEXP testPoolParam, SEXP fitParamsAsJsonParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle learnPool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(learnPoolParam));
    auto fitParams = LoadFitParams(fitParamsAsJsonParam);
    TFullModelPtr modelPtr = std::make_unique<TFullModel>();
    yvector<yvector<double>> testApprox;
    if (testPoolParam != R_NilValue) {
        TPoolHandle testPool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(testPoolParam));
        TrainModel(fitParams, Nothing(), Nothing(), *learnPool, *testPool, "", modelPtr.get(), &testApprox);
    }
    else {
        TrainModel(fitParams, Nothing(), Nothing(), *learnPool, TPool(), "", modelPtr.get(), &testApprox);
    }
    result = PROTECT(R_MakeExternalPtr(modelPtr.get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TFullModelHandle>, TRUE);
    modelPtr.release();
    R_API_END();
    UNPROTECT(1);
    return result;
}

SEXP CatBoostOutputModel_R(SEXP modelParam, SEXP fileParam) {
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    OutputModel(*model, CHAR(asChar(fileParam)));
    R_API_END();
    return ScalarLogical(1);
}

SEXP CatBoostReadModel_R(SEXP fileParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelPtr modelPtr = std::make_unique<TFullModel>();
    ReadModel(CHAR(asChar(fileParam))).Swap(*modelPtr);
    result = PROTECT(R_MakeExternalPtr(modelPtr.get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TFullModelHandle>, TRUE);
    modelPtr.release();
    R_API_END();
    UNPROTECT(1);
    return result;
}

SEXP CatBoostPredictMulti_R(SEXP modelParam, SEXP poolParam, SEXP verboseParam,
                            SEXP typeParam, SEXP treeCountLimitParam, SEXP threadCountParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    EPredictionType predictionType;
    CB_ENSURE(TryFromString<EPredictionType>(CHAR(asChar(typeParam)), predictionType),
              "unsupported prediction type: 'Probability', 'Class' or 'RawFormulaVal' was expected");
    yvector<yvector<double>> prediction = ApplyModelMulti(*model, *pool,
                                                          asLogical(verboseParam),
                                                          predictionType,
                                                          0,
                                                          asInteger(treeCountLimitParam),
                                                          asInteger(threadCountParam));
    size_t predictionSize = prediction.size() * pool->Docs.size();
    result = PROTECT(allocVector(REALSXP, predictionSize));
    for (size_t i = 0, k = 0; i < pool->Docs.size(); ++i) {
        for (size_t j = 0; j < prediction.size(); ++j) {
            REAL(result)[k++] = prediction[j][i];
        }
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

SEXP CatBoostGetModelParams_R(SEXP modelParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    result = PROTECT(mkString(model->ModelInfo.at("params").c_str()));
    R_API_END();
    UNPROTECT(1);
    return result;
}

SEXP CatBoostCalcRegularFeatureEffect_R(SEXP modelParam, SEXP poolParam, SEXP threadCountParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    yvector<double> effect = CalcRegularFeatureEffect(*model, *pool, asInteger(threadCountParam));
    result = PROTECT(allocVector(REALSXP, effect.size()));
    for (size_t i = 0; i < effect.size(); ++i) {
        REAL(result)[i] = effect[i];
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}
}
