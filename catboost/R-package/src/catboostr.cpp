#include <catboost/libs/data/pool.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/algo/apply.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/libs/train_lib/cross_validation.h>
#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/fstr/calc_fstr.h>
#include <catboost/libs/documents_importance/docs_importance.h>
#include <catboost/libs/documents_importance/enums.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/options/cross_validation_params.h>

#include <util/generic/mem_copy.h>
#include <util/generic/singleton.h>
#include <util/system/info.h>

#if defined(SIZEOF_SIZE_T)
#undef SIZEOF_SIZE_T
#endif

#include <Rinternals.h>


#define R_API_BEGIN()                                               \
    auto loggingFunc = [](const char* str, size_t len) {            \
        TString slicedStr(str, 0, len);                             \
        Rprintf("%s", slicedStr.c_str());                           \
    };                                                               \
    SetCustomLoggingFunction(loggingFunc, loggingFunc);             \
    *Singleton<TRPackageInitializer>();                             \
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

class TRPackageInitializer {
    Y_DECLARE_SINGLETON_FRIEND();
    TRPackageInitializer() {
        ConfigureMalloc();
    }
};

template <typename T>
void _Finalizer(SEXP ext) {
    if (R_ExternalPtrAddr(ext) == NULL) return;
    delete reinterpret_cast<T>(R_ExternalPtrAddr(ext)); // delete allocated memory
    R_ClearExternalPtr(ext);
}

template <typename T>
static TVector<T> GetVectorFromSEXP(SEXP arg) {
    TVector<T> result(length(arg));
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

static int UpdateThreadCount(int threadCount) {
    if (threadCount == -1) {
        threadCount = NSystemInfo::CachedNumberOfCpus();
    }
    return threadCount;
}

extern "C" {

SEXP CatBoostCreateFromFile_R(SEXP poolFileParam,
                              SEXP cdFileParam,
                              SEXP pairsFileParam,
                              SEXP delimiterParam,
                              SEXP hasHeaderParam,
                              SEXP threadCountParam,
                              SEXP verboseParam) {
    SEXP result = NULL;
    R_API_BEGIN();

    NCatboostOptions::TDsvPoolFormatParams dsvPoolFormatParams;
    dsvPoolFormatParams.Format =
        NCB::TDsvFormatOptions{static_cast<bool>(asLogical(hasHeaderParam)),
                               CHAR(asChar(delimiterParam))[0]};

    TStringBuf cdPathWithScheme(CHAR(asChar(cdFileParam)));
    if (!cdPathWithScheme.empty()) {
        dsvPoolFormatParams.CdFilePath = NCB::TPathWithScheme(cdPathWithScheme, "dsv");
    }

    TStringBuf pairsPathWithScheme(CHAR(asChar(pairsFileParam)));

    TPoolPtr poolPtr = std::make_unique<TPool>();
    NCB::ReadPool(NCB::TPathWithScheme(CHAR(asChar(poolFileParam)), "dsv"),
                  !pairsPathWithScheme.empty() ?
                      NCB::TPathWithScheme(pairsPathWithScheme, "dsv") : NCB::TPathWithScheme(),
                  /*groupWeightsFilePath=*/NCB::TPathWithScheme(),
                  dsvPoolFormatParams,
                  TVector<int>(),
                  UpdateThreadCount(asInteger(threadCountParam)),
                  asLogical(verboseParam),
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
                                SEXP catFeaturesParam,
                                SEXP pairsParam,
                                SEXP weightParam,
                                SEXP groupIdParam,
                                SEXP groupWeightParam,
                                SEXP subgroupIdParam,
                                SEXP pairsWeightParam,
                                SEXP baselineParam,
                                SEXP featureNamesParam) {
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
    CB_ENSURE(weightParam == R_NilValue || groupWeightParam == R_NilValue,
              "Pool must have either Weight column or GroupWeight column");

    TPoolPtr poolPtr = std::make_unique<TPool>();

    poolPtr->MetaInfo.FeatureCount = dataColumns;
    poolPtr->MetaInfo.BaselineCount = baselineColumns;
    poolPtr->MetaInfo.HasGroupId = groupIdParam != R_NilValue;
    poolPtr->MetaInfo.HasGroupWeight = groupWeightParam != R_NilValue;
    poolPtr->MetaInfo.HasSubgroupIds = subgroupIdParam != R_NilValue;
    poolPtr->MetaInfo.HasWeights = weightParam != R_NilValue || groupWeightParam != R_NilValue;

    poolPtr->CatFeatures = GetVectorFromSEXP<int>(catFeaturesParam);

    poolPtr->Docs.Resize(
        dataRows,
        dataColumns,
        baselineColumns,
        poolPtr->MetaInfo.HasGroupId,
        poolPtr->MetaInfo.HasSubgroupIds
    );
    for (size_t i = 0; i < dataRows; ++i) {
        if (targetParam != R_NilValue) {
            poolPtr->Docs.Target[i] = static_cast<float>(REAL(targetParam)[i]);
        }
        if (weightParam != R_NilValue) {
            poolPtr->Docs.Weight[i] = static_cast<float>(REAL(weightParam)[i]);
        }
        if (poolPtr->MetaInfo.HasGroupId) {
            poolPtr->Docs.QueryId[i] = static_cast<uint32_t>(INTEGER(groupIdParam)[i]);
        }
        if (groupWeightParam != R_NilValue) {
            poolPtr->Docs.Weight[i] = static_cast<float>(REAL(groupWeightParam)[i]);
        }
        if (poolPtr->MetaInfo.HasSubgroupIds) {
            poolPtr->Docs.SubgroupId[i] = static_cast<uint32_t>(INTEGER(subgroupIdParam)[i]);
        }
        if (baselineParam != R_NilValue) {
            for (size_t j = 0; j < baselineColumns; ++j) {
                poolPtr->Docs.Baseline[j][i] = REAL(baselineParam)[i + baselineRows * j];
            }
        }
        for (size_t j = 0; j < dataColumns; ++j) {
            poolPtr->Docs.Factors[j][i] = static_cast<float>(REAL(matrixParam)[i + dataRows * j]);  // casting double to float
        }
    }
    if (pairsParam != R_NilValue) {
        size_t pairsCount = static_cast<size_t>(INTEGER(getAttrib(pairsParam, R_DimSymbol))[0]);
        for (size_t i = 0; i < pairsCount; ++i) {
            float weight = 1;
            if (pairsWeightParam != R_NilValue) {
                weight = static_cast<float>(REAL(pairsWeightParam)[i]);
            }
            poolPtr->Pairs.emplace_back(
                static_cast<int>(INTEGER(pairsParam)[i + pairsCount * 0]),
                static_cast<int>(INTEGER(pairsParam)[i + pairsCount * 1]),
                weight
            );
        }
    }
    if (featureNamesParam != R_NilValue) {
        for (size_t i = 0; i < dataColumns; ++i) {
            poolPtr->FeatureId.push_back(CHAR(asChar(VECTOR_ELT(featureNamesParam, i))));
        }
    } else {
        poolPtr->FeatureId.assign(dataColumns, TString());
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
    result = ScalarInteger(static_cast<int>(pool->Docs.GetDocCount()));
    R_API_END();
    return result;
}

SEXP CatBoostPoolNumCol_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    result = ScalarInteger(0);
    if (pool->Docs.GetDocCount() != 0) {
        result = ScalarInteger(static_cast<int>(pool->Docs.GetEffectiveFactorCount()));
    }
    R_API_END();
    return result;
}

SEXP CatBoostGetNumTrees_R(SEXP modelParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    result = ScalarInteger(static_cast<int>(model->ObliviousTrees.GetTreeCount()));
    R_API_END();
    return result;
}

// TODO(dbakshee): remove this backward compatibility gag in v0.11
SEXP CatBoostPoolNumTrees_R(SEXP modelParam) {
    return CatBoostGetNumTrees_R(modelParam);
}

SEXP CatBoostPoolSlice_R(SEXP poolParam, SEXP sizeParam, SEXP offsetParam) {
    SEXP result = NULL;
    size_t size, offset;
    R_API_BEGIN();
    size = static_cast<size_t>(asInteger(sizeParam));
    offset = static_cast<size_t>(asInteger(offsetParam));
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    result = PROTECT(allocVector(VECSXP, size));
    for (size_t i = offset; i < std::min(pool->Docs.GetDocCount(), offset + size); ++i) {
        SEXP row = PROTECT(allocVector(REALSXP, pool->Docs.GetEffectiveFactorCount() + 2));
        REAL(row)[0] = pool->Docs.Target[i];
        REAL(row)[1] = pool->Docs.Weight[i];
        for (int j = 0; j < pool->Docs.GetEffectiveFactorCount(); ++j) {
            REAL(row)[j + 2] = pool->Docs.Factors[j][i];
        }
        SET_VECTOR_ELT(result, i - offset, row);
    }
    R_API_END();
    UNPROTECT(size - offset + 1);
    return result;
}

SEXP CatBoostFit_R(SEXP learnPoolParam, SEXP testPoolParam, SEXP fitParamsAsJsonParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle learnPool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(learnPoolParam));
    auto fitParams = LoadFitParams(fitParamsAsJsonParam);
    TFullModelPtr modelPtr = std::make_unique<TFullModel>();
    if (testPoolParam != R_NilValue) {
        TEvalResult evalResult;
        TPoolHandle testPool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(testPoolParam));
        TrainModel(
            fitParams,
            Nothing(),
            Nothing(),
            TClearablePoolPtrs(*learnPool, {testPool}),
            "",
            modelPtr.get(),
            {&evalResult}
        );
    }
    else {
        TrainModel(
            fitParams,
            Nothing(),
            Nothing(),
            TClearablePoolPtrs(*learnPool, {}),
            "",
            modelPtr.get(),
            {}
        );
    }
    result = PROTECT(R_MakeExternalPtr(modelPtr.get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TFullModelHandle>, TRUE);
    modelPtr.release();
    R_API_END();
    UNPROTECT(1);
    return result;
}

SEXP CatBoostCV_R(SEXP fitParamsAsJsonParam, 
                  SEXP poolParam,
                  SEXP foldCountParam,
                  SEXP invertedParam,
                  SEXP partitionRandomSeedParam,
                  SEXP shuffleParam,
                  SEXP stratifiedParam) {

    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    auto fitParams = LoadFitParams(fitParamsAsJsonParam);

    TCrossValidationParams cvParams;
    cvParams.FoldCount = asInteger(foldCountParam);
    cvParams.PartitionRandSeed = asInteger(partitionRandomSeedParam);
    cvParams.Shuffle = asLogical(shuffleParam);
    cvParams.Stratified = asLogical(stratifiedParam);
    cvParams.Inverted = asLogical(invertedParam);

    TVector<TCVResult> cvResults;

    CrossValidate(
        fitParams,
        Nothing(),
        Nothing(),
        *pool,
        cvParams,
        &cvResults);

    size_t metricCount = cvResults.size();
    result = PROTECT(allocVector(VECSXP, metricCount * 4));
    for (size_t metricIdx = 0; metricIdx < metricCount; ++metricIdx) {
        TString metricName = cvResults[metricIdx].Metric;

        size_t iterationsCount = cvResults[metricIdx].AverageTrain.size();

        SEXP row_test_mean = PROTECT(allocVector(REALSXP, iterationsCount));
        SEXP row_test_std = PROTECT(allocVector(REALSXP, iterationsCount));
        SEXP row_train_mean = PROTECT(allocVector(REALSXP, iterationsCount));
        SEXP row_train_std = PROTECT(allocVector(REALSXP, iterationsCount));

        for (size_t i = 0; i < iterationsCount; ++i) {
            REAL(row_test_mean)[i] = cvResults[metricIdx].AverageTest[i];
            REAL(row_test_std)[i] = cvResults[metricIdx].StdDevTest[i];
            REAL(row_train_mean)[i] = cvResults[metricIdx].AverageTrain[i];
            REAL(row_train_std)[i] = cvResults[metricIdx].StdDevTrain[i];
        }

        setAttrib(result, mkString(("test-" + metricName + "-mean").c_str()), row_test_mean);
        setAttrib(result, mkString(("test-" + metricName + "-std").c_str()), row_test_std);
        setAttrib(result, mkString(("train-" + metricName + "-mean").c_str()), row_train_mean);
        setAttrib(result, mkString(("train-" + metricName + "-std").c_str()), row_train_std);
    }
    
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

SEXP CatBoostSerializeModel_R(SEXP handleParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle modelHandle = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(handleParam));
    const TString& raw = SerializeModel(*modelHandle);
    result = PROTECT(allocVector(RAWSXP, raw.Size()));
    MemCopy(RAW(result), (const unsigned char*)(raw.Data()), raw.Size());
    R_API_END();
    UNPROTECT(1);
    return result;
}

SEXP CatBoostDeserializeModel_R(SEXP rawParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelPtr modelPtr = std::make_unique<TFullModel>();
    DeserializeModel(TMemoryInput(RAW(rawParam), length(rawParam))).Swap(*modelPtr);
    result = PROTECT(R_MakeExternalPtr(modelPtr.get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TFullModelHandle>, TRUE);
    modelPtr.release();
    R_API_END();
    UNPROTECT(1);
    return result;
}

SEXP CatBoostPredictMulti_R(SEXP modelParam, SEXP poolParam, SEXP verboseParam,
                            SEXP typeParam, SEXP treeCountStartParam, SEXP treeCountEndParam, SEXP threadCountParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    EPredictionType predictionType;
    CB_ENSURE(TryFromString<EPredictionType>(CHAR(asChar(typeParam)), predictionType),
              "unsupported prediction type: 'Probability', 'Class' or 'RawFormulaVal' was expected");
    TVector<TVector<double>> prediction = ApplyModelMulti(*model,
                                                          *pool,
                                                          asLogical(verboseParam),
                                                          predictionType,
                                                          asInteger(treeCountStartParam),
                                                          asInteger(treeCountEndParam),
                                                          UpdateThreadCount(asInteger(threadCountParam)));
    size_t predictionSize = prediction.size() * pool->Docs.GetDocCount();
    result = PROTECT(allocVector(REALSXP, predictionSize));
    for (size_t i = 0, k = 0; i < pool->Docs.GetDocCount(); ++i) {
        for (size_t j = 0; j < prediction.size(); ++j) {
            REAL(result)[k++] = prediction[j][i];
        }
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

SEXP CatBoostPrepareEval_R(SEXP approxParam, SEXP typeParam, SEXP columnCountParam, SEXP threadCountParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    SEXP dataDim = getAttrib(approxParam, R_DimSymbol);
    size_t dataRows = static_cast<size_t>(INTEGER(dataDim)[0]) / asInteger(columnCountParam);
    TVector<TVector<double>> prediction(asInteger(columnCountParam), TVector<double>(dataRows));
    for (size_t i = 0, k = 0; i < dataRows; ++i) {
        for (size_t j = 0; j < prediction.size(); ++j) {
            prediction[j][i] = static_cast<double>(REAL(approxParam)[k++]);
        }
    }

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(UpdateThreadCount(asInteger(threadCountParam)) - 1);
    EPredictionType predictionType;
    CB_ENSURE(TryFromString<EPredictionType>(CHAR(asChar(typeParam)), predictionType),
              "unsupported prediction type: 'Probability', 'Class' or 'RawFormulaVal' was expected");
    prediction = PrepareEval(predictionType, prediction, &executor);

    size_t predictionSize = prediction.size() * dataRows;
    result = PROTECT(allocVector(REALSXP, predictionSize));
    for (size_t i = 0, k = 0; i < dataRows; ++i) {
        for (size_t j = 0; j < prediction.size(); ++j) {
            REAL(result)[k++] = prediction[j][i];
        }
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

SEXP CatBoostShrinkModel_R(SEXP modelParam, SEXP treeCountStartParam, SEXP treeCountEndParam) {
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    model->Truncate(asInteger(treeCountStartParam), asInteger(treeCountEndParam));
    R_API_END();
    return ScalarLogical(1);
}

SEXP CatBoostDropUnusedFeaturesFromModel_R(SEXP modelParam) {
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    model->ObliviousTrees.DropUnusedFeatures();
    R_API_END();
    return ScalarLogical(1);
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

SEXP CatBoostCalcRegularFeatureEffect_R(SEXP modelParam, SEXP poolParam, SEXP fstrTypeParam, SEXP threadCountParam) {
    SEXP result = NULL;
    SEXP resultDim = NULL;
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    TString fstrType = CHAR(asChar(fstrTypeParam));

    const int threadCount = UpdateThreadCount(asInteger(threadCountParam));
    const bool multiClass = model->ObliviousTrees.ApproxDimension > 1;
    const bool verbose = false;
    // TODO(akhropov): make prettified mode as in python-package
    if (fstrType == "ShapValues" && multiClass) {
        TVector<TVector<TVector<double>>> fstr = GetFeatureImportancesMulti(fstrType, *model, pool, threadCount, verbose);
        size_t numDocs = fstr.size();
        size_t numClasses = numDocs > 0 ? fstr[0].size() : 0;
        size_t numValues = numClasses > 0 ? fstr[0][0].size() : 0;
        size_t resultSize = numDocs * numClasses * numValues;
        result = PROTECT(allocVector(REALSXP, resultSize));
        size_t r = 0;
        for (size_t k = 0; k < numValues; ++k) {
            for (size_t j = 0; j < numClasses; ++j) {
               for (size_t i = 0; i < numDocs; ++i) {
                    REAL(result)[r++] = fstr[i][j][k];
                }
            }
        }
        PROTECT(resultDim = allocVector(INTSXP, 3));
        INTEGER(resultDim)[0] = numDocs;
        INTEGER(resultDim)[1] = numClasses;
        INTEGER(resultDim)[2] = numValues;
        setAttrib(result, R_DimSymbol, resultDim);
    } else {
        TVector<TVector<double>> fstr = GetFeatureImportances(fstrType, *model, pool, threadCount, verbose);
        size_t numRows = fstr.size();
        size_t numCols = numRows > 0 ? fstr[0].size() : 0;
        size_t resultSize = numRows * numCols;
        result = PROTECT(allocVector(REALSXP, resultSize));
        size_t r = 0;
        for (size_t j = 0; j < numCols; ++j) {
            for (size_t i = 0; i < numRows; ++i) {
                REAL(result)[r++] = fstr[i][j];
            }
        }
        PROTECT(resultDim = allocVector(INTSXP, 2));
        INTEGER(resultDim)[0] = numRows;
        INTEGER(resultDim)[1] = numCols;
        setAttrib(result, R_DimSymbol, resultDim);
    }

    R_API_END();
    UNPROTECT(2);
    return result;
}

SEXP CatBoostEvaluateObjectImportances_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP trainPoolParam,
    SEXP topSizeParam,
    SEXP ostrTypeParam,
    SEXP updateMethodParam,
    SEXP threadCountParam
) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    TPoolHandle trainPool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(trainPoolParam));
    TString ostrType = CHAR(asChar(ostrTypeParam));
    TString updateMethod = CHAR(asChar(updateMethodParam));
    TDStrResult dstrResult = GetDocumentImportances(
        *model,
        *trainPool,
        *pool,
        ostrType,
        asInteger(topSizeParam),
        updateMethod,
        /*importanceValuesSignStr=*/ToString(EImportanceValuesSign::All),
        UpdateThreadCount(asInteger(threadCountParam))
    );
    size_t resultSize = 0;
    if (!dstrResult.Indices.empty()) {
        resultSize += dstrResult.Indices.size() * dstrResult.Indices[0].size();
    }
    if (!dstrResult.Scores.empty()) {
        resultSize += dstrResult.Scores.size() * dstrResult.Scores[0].size();
    }
    result = PROTECT(allocVector(REALSXP, resultSize));
    size_t k = 0;
    for (size_t i = 0; i < dstrResult.Indices.size(); ++i) {
        for (size_t j = 0; j < dstrResult.Indices[0].size(); ++j) {
            REAL(result)[k++] = dstrResult.Indices[i][j];
        }
    }
    for (size_t i = 0; i < dstrResult.Scores.size(); ++i) {
        for (size_t j = 0; j < dstrResult.Scores[0].size(); ++j) {
            REAL(result)[k++] = dstrResult.Scores[i][j];
        }
    }
    R_API_END();
    UNPROTECT(1);
    return result;
}

SEXP CatBoostIsNullHandle_R(SEXP handleParam) {
    return ScalarLogical(!R_ExternalPtrAddr(handleParam));
}
}
