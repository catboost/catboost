#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/data_new/data_provider_builders.h>
#include <catboost/libs/data_new/load_data.h>
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
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/cat_feature/cat_feature.h>

#include <util/generic/cast.h>
#include <util/generic/mem_copy.h>
#include <util/generic/singleton.h>
#include <util/generic/xrange.h>
#include <util/string/cast.h>
#include <util/system/info.h>

#include <algorithm>


#if defined(SIZEOF_SIZE_T)
#undef SIZEOF_SIZE_T
#endif

#include <Rinternals.h>


using namespace NCB;


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


typedef TDataProvider* TPoolHandle;
typedef TDataProviderPtr TPoolPtr;

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
        TDsvFormatOptions{static_cast<bool>(asLogical(hasHeaderParam)),
                               CHAR(asChar(delimiterParam))[0]};

    TStringBuf cdPathWithScheme(CHAR(asChar(cdFileParam)));
    if (!cdPathWithScheme.empty()) {
        dsvPoolFormatParams.CdFilePath = TPathWithScheme(cdPathWithScheme, "dsv");
    }

    TStringBuf pairsPathWithScheme(CHAR(asChar(pairsFileParam)));

    TDataProviderPtr poolPtr = ReadDataset(TPathWithScheme(CHAR(asChar(poolFileParam)), "dsv"),
                                           !pairsPathWithScheme.empty() ?
                                               TPathWithScheme(pairsPathWithScheme, "dsv") : TPathWithScheme(),
                                           /*groupWeightsFilePath=*/TPathWithScheme(),
                                           dsvPoolFormatParams,
                                           TVector<ui32>(),
                                           EObjectsOrder::Undefined,
                                           UpdateThreadCount(asInteger(threadCountParam)),
                                           asLogical(verboseParam));
    result = PROTECT(R_MakeExternalPtr(poolPtr.Get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TPoolHandle>, TRUE);
    Y_UNUSED(poolPtr.Release());
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
    ui32 dataRows = SafeIntegerCast<ui32>(INTEGER(dataDim)[0]);
    ui32 dataColumns = SafeIntegerCast<ui32>(INTEGER(dataDim)[1]);
    SEXP baselineDim = getAttrib(baselineParam, R_DimSymbol);
    size_t baselineRows = 0;
    size_t baselineColumns = 0;
    if (baselineParam != R_NilValue) {
        baselineRows = static_cast<size_t>(INTEGER(baselineDim)[0]);
        baselineColumns = static_cast<size_t>(INTEGER(baselineDim)[1]);
    }

    auto loaderFunc = [&] (IRawFeaturesOrderDataVisitor* visitor) {
        TDataMetaInfo metaInfo;

        TVector<TString> featureId;
        if (featureNamesParam != R_NilValue) {
            for (size_t i = 0; i < dataColumns; ++i) {
                featureId.push_back(CHAR(asChar(VECTOR_ELT(featureNamesParam, i))));
            }
        }

        metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
            dataColumns,
            ToUnsigned(GetVectorFromSEXP<int>(catFeaturesParam)),
            featureId,
            nullptr);

        metaInfo.HasTarget = targetParam != R_NilValue;
        metaInfo.BaselineCount = baselineColumns;
        metaInfo.HasGroupId = groupIdParam != R_NilValue;
        metaInfo.HasGroupWeight = groupWeightParam != R_NilValue;
        metaInfo.HasSubgroupIds = subgroupIdParam != R_NilValue;
        metaInfo.HasWeights = weightParam != R_NilValue;

        visitor->Start(metaInfo, dataRows, EObjectsOrder::Undefined, {});

        TVector<float> target(metaInfo.HasTarget ? dataRows : 0);
        TVector<float> weights(metaInfo.HasWeights ? dataRows : 0);
        TVector<float> groupWeights(metaInfo.HasGroupWeight ? dataRows : 0);

        for (ui32 i = 0; i < dataRows; ++i) {
            if (metaInfo.HasGroupId) {
                visitor->AddGroupId(i, static_cast<uint32_t>(INTEGER(groupIdParam)[i]));
            }
            if (metaInfo.HasSubgroupIds) {
                visitor->AddSubgroupId(i, static_cast<uint32_t>(INTEGER(subgroupIdParam)[i]));
            }

            if (targetParam != R_NilValue) {
                target[i] = static_cast<float>(REAL(targetParam)[i]);
            }
            if (weightParam != R_NilValue) {
                weights[i] = static_cast<float>(REAL(weightParam)[i]);
            }
            if (groupWeightParam != R_NilValue) {
                groupWeights[i] = static_cast<float>(REAL(groupWeightParam)[i]);
            }
        }
        if (metaInfo.HasTarget) {
            visitor->AddTarget(target);
        }
        if (metaInfo.HasWeights) {
            visitor->AddWeights(weights);
        }
        if (metaInfo.HasGroupWeight) {
            visitor->SetGroupWeights(std::move(groupWeights));
        }
        if (metaInfo.BaselineCount) {
            TVector<float> baseline(dataRows);
            for (size_t j = 0; j < baselineColumns; ++j) {
                for (ui32 i = 0; i < dataRows; ++i) {
                    baseline[i] = static_cast<float>(REAL(baselineParam)[i + baselineRows * j]);
                }
                visitor->AddBaseline(j, baseline);
            }
        }

        for (size_t j = 0; j < dataColumns; ++j) {
            if (metaInfo.FeaturesLayout->GetExternalFeatureType(j) == EFeatureType::Categorical) {
                TVector<ui32> catValues;
                catValues.yresize(dataRows);
                for (ui32 i = 0; i < dataRows; ++i) {
                    catValues[i] =
                        ConvertFloatCatFeatureToIntHash(static_cast<float>(REAL(matrixParam)[i + dataRows * j]));
                }
                visitor->AddCatFeature(j, TMaybeOwningConstArrayHolder<ui32>::CreateOwning(std::move(catValues)));
            } else {
                TVector<float> floatValues;
                floatValues.yresize(dataRows);
                for (ui32 i = 0; i < dataRows; ++i) {
                    floatValues[i] = static_cast<float>(REAL(matrixParam)[i + dataRows * j]);
                }
                visitor->AddFloatFeature(j, TMaybeOwningConstArrayHolder<float>::CreateOwning(std::move(floatValues)));
            }
        }

        if (pairsParam != R_NilValue) {
            TVector<TPair> pairs;
            size_t pairsCount = static_cast<size_t>(INTEGER(getAttrib(pairsParam, R_DimSymbol))[0]);
            for (size_t i = 0; i < pairsCount; ++i) {
                float weight = 1;
                if (pairsWeightParam != R_NilValue) {
                    weight = static_cast<float>(REAL(pairsWeightParam)[i]);
                }
                pairs.emplace_back(
                    static_cast<int>(INTEGER(pairsParam)[i + pairsCount * 0]),
                    static_cast<int>(INTEGER(pairsParam)[i + pairsCount * 1]),
                    weight
                );
            }
            visitor->SetPairs(std::move(pairs));
        }
        visitor->Finish();
    };

    TDataProviderPtr poolPtr = CreateDataProvider(std::move(loaderFunc));

    result = PROTECT(R_MakeExternalPtr(poolPtr.Get(), R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(result, _Finalizer<TPoolHandle>, TRUE);
    Y_UNUSED(poolPtr.Release());
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
    result = ScalarInteger(static_cast<int>(pool->ObjectsGrouping->GetObjectCount()));
    R_API_END();
    return result;
}

SEXP CatBoostPoolNumCol_R(SEXP poolParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    result = ScalarInteger(0);
    if (pool->ObjectsGrouping->GetObjectCount() != 0) {
        result = ScalarInteger(static_cast<int>(pool->MetaInfo.GetFeatureCount()));
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
    const TRawObjectsDataProvider* rawObjectsData
        = dynamic_cast<const TRawObjectsDataProvider*>(pool->ObjectsData.Get());
    CB_ENSURE(rawObjectsData, "Cannot Slice quantized features data");

    result = PROTECT(allocVector(VECSXP, size));
    ui32 featureCount = pool->MetaInfo.GetFeatureCount();
    auto target = *pool->RawTargetData.GetTarget();
    const auto& weights = pool->RawTargetData.GetWeights();

    // TODO(akhropov): get only data for slice objects
    TVector<TVector<float>> rawFeatures(featureCount); // [flatFeatureIdx][objectIdx]
    for (auto featureIdx : xrange(featureCount)) {
        rawFeatures[featureIdx] = rawObjectsData->GetFeatureDataOldFormat(featureIdx);
    }

    for (size_t i = offset; i < std::min((size_t)pool->GetObjectCount(), offset + size); ++i) {
        ui32 featureCount = pool->MetaInfo.GetFeatureCount();
        SEXP row = PROTECT(allocVector(REALSXP, featureCount + 2));
        REAL(row)[0] = FromString<double>(target[i]);
        REAL(row)[1] = weights[i];
        for (ui32 j = 0; j < featureCount; ++j) {
            REAL(row)[j + 2] = rawFeatures[j].empty() ? 0.0f : rawFeatures[j][i];
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
    TDataProviders pools;
    pools.Learn = learnPool;

    auto fitParams = LoadFitParams(fitParamsAsJsonParam);
    TFullModelPtr modelPtr = std::make_unique<TFullModel>();
    if (testPoolParam != R_NilValue) {
        TEvalResult evalResult;
        TPoolHandle testPool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(testPoolParam));
        pools.Test.emplace_back(testPool);
        TrainModel(
            fitParams,
            nullptr,
            Nothing(),
            Nothing(),
            pools,
            "",
            modelPtr.get(),
            {&evalResult}
        );
        Y_UNUSED(pools.Test.back().Release());
    }
    else {
        TrainModel(
            fitParams,
            nullptr,
            Nothing(),
            Nothing(),
            pools,
            "",
            modelPtr.get(),
            {}
        );
    }
    Y_UNUSED(pools.Learn.Release());
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
    size_t metricCount;

    R_API_BEGIN();
    TPoolPtr pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
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
        pool,
        cvParams,
        &cvResults);
    Y_UNUSED(pool.Release());

    metricCount = cvResults.size();
    result = PROTECT(allocVector(VECSXP, metricCount * 4));
    SEXP columnNames = PROTECT(allocVector(STRSXP, metricCount * 4));

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

        SET_VECTOR_ELT(result, metricIdx * 4 + 0, row_test_mean);
        SET_VECTOR_ELT(result, metricIdx * 4 + 1, row_test_std);
        SET_VECTOR_ELT(result, metricIdx * 4 + 2, row_train_mean);
        SET_VECTOR_ELT(result, metricIdx * 4 + 3, row_train_std);

        SET_STRING_ELT(columnNames, metricIdx * 4 + 0, mkChar(("test-" + metricName + "-mean").c_str()));
        SET_STRING_ELT(columnNames, metricIdx * 4 + 1, mkChar(("test-" + metricName + "-std").c_str()));
        SET_STRING_ELT(columnNames, metricIdx * 4 + 2, mkChar(("train-" + metricName + "-mean").c_str()));
        SET_STRING_ELT(columnNames, metricIdx * 4 + 3, mkChar(("train-" + metricName + "-std").c_str()));
    }

    setAttrib(result, R_NamesSymbol, columnNames);

    R_API_END();
    UNPROTECT(metricCount * 4 + 2);
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
    result = PROTECT(allocVector(RAWSXP, raw.size()));
    MemCopy(RAW(result), (const unsigned char*)(raw.data()), raw.size());
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
                                                          *pool->ObjectsData,
                                                          asLogical(verboseParam),
                                                          predictionType,
                                                          asInteger(treeCountStartParam),
                                                          asInteger(treeCountEndParam),
                                                          UpdateThreadCount(asInteger(threadCountParam)));
    size_t predictionSize = prediction.size() * pool->ObjectsGrouping->GetObjectCount();
    result = PROTECT(allocVector(REALSXP, predictionSize));
    for (size_t i = 0, k = 0; i < pool->ObjectsGrouping->GetObjectCount(); ++i) {
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
    TDataProviderPtr pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
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
    Y_UNUSED(pool.Release());
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
    const bool verbose = false;
    TDStrResult dstrResult = GetDocumentImportances(
        *model,
        *trainPool,
        *pool,
        ostrType,
        asInteger(topSizeParam),
        updateMethod,
        /*importanceValuesSignStr=*/ToString(EImportanceValuesSign::All),
        UpdateThreadCount(asInteger(threadCountParam)),
        verbose
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
