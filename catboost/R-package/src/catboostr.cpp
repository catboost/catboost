#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/fstr/calc_fstr.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_export/model_exporter.h>
#include <catboost/libs/model/utils.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/libs/train_lib/cross_validation.h>
#include <catboost/private/libs/algo/apply.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/algo/mvs.h>
#include <catboost/private/libs/algo/plot.h>
#include <catboost/private/libs/documents_importance/docs_importance.h>
#include <catboost/private/libs/documents_importance/enums.h>
#include <catboost/private/libs/options/cross_validation_params.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/target/data_providers.h>

#include <util/generic/cast.h>
#include <util/generic/mem_copy.h>
#include <util/generic/singleton.h>
#include <util/generic/xrange.h>
#include <util/string/cast.h>
#include <util/system/info.h>

#include <algorithm>
#include <cstdio>

#if defined(SIZEOF_SIZE_T)
#undef SIZEOF_SIZE_T
#endif

#include "catboostr.h"

using namespace NCB;


#define MAXLEN_R_ERRMSG 1024
char buffer_R_errmsg[MAXLEN_R_ERRMSG];
void copy_exception_msg_to_buffer(std::exception &e) {
    std::snprintf(buffer_R_errmsg, MAXLEN_R_ERRMSG, "%s\n", e.what());
}

struct CB_R_ERROR {SEXP uw_token;};

void loggingFunc(const char* str, size_t len, TCustomLoggingObject) {
    TString slicedStr(str, 0, len);
    Rprintf("%s", slicedStr.c_str());
    R_FlushConsole();
}

void loggingErrFunc(const char* str, size_t len, TCustomLoggingObject) {
    TString slicedStr(str, 0, len);
    REprintf("%s", slicedStr.c_str());
    R_FlushConsole();
}

class CB_R_context {
public:
    CB_R_context() {
        SetCustomLoggingFunction(loggingFunc, loggingErrFunc);
    }

    ~CB_R_context() {
        RestoreOriginalLogger();
    }
};

#define R_API_BEGIN()                                                           \
    bool threw_error = false;                                                   \
    try {                                                                       \
    *Singleton<TRPackageInitializer>();                                         \
    CB_R_context context_manager;                                               \


#define R_API_END()                                                 \
    } catch (std::exception& e) {                                   \
        copy_exception_msg_to_buffer(e);                            \
        threw_error = true;                                         \
    } catch (CB_R_ERROR &err_token) {                               \
        R_ContinueUnwind(err_token.uw_token);                       \
    } catch (...) {                                                 \
        std::sprintf(buffer_R_errmsg, "%s\n", "Unexpected error");  \
        threw_error = true;                                         \
    }                                                               \
    if (threw_error) Rf_error(buffer_R_errmsg);                     \
    return R_NilValue;                                              \

typedef TDataProvider* TPoolHandle;
typedef TDataProviderPtr TPoolPtr;

typedef TFullModel* TFullModelHandle;
typedef std::unique_ptr<TFullModel> TFullModelPtr;
typedef const TFullModel* TFullModelConstPtr;

class TRPackageInitializer {
    Y_DECLARE_SINGLETON_FRIEND();
    TRPackageInitializer() {
        ConfigureMalloc();
    }
};

void safe_R_error(void *ptr_uw_token, Rboolean jump) {
    if (jump) {
        throw CB_R_ERROR{*static_cast<SEXP*>(ptr_uw_token)};
    }
}

SEXP wrapped_R_VECSXP(void *ptr_size) {
    return Rf_allocVector(VECSXP, *static_cast<R_xlen_t*>(ptr_size));
}

SEXP wrapped_R_REALSXP(void *ptr_size) {
    return Rf_allocVector(REALSXP, *static_cast<R_xlen_t*>(ptr_size));
}

SEXP wrapped_R_STRSXPSXP(void *ptr_size) {
    return Rf_allocVector(STRSXP, *static_cast<R_xlen_t*>(ptr_size));
}

SEXP wrapped_R_RAWSXP(void *ptr_size) {
    return Rf_allocVector(RAWSXP, *static_cast<R_xlen_t*>(ptr_size));
}

SEXP wrapped_R_INTSXP(void *ptr_size) {
    return Rf_allocVector(INTSXP, *static_cast<R_xlen_t*>(ptr_size));
}

SEXP wrapped_R_LGLSXPSXP(void *ptr_size) {
    return Rf_allocVector(LGLSXP, *static_cast<R_xlen_t*>(ptr_size));
}

SEXP safe_R_vector(int Rtype, R_xlen_t size, SEXP &uw_token) {
    switch (Rtype) {
        case VECSXP:
            return R_UnwindProtect(wrapped_R_VECSXP, static_cast<void*>(&size),
                                   safe_R_error, &uw_token, uw_token);
        case REALSXP:
            return R_UnwindProtect(wrapped_R_REALSXP, static_cast<void*>(&size),
                                   safe_R_error, &uw_token, uw_token);
        case STRSXP:
            return R_UnwindProtect(wrapped_R_STRSXPSXP, static_cast<void*>(&size),
                                   safe_R_error, &uw_token, uw_token);
        case RAWSXP:
            return R_UnwindProtect(wrapped_R_RAWSXP, static_cast<void*>(&size),
                                   safe_R_error, &uw_token, uw_token);
        case INTSXP:
            return R_UnwindProtect(wrapped_R_INTSXP, static_cast<void*>(&size),
                                   safe_R_error, &uw_token, uw_token);
        case LGLSXP:
            return R_UnwindProtect(wrapped_R_LGLSXPSXP, static_cast<void*>(&size),
                                   safe_R_error, &uw_token, uw_token);
        default:
            throw std::runtime_error("Unexpected error.");
            return R_NilValue;
    }
}

SEXP wrapped_mkChar(void *txt) {
    return Rf_mkChar(*static_cast<const char**>(txt));
}

SEXP safe_R_mkChar(const char* txt, SEXP &uw_token) {
    return R_UnwindProtect(wrapped_mkChar, static_cast<void*>(&txt),
                           safe_R_error, &uw_token, uw_token);
}

SEXP wrapped_asChar(void *R_obj) {
    return Rf_asChar(*static_cast<SEXP*>(R_obj));
}

SEXP safe_R_asChar(SEXP R_obj, SEXP &uw_token) {
    return R_UnwindProtect(wrapped_asChar, static_cast<void*>(&R_obj),
                           safe_R_error, &uw_token, uw_token);
}

SEXP wrapped_mkString(void *txt) {
    return Rf_mkString(*static_cast<const char**>(txt));
}

SEXP safe_R_mkString(const char *txt, SEXP &uw_token) {
    return R_UnwindProtect(wrapped_mkString, static_cast<void*>(&txt),
                           safe_R_error, &uw_token, uw_token);
}

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

static NJson::TJsonValue LoadFitParams(const char *fitParamsAsJson) {
    TString paramsStr(fitParamsAsJson);
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

EXPORT_FUNCTION CatBoostCreateFromFile_R(SEXP poolFileParam,
                              SEXP cdFileParam,
                              SEXP pairsFileParam,
                              SEXP featureNamesFileParam,
                              SEXP delimiterParam,
                              SEXP numVectorDelimiterParam,
                              SEXP hasHeaderParam,
                              SEXP threadCountParam,
                              SEXP verboseParam) {
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    SEXP result = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
    R_API_BEGIN();
    SEXP delimiterParam_char = PROTECT(safe_R_asChar(delimiterParam, uw_token));
    SEXP numVectorDelimiterParam_char = PROTECT(safe_R_asChar(numVectorDelimiterParam, uw_token));

    NCatboostOptions::TColumnarPoolFormatParams columnarPoolFormatParams;
    columnarPoolFormatParams.DsvFormat =
        TDsvFormatOptions{
            static_cast<bool>(asLogical(hasHeaderParam)),
            CHAR(delimiterParam_char)[0],
            CHAR(numVectorDelimiterParam_char)[0]
        };

    SEXP cdFileParam_char = PROTECT(safe_R_asChar(cdFileParam, uw_token));
    TStringBuf cdPathWithScheme(CHAR(cdFileParam_char));
    if (!cdPathWithScheme.empty()) {
        columnarPoolFormatParams.CdFilePath = TPathWithScheme(cdPathWithScheme, "dsv");
    }

    SEXP pairsFileParam_char = PROTECT(safe_R_asChar(pairsFileParam, uw_token));
    SEXP featureNamesFileParam_char = PROTECT(safe_R_asChar(featureNamesFileParam, uw_token));
    TStringBuf pairsPathWithScheme(CHAR(pairsFileParam_char));
    TStringBuf featureNamesPathWithScheme(CHAR(featureNamesFileParam_char));
    SEXP poolFileParam_char = PROTECT(safe_R_asChar(poolFileParam, uw_token));

    TDataProviderPtr poolPtr = ReadDataset(/*taskType*/Nothing(),
                                           TPathWithScheme(CHAR(poolFileParam_char), "dsv"),
                                           !pairsPathWithScheme.empty() ?
                                               TPathWithScheme(pairsPathWithScheme, "dsv-flat") : TPathWithScheme(),
                                           /*groupWeightsFilePath=*/TPathWithScheme(),
                                           /*timestampsFilePath=*/TPathWithScheme(),
                                           /*baselineFilePath=*/TPathWithScheme(),
                                           !featureNamesPathWithScheme.empty() ?
                                                TPathWithScheme(featureNamesPathWithScheme, "dsv") : TPathWithScheme(),
                                           /*poolMetaInfoPath=*/TPathWithScheme(),
                                           columnarPoolFormatParams,
                                           TVector<ui32>(),
                                           EObjectsOrder::Undefined,
                                           UpdateThreadCount(asInteger(threadCountParam)),
                                           asLogical(verboseParam),
                                           /*forceUnitAutoPairWeights*/ false,
                                           /*classLabels=*/Nothing());
    R_SetExternalPtrAddr(result, poolPtr.Get());
    R_RegisterCFinalizerEx(result, _Finalizer<TPoolHandle>, TRUE);
    Y_UNUSED(poolPtr.Release());
    UNPROTECT(8);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostCreateFromMatrix_R(SEXP floatAndCatMatrixParam,
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
                                SEXP featureNamesParam) {
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    SEXP result = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
    size_t n_protected = 2;
    R_API_BEGIN();
    SEXP dataDim = floatAndCatMatrixParam != R_NilValue ?
                   getAttrib(floatAndCatMatrixParam, R_DimSymbol) :
                   getAttrib(textMatrixParam, R_DimSymbol);
    ui32 dataRows = SafeIntegerCast<ui32>(INTEGER(dataDim)[0]);
    ui32 floatAndCatColumns = floatAndCatMatrixParam == R_NilValue ? 0 :
                       SafeIntegerCast<ui32>(INTEGER(getAttrib(floatAndCatMatrixParam, R_DimSymbol))[1]);
    ui32 textColumns = textMatrixParam == R_NilValue ? 0 :
                       SafeIntegerCast<ui32>(INTEGER(getAttrib(textMatrixParam, R_DimSymbol))[1]);
    ui32 dataColumns = floatAndCatColumns + textColumns;
    SEXP targetDim = getAttrib(targetParam, R_DimSymbol);
    ui32 targetRows = 0;
    ui32 targetColumns = 0;
    if (targetDim != R_NilValue) {
        targetRows = SafeIntegerCast<ui32>(INTEGER(targetDim)[0]);
        targetColumns = SafeIntegerCast<ui32>(INTEGER(targetDim)[1]);
    }
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
                SEXP temp = PROTECT(safe_R_asChar(VECTOR_ELT(featureNamesParam, i), uw_token));
                n_protected++;
                featureId.push_back(CHAR(temp));
            }
        }

        metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
            dataColumns,
            ToUnsigned(GetVectorFromSEXP<int>(catFeaturesIndicesParam)),
            ToUnsigned(GetVectorFromSEXP<int>(textFeaturesIndicesParam)),
            TVector<ui32>{}, // TODO(akhropov) support embedding features in R
            featureId);

        metaInfo.TargetType = targetColumns ? ERawTargetType::Float : ERawTargetType::None;
        metaInfo.TargetCount = targetColumns;
        metaInfo.BaselineCount = baselineColumns;
        metaInfo.HasGroupId = groupIdParam != R_NilValue;
        metaInfo.HasGroupWeight = groupWeightParam != R_NilValue;
        metaInfo.HasSubgroupIds = subgroupIdParam != R_NilValue;
        metaInfo.HasWeights = weightParam != R_NilValue;
        metaInfo.HasTimestamp = false;

        visitor->Start(metaInfo, dataRows, EObjectsOrder::Undefined, {});

        double *ptr_targetParam = Rf_isNull(targetParam)? nullptr : REAL(targetParam);
        for (auto targetIdx : xrange(targetColumns)) {
            TVector<float> target(targetRows);
            for (auto docIdx : xrange(targetRows)) {
                target[docIdx] = static_cast<float>(ptr_targetParam[docIdx + targetRows * targetIdx]);
            }
            visitor->AddTarget(
                targetIdx,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(std::move(target))
            );
        }
        TVector<float> weights(metaInfo.HasWeights ? dataRows : 0);
        TVector<float> groupWeights(metaInfo.HasGroupWeight ? dataRows : 0);

        int *ptr_groupIdParam = Rf_isNull(groupIdParam)? nullptr : INTEGER(groupIdParam);
        int *ptr_subgroupIdParam = Rf_isNull(subgroupIdParam)? nullptr : INTEGER(subgroupIdParam);
        double *ptr_weightParam = Rf_isNull(weightParam)? nullptr : REAL(weightParam);
        double *ptr_groupWeightParam = Rf_isNull(groupWeightParam)? nullptr : REAL(groupWeightParam);
        for (ui32 i = 0; i < dataRows; ++i) {
            if (metaInfo.HasGroupId) {
                visitor->AddGroupId(i, static_cast<uint32_t>(ptr_groupIdParam[i]));
            }
            if (metaInfo.HasSubgroupIds) {
                visitor->AddSubgroupId(i, static_cast<uint32_t>(ptr_subgroupIdParam[i]));
            }

            if (weightParam != R_NilValue) {
                weights[i] = static_cast<float>(ptr_weightParam[i]);
            }
            if (groupWeightParam != R_NilValue) {
                groupWeights[i] = static_cast<float>(ptr_groupWeightParam[i]);
            }
        }
        if (metaInfo.HasWeights) {
            visitor->AddWeights(weights);
        }
        if (metaInfo.HasGroupWeight) {
            visitor->SetGroupWeights(std::move(groupWeights));
        }
        if (metaInfo.BaselineCount) {
            TVector<float> baseline(dataRows);
            double *ptr_baselineParam = Rf_isNull(baselineParam)? nullptr : REAL(baselineParam);
            for (size_t j = 0; j < baselineColumns; ++j) {
                for (ui32 i = 0; i < dataRows; ++i) {
                    baseline[i] = static_cast<float>(ptr_baselineParam[i + baselineRows * j]);
                }
                visitor->AddBaseline(j, baseline);
            }
        }

        double *ptr_floatAndCatMatrixParam = Rf_isNull(floatAndCatMatrixParam)? nullptr : REAL(floatAndCatMatrixParam);
        size_t indexTextMatrix = 0;
        size_t indexFloatAndCatMatrix = 0;
        for (size_t j = 0; j < dataColumns; ++j){
            if (metaInfo.FeaturesLayout->GetExternalFeatureType(j) == EFeatureType::Text) {
                TVector<TString> textValues;
                textValues.yresize(dataRows);
                for (ui32 i = 0; i < dataRows; ++i) {
                    textValues[i] = CHAR(STRING_PTR(textMatrixParam)[i + dataRows * indexTextMatrix]);
                }
                visitor->AddTextFeature(j, TMaybeOwningConstArrayHolder<TString>::CreateOwning(std::move(textValues)));
                indexTextMatrix++;
            } else {
                if (metaInfo.FeaturesLayout->GetExternalFeatureType(j) == EFeatureType::Categorical) {
                    TVector<ui32> catValues;
                    catValues.yresize(dataRows);
                    for (ui32 i = 0; i < dataRows; ++i) {
                        catValues[i] =
                            ConvertFloatCatFeatureToIntHash(static_cast<float>(ptr_floatAndCatMatrixParam[i + dataRows * indexFloatAndCatMatrix]));
                    }
                    visitor->AddCatFeature(j, TMaybeOwningConstArrayHolder<ui32>::CreateOwning(std::move(catValues)));
                } else {
                    TVector<float> floatValues;
                    floatValues.yresize(dataRows);
                    for (ui32 i = 0; i < dataRows; ++i) {
                        floatValues[i] = static_cast<float>(ptr_floatAndCatMatrixParam[i + dataRows * indexFloatAndCatMatrix]);
                    }
                    visitor->AddFloatFeature(j, MakeTypeCastArrayHolderFromVector<float, float>(floatValues));
                }
                indexFloatAndCatMatrix++;
            }
        }

        if (pairsParam != R_NilValue) {
            TVector<TPair> pairs;
            size_t pairsCount = static_cast<size_t>(INTEGER(getAttrib(pairsParam, R_DimSymbol))[0]);
            double *ptr_pairsWeightParam = Rf_isNull(pairsWeightParam)? nullptr : REAL(pairsWeightParam);
            int *ptr_pairsParam = INTEGER(pairsParam);
            for (size_t i = 0; i < pairsCount; ++i) {
                float weight = 1;
                if (pairsWeightParam != R_NilValue) {
                    weight = static_cast<float>(ptr_pairsWeightParam[i]);
                }
                pairs.emplace_back(
                    static_cast<int>(ptr_pairsParam[i + pairsCount * 0]),
                    static_cast<int>(ptr_pairsParam[i + pairsCount * 1]),
                    weight
                );
            }
            visitor->SetPairs(TRawPairsData(std::move(pairs)));
        }
        visitor->Finish();
    };

    TDataProviderPtr poolPtr = CreateDataProvider(std::move(loaderFunc));

    R_SetExternalPtrAddr(result, poolPtr.Get());
    R_RegisterCFinalizerEx(result, _Finalizer<TPoolHandle>, TRUE);
    Y_UNUSED(poolPtr.Release());
    UNPROTECT(n_protected);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostHashStrings_R(SEXP stringsParam) {
   SEXP result = PROTECT(allocVector(REALSXP, length(stringsParam)));
   R_API_BEGIN();
   double *ptr_result = REAL(result);
   for (int i = 0; i < length(stringsParam); ++i) {
       ptr_result[i] = static_cast<double>(ConvertCatFeatureHashToFloat(CalcCatFeatureHash(TString(CHAR(STRING_ELT(stringsParam, i))))));
   }
   UNPROTECT(1);
   return result;
   R_API_END();
}

EXPORT_FUNCTION CatBoostPoolNumRow_R(SEXP poolParam) {
    SEXP result = PROTECT(Rf_allocVector(INTSXP, 1));
    R_API_BEGIN();
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    INTEGER(result)[0] = static_cast<int>(pool->ObjectsGrouping->GetObjectCount());
    UNPROTECT(1);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostPoolNumCol_R(SEXP poolParam) {
    SEXP result = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(result)[0] = 0;
    R_API_BEGIN();
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    if (pool->ObjectsGrouping->GetObjectCount() != 0) {
        INTEGER(result)[0] = static_cast<int>(pool->MetaInfo.GetFeatureCount());
    }
    UNPROTECT(1);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostGetNumTrees_R(SEXP modelParam) {
    SEXP result = PROTECT(Rf_allocVector(INTSXP, 1));
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    INTEGER(result)[0] = static_cast<int>(model->GetTreeCount());
    UNPROTECT(1);
    return result;
    R_API_END();
}

// TODO(dbakshee): remove this backward compatibility gag in v0.11
EXPORT_FUNCTION CatBoostPoolNumTrees_R(SEXP modelParam) {
    return CatBoostGetNumTrees_R(modelParam);
}

EXPORT_FUNCTION CatBoostIsOblivious_R(SEXP modelParam) {
    SEXP result = PROTECT(Rf_allocVector(LGLSXP, 1));
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    LOGICAL(result)[0] = static_cast<int>(model->IsOblivious());
    UNPROTECT(1);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostIsGroupwiseMetric_R(SEXP modelParam) {
    SEXP result = NULL;
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    result = ScalarLogical(static_cast<int>(IsGroupwiseMetric(model->GetLossFunctionName())));
    R_API_END();
    return result;
}

EXPORT_FUNCTION CatBoostPoolSlice_R(SEXP poolParam, SEXP sizeParam, SEXP offsetParam) {
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    R_API_BEGIN();
    size_t size, offset;
    size = static_cast<size_t>(asInteger(sizeParam));
    offset = static_cast<size_t>(asInteger(offsetParam));
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    const TRawObjectsDataProvider* rawObjectsData
        = dynamic_cast<const TRawObjectsDataProvider*>(pool->ObjectsData.Get());
    CB_ENSURE(rawObjectsData, "Cannot Slice quantized features data");

    const auto& featuresLayout = *(rawObjectsData->GetFeaturesLayout());

    CB_ENSURE(
        featuresLayout.GetExternalFeatureCount() == featuresLayout.GetFloatFeatureCount(),
        "Dataset slicing error: non-numeric features present, slicing datasets with categorical and text features is not supported"
    );

    SEXP result = PROTECT(safe_R_vector(VECSXP, size, uw_token));
    ui32 featureCount = pool->MetaInfo.GetFeatureCount();
    auto target = pool->RawTargetData.GetTarget();
    const auto& weights = pool->RawTargetData.GetWeights();


    const size_t sliceEnd = std::min((size_t)pool->GetObjectCount(), offset + size);

    TRangesSubset<ui32>::TBlocks subsetBlocks = { TSubsetBlock<ui32>(TIndexRange<ui32>(offset, sliceEnd), 0) };

    TObjectsGroupingSubset objectsGroupingSubset = GetGroupingSubsetFromObjectsSubset(
        rawObjectsData->GetObjectsGrouping(),
        TArraySubsetIndexing<ui32>(TRangesSubset<ui32>(subsetBlocks[0].GetSize(), std::move(subsetBlocks))),
        EObjectsOrder::Ordered
    );

    TObjectsDataProviderPtr sliceObjectsData = rawObjectsData->GetSubset(
        objectsGroupingSubset,
        GetMonopolisticFreeCpuRam(),
        &NPar::LocalExecutor()
    );

    const TRawObjectsDataProvider& sliceRawObjectsData
        = dynamic_cast<const TRawObjectsDataProvider&>(*sliceObjectsData);

    TVector<double*> rows;
    const auto targetCount = pool->MetaInfo.TargetCount;

    for (size_t i = offset; i < sliceEnd; ++i) {
        ui32 featureCount = pool->MetaInfo.GetFeatureCount();
        SEXP row = safe_R_vector(REALSXP, featureCount + targetCount + 1, uw_token);
        SET_VECTOR_ELT(result, i - offset, row);
        REAL(row)[targetCount] = weights[i];
        rows.push_back(REAL(row));
    }

    for (auto targetIdx : xrange(targetCount)) {
        if (const ITypedSequencePtr<float>* typedSequence
                = std::get_if<ITypedSequencePtr<float>>(&((*target)[targetIdx])))
        {
            TIntrusivePtr<ITypedArraySubset<float>> subset = (*typedSequence)->GetSubset(
                &objectsGroupingSubset.GetObjectsIndexing()
            );
            subset->ForEach(
                [&rows, targetIdx] (ui32 i, float value) {
                    rows[i][targetIdx] = value;
                }
            );
        } else {
            TConstArrayRef<TString> stringTargetPart = std::get<TVector<TString>>((*target)[targetIdx]);

            for (size_t i = offset; i < sliceEnd; ++i) {
                rows[i - offset][targetIdx] = FromString<double>(stringTargetPart[i]);
            }
        }
    }


    for (auto flatFeatureIdx : xrange(featureCount)) {
        TMaybeData<const TFloatValuesHolder*> maybeFeatureData
            = sliceRawObjectsData.GetFloatFeature(flatFeatureIdx);
        if (maybeFeatureData) {
            if (const auto* arrayColumn = dynamic_cast<const TFloatArrayValuesHolder*>(*maybeFeatureData)) {
                arrayColumn->GetData()->ForEach(
                    [&] (ui32 i, float value) {
                        rows[i][flatFeatureIdx + targetCount + 1] = value;
                    }
                );
            } else {
                CB_ENSURE_INTERNAL(false, "CatBoostPoolSlice_R: Unsupported column type");
            }
        } else {
            for (auto i : xrange(sliceRawObjectsData.GetObjectCount())) {
                rows[i][flatFeatureIdx + targetCount + 1] = 0.0f;
            }
        }
    }

    UNPROTECT(2);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostFit_R(SEXP learnPoolParam, SEXP testPoolParam, SEXP fitParamsAsJsonParam) {
    SEXP fitParamsAsJsonParam_char = PROTECT(Rf_asChar(fitParamsAsJsonParam));
    SEXP result = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
    R_API_BEGIN();
    TPoolHandle learnPool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(learnPoolParam));
    TDataProviders pools;
    pools.Learn = learnPool;
    pools.Learn->Ref();

    auto fitParams = LoadFitParams(CHAR(fitParamsAsJsonParam_char));
    TFullModelPtr modelPtr = std::make_unique<TFullModel>();
    if (testPoolParam != R_NilValue) {
        TEvalResult evalResult;
        TPoolHandle testPool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(testPoolParam));
        pools.Test.emplace_back(testPool);
        pools.Test.back()->Ref();
        TrainModel(
            fitParams,
            nullptr,
            Nothing(),
            Nothing(),
            Nothing(),
            pools,
            /*initModel*/ Nothing(),
            /*initLearnProgress*/ nullptr,
            "",
            modelPtr.get(),
            {&evalResult}
        );
    }
    else {
        TrainModel(
            fitParams,
            nullptr,
            Nothing(),
            Nothing(),
            Nothing(),
            pools,
            /*initModel*/ Nothing(),
            /*initLearnProgress*/ nullptr,
            "",
            modelPtr.get(),
            {}
        );
    }
    R_SetExternalPtrAddr(result, modelPtr.get());
    R_RegisterCFinalizerEx(result, _Finalizer<TFullModelHandle>, TRUE);
    modelPtr.release();
    UNPROTECT(2);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostSumModels_R(SEXP modelsParam,
                         SEXP weightsParam,
                         SEXP ctrMergePolicyParam) {
    SEXP result =  PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
    SEXP weightsParam_char = PROTECT(Rf_asChar(weightsParam));
    SEXP ctrMergePolicyParam_char = PROTECT(Rf_asChar(ctrMergePolicyParam));
    R_API_BEGIN();
    const auto& weights = GetVectorFromSEXP<double>(weightsParam);
    ECtrTableMergePolicy mergePolicy;
    CB_ENSURE(TryFromString<ECtrTableMergePolicy>(CHAR(weightsParam_char), mergePolicy),
        "Unknown value of ctr_table_merge_policy: " << CHAR(ctrMergePolicyParam_char));

    TVector<TFullModelConstPtr> models;
    for (int idx = 0; idx < length(modelsParam); ++idx) {
        TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(VECTOR_ELT(modelsParam, idx)));
        models.push_back(model);
    }
    TFullModelPtr modelPtr = std::make_unique<TFullModel>();
    SumModels(models, weights, mergePolicy).Swap(*modelPtr);
    R_SetExternalPtrAddr(result, modelPtr.get());
    R_RegisterCFinalizerEx(result, _Finalizer<TFullModelHandle>, TRUE);
    modelPtr.release();
    UNPROTECT(3);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostCV_R(SEXP fitParamsAsJsonParam,
                  SEXP poolParam,
                  SEXP foldCountParam,
                  SEXP typeParam,
                  SEXP partitionRandomSeedParam,
                  SEXP shuffleParam,
                  SEXP stratifiedParam) {
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    size_t n_protected = 1;
    R_API_BEGIN();
    size_t metricCount;
    size_t columnCount;

    TPoolPtr pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    pool->Ref();
    SEXP fitParamsAsJsonParam_char = PROTECT(safe_R_asChar(fitParamsAsJsonParam, uw_token));
    n_protected++;
    auto fitParams = LoadFitParams(CHAR(fitParamsAsJsonParam_char));

    TCrossValidationParams cvParams;
    cvParams.FoldCount = asInteger(foldCountParam);
    cvParams.PartitionRandSeed = asInteger(partitionRandomSeedParam);
    cvParams.Shuffle = asLogical(shuffleParam);
    cvParams.Stratified = asLogical(stratifiedParam);

    SEXP typeParam_char = PROTECT(safe_R_asChar(typeParam, uw_token));
    n_protected++;
    CB_ENSURE(TryFromString<ECrossValidation>(CHAR(typeParam_char), cvParams.Type),
              "unsupported type of cross_validation: 'Classical', 'Inverted', 'TimeSeries' was expected");

    TVector<TCVResult> cvResults;

    CrossValidate(
        fitParams,
        TQuantizedFeaturesInfoPtr(nullptr),
        Nothing(),
        Nothing(),
        pool,
        cvParams,
        &cvResults);

    metricCount = cvResults.size();
    TVector<size_t> offsets(metricCount);

    columnCount = 0;
    size_t currentColumnCount = 0;
    for (size_t metricIdx = 0; metricIdx < metricCount; ++metricIdx) {
        offsets[metricIdx] = columnCount;
        if (cvResults[metricIdx].AverageTrain.size() == 0) {
            currentColumnCount = 2;
        } else {
            currentColumnCount = 4;
        }
        columnCount += currentColumnCount;
    }

    SEXP result = PROTECT(safe_R_vector(VECSXP, columnCount, uw_token));
    SEXP columnNames = PROTECT(safe_R_vector(STRSXP, columnCount, uw_token));
    n_protected += 2;

    for (size_t metricIdx = 0; metricIdx < metricCount; ++metricIdx) {
        TString metricName = cvResults[metricIdx].Metric;
        size_t numberOfIterations = cvResults[metricIdx].Iterations.size();

        SEXP row_test_mean = PROTECT(safe_R_vector(REALSXP, numberOfIterations, uw_token));
        SEXP row_test_std = PROTECT(safe_R_vector(REALSXP, numberOfIterations, uw_token));
        n_protected += 2;
        SEXP row_train_mean = NULL;
        SEXP row_train_std = NULL;
        const bool haveTrainResult = (cvResults[metricIdx].AverageTrain.size() != 0);
        if (haveTrainResult) {
            row_train_mean = PROTECT(safe_R_vector(REALSXP, numberOfIterations, uw_token));
            row_train_std = PROTECT(safe_R_vector(REALSXP, numberOfIterations, uw_token));
            n_protected += 2;
        }

        for (size_t i = 0; i < numberOfIterations; ++i) {
            REAL(row_test_mean)[i] = cvResults[metricIdx].AverageTest[i];
            REAL(row_test_std)[i] = cvResults[metricIdx].StdDevTest[i];
            if (haveTrainResult) {
                REAL(row_train_mean)[i] = cvResults[metricIdx].AverageTrain[i];
                REAL(row_train_std)[i] = cvResults[metricIdx].StdDevTrain[i];
            }
        }

        const size_t offset = offsets[metricIdx];

        SET_VECTOR_ELT(result, offset + 0, row_test_mean);
        SET_VECTOR_ELT(result, offset + 1, row_test_std);

        SET_STRING_ELT(columnNames, offset + 0,
                       safe_R_mkChar(("test-" + metricName + "-mean").c_str(), uw_token));
        SET_STRING_ELT(columnNames, offset + 1,
                       safe_R_mkChar(("test-" + metricName + "-std").c_str(), uw_token));
        if (haveTrainResult) {
            SET_VECTOR_ELT(result, offset + 2, row_train_mean);
            SET_VECTOR_ELT(result, offset + 3, row_train_std);

            SET_STRING_ELT(columnNames, offset + 2,
                           safe_R_mkChar(("train-" + metricName + "-mean").c_str(), uw_token));
            SET_STRING_ELT(columnNames, offset + 3,
                           safe_R_mkChar(("train-" + metricName + "-std").c_str(), uw_token));
        }
    }

    setAttrib(result, R_NamesSymbol, columnNames);

    UNPROTECT(n_protected);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostOutputModel_R(SEXP modelParam, SEXP fileParam,
                           SEXP formatParam, SEXP exportParametersParam, SEXP poolParam) {
    SEXP result = PROTECT(ScalarLogical(1));
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    THashMap<ui32, TString> catFeaturesHashToString;
    TVector<TString> featureId;

    if (poolParam != R_NilValue) {
        TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
        catFeaturesHashToString = MergeCatFeaturesHashToString(*pool->ObjectsData.Get());
        featureId = pool->MetaInfo.FeaturesLayout.Get()->GetExternalFeatureIds();
    }

    SEXP formatParam_char = PROTECT(safe_R_asChar(formatParam, uw_token));
    SEXP fileParam_char = PROTECT(safe_R_asChar(fileParam, uw_token));
    SEXP exportParametersParam_char = PROTECT(safe_R_asChar(exportParametersParam, uw_token));
    EModelType modelType;
    CB_ENSURE(TryFromString<EModelType>(CHAR(formatParam_char), modelType),
              "unsupported model type: 'cbm', 'coreml', 'cpp', 'python', 'json', 'onnx' or 'pmml' was expected");

    ExportModel(*model,
                CHAR(fileParam_char),
                modelType,
                CHAR(exportParametersParam_char),
                false,
                &featureId,
                &catFeaturesHashToString
                );
    UNPROTECT(5);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostReadModel_R(SEXP fileParam, SEXP formatParam) {
    SEXP formatParam_char = PROTECT(Rf_asChar(formatParam));
    SEXP fileParam_char = PROTECT(Rf_asChar(fileParam));
    SEXP result = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
    R_API_BEGIN();
    EModelType modelType;
    CB_ENSURE(TryFromString<EModelType>(CHAR(formatParam_char), modelType),
              "unsupported model type: 'CatboostBinary', 'AppleCoreML','Cpp','Python','Json','Onnx' or 'Pmml'  was expected");
    TFullModelPtr modelPtr = std::make_unique<TFullModel>();
    ReadModel(CHAR(fileParam_char), modelType).Swap(*modelPtr);
    R_SetExternalPtrAddr(result, modelPtr.get());
    R_RegisterCFinalizerEx(result, _Finalizer<TFullModelHandle>, TRUE);
    modelPtr.release();
    UNPROTECT(3);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostSerializeModel_R(SEXP handleParam) {
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    R_API_BEGIN();
    TFullModelHandle modelHandle = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(handleParam));
    const TString& raw = SerializeModel(*modelHandle);
    SEXP result = PROTECT(safe_R_vector(RAWSXP, raw.size(), uw_token));
    MemCopy(RAW(result), (const unsigned char*)(raw.data()), raw.size());
    UNPROTECT(2);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostDeserializeModel_R(SEXP rawParam) {
    SEXP result = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
    R_API_BEGIN();
    TFullModelPtr modelPtr = std::make_unique<TFullModel>();
    DeserializeModel(TMemoryInput(RAW(rawParam), length(rawParam))).Swap(*modelPtr);
    R_SetExternalPtrAddr(result, modelPtr.get());
    R_RegisterCFinalizerEx(result, _Finalizer<TFullModelHandle>, TRUE);
    modelPtr.release();
    UNPROTECT(1);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostPredictMulti_R(SEXP modelParam, SEXP poolParam, SEXP verboseParam,
                            SEXP typeParam, SEXP treeCountStartParam, SEXP treeCountEndParam, SEXP threadCountParam) {
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    R_API_BEGIN();
    SEXP typeParam_char = PROTECT(safe_R_asChar(typeParam, uw_token));
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    EPredictionType predictionType;
    CB_ENSURE(TryFromString<EPredictionType>(CHAR(typeParam_char), predictionType) &&
              !IsUncertaintyPredictionType(predictionType) && predictionType != EPredictionType::InternalRawFormulaVal,
              "Unsupported prediction type: 'Probability', 'LogProbability', 'Class', 'RawFormulaVal', 'Exponent' or 'RMSEWithUncertainty' was expected");
    TVector<TVector<double>> prediction = ApplyModelMulti(*model,
                                                          *pool,
                                                          asLogical(verboseParam),
                                                          predictionType,
                                                          asInteger(treeCountStartParam),
                                                          asInteger(treeCountEndParam),
                                                          UpdateThreadCount(asInteger(threadCountParam)));
    size_t predictionSize = prediction.size() * pool->ObjectsGrouping->GetObjectCount();
    SEXP result = PROTECT(safe_R_vector(REALSXP, predictionSize, uw_token));
    double *ptr_result = REAL(result);
    for (size_t i = 0, k = 0; i < pool->ObjectsGrouping->GetObjectCount(); ++i) {
        for (size_t j = 0; j < prediction.size(); ++j) {
            ptr_result[k++] = prediction[j][i];
        }
    }
    UNPROTECT(3);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostPrepareEval_R(SEXP approxParam, SEXP typeParam, SEXP lossFunctionName, SEXP columnCountParam, SEXP threadCountParam) {
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    R_API_BEGIN();
    SEXP typeParam_char = PROTECT(safe_R_asChar(typeParam, uw_token));
    SEXP lossFunctionName_char = PROTECT(safe_R_asChar(lossFunctionName, uw_token));
    SEXP dataDim = getAttrib(approxParam, R_DimSymbol);
    size_t dataRows = static_cast<size_t>(INTEGER(dataDim)[0]) / asInteger(columnCountParam);
    TVector<TVector<double>> prediction(asInteger(columnCountParam), TVector<double>(dataRows));
    double *ptr_approxParam = Rf_isNull(approxParam)? nullptr : REAL(approxParam);
    for (size_t i = 0, k = 0; i < dataRows; ++i) {
        for (size_t j = 0; j < prediction.size(); ++j) {
            prediction[j][i] = static_cast<double>(ptr_approxParam[k++]);
        }
    }

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(UpdateThreadCount(asInteger(threadCountParam)) - 1);
    EPredictionType predictionType;
    CB_ENSURE(TryFromString<EPredictionType>(CHAR(typeParam_char), predictionType),
              "unsupported prediction type: 'Probability', 'Class' or 'RawFormulaVal' was expected");
    prediction = PrepareEval(predictionType, /* virtualEnsemblesCount*/ 1, CHAR(lossFunctionName_char), prediction, &executor);

    size_t predictionSize = prediction.size() * dataRows;
    SEXP result = PROTECT(safe_R_vector(REALSXP, predictionSize, uw_token));
    double *ptr_result = REAL(result);
    for (size_t i = 0, k = 0; i < dataRows; ++i) {
        for (size_t j = 0; j < prediction.size(); ++j) {
            ptr_result[k++] = prediction[j][i];
        }
    }
    UNPROTECT(4);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostPredictVirtualEnsembles_R(SEXP modelParam, SEXP poolParam, SEXP verboseParam,
                            SEXP typeParam, SEXP treeCountEndParam, SEXP virtualEnsemblesCountParam, SEXP threadCountParam) {
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    R_API_BEGIN();
    SEXP typeParam_char = PROTECT(safe_R_asChar(typeParam, uw_token));
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    EPredictionType predictionType;
    CB_ENSURE(TryFromString<EPredictionType>(CHAR(typeParam_char), predictionType) && IsUncertaintyPredictionType(predictionType),
              "Unsupported virtual ensembles prediction type: 'VirtEnsembles' or 'TotalUncertainty' was expected");
    TVector<TVector<double>> prediction = ApplyUncertaintyPredictions(*model,
                                                                      *pool,
                                                                      asLogical(verboseParam),
                                                                      predictionType,
                                                                      asInteger(treeCountEndParam),
                                                                      asInteger(virtualEnsemblesCountParam),
                                                                      UpdateThreadCount(asInteger(threadCountParam)));
    size_t predictionSize = prediction.size() * pool->ObjectsGrouping->GetObjectCount();
    SEXP result = PROTECT(safe_R_vector(REALSXP, predictionSize, uw_token));
    double *ptr_result = REAL(result);
    for (size_t i = 0, k = 0; i < pool->ObjectsGrouping->GetObjectCount(); ++i) {
        for (size_t j = 0; j < prediction.size(); ++j) {
            ptr_result[k++] = prediction[j][i];
        }
    }
    UNPROTECT(3);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostShrinkModel_R(SEXP modelParam, SEXP treeCountStartParam, SEXP treeCountEndParam) {
    SEXP result = PROTECT(ScalarLogical(1));
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    model->Truncate(asInteger(treeCountStartParam), asInteger(treeCountEndParam));
    UNPROTECT(1);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostDropUnusedFeaturesFromModel_R(SEXP modelParam) {
    SEXP result = PROTECT(ScalarLogical(1));
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    model->ModelTrees.GetMutable()->DropUnusedFeatures();
    UNPROTECT(1);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostGetModelParams_R(SEXP modelParam) {
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    auto temp = model->ModelInfo.at("params");
    SEXP result = PROTECT(safe_R_mkString(temp.c_str(), uw_token));
    UNPROTECT(2);
    return result;
    R_API_END();
}


EXPORT_FUNCTION CatBoostGetPlainParams_R(SEXP modelParam) {
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    auto temp = ToString(GetPlainJsonWithAllOptions(*model)).c_str();
    SEXP result = PROTECT(safe_R_mkString(temp, uw_token));
    UNPROTECT(2);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostCalcRegularFeatureEffect_R(SEXP modelParam, SEXP poolParam, SEXP fstrTypeParam, SEXP threadCountParam) {
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    R_API_BEGIN();
    SEXP result = NULL;
    SEXP resultDim = NULL;
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TDataProviderPtr pool = Rf_isNull(poolParam) ? nullptr :
                            reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    if (pool) {
        pool->Ref();
    }
    SEXP fstrTypeParam_char = PROTECT(safe_R_asChar(fstrTypeParam, uw_token));
    EFstrType fstrType = FromString<EFstrType>(CHAR(fstrTypeParam_char));
    const int threadCount = UpdateThreadCount(asInteger(threadCountParam));
    const bool multiClass = model->GetDimensionsCount() > 1;
    const bool verbose = false;
    // TODO(akhropov): make prettified mode as in python-package
    if (fstrType == EFstrType::ShapValues && multiClass) {
        TVector<TVector<TVector<double>>> fstr = GetFeatureImportancesMulti(fstrType,
                                                                            *model,
                                                                            pool,
                                                                            /*referenceDataset*/ nullptr,
                                                                            threadCount,
                                                                            EPreCalcShapValues::Auto,
                                                                            verbose);
        size_t numDocs = fstr.size();
        size_t numClasses = numDocs > 0 ? fstr[0].size() : 0;
        size_t numValues = numClasses > 0 ? fstr[0][0].size() : 0;
        size_t resultSize = numDocs * numClasses * numValues;
        result = PROTECT(safe_R_vector(REALSXP, resultSize, uw_token));
        double *ptr_result = REAL(result);
        size_t r = 0;
        for (size_t k = 0; k < numValues; ++k) {
            for (size_t j = 0; j < numClasses; ++j) {
               for (size_t i = 0; i < numDocs; ++i) {
                    ptr_result[r++] = fstr[i][j][k];
                }
            }
        }
        resultDim = PROTECT(safe_R_vector(INTSXP, 3, uw_token));
        INTEGER(resultDim)[0] = numDocs;
        INTEGER(resultDim)[1] = numClasses;
        INTEGER(resultDim)[2] = numValues;
        setAttrib(result, R_DimSymbol, resultDim);
    } else {
        TVector<TVector<double>> fstr = GetFeatureImportances(fstrType,
                                                              *model,
                                                              pool,
                                                              /*referenceDataset*/ nullptr,
                                                              threadCount,
                                                              EPreCalcShapValues::Auto,
                                                              verbose);
        size_t numRows = fstr.size();
        size_t numCols = numRows > 0 ? fstr[0].size() : 0;
        size_t resultSize = numRows * numCols;
        result = PROTECT(safe_R_vector(REALSXP, resultSize, uw_token));
        double *ptr_result = REAL(result);
        size_t r = 0;
        for (size_t j = 0; j < numCols; ++j) {
            for (size_t i = 0; i < numRows; ++i) {
                ptr_result[r++] = fstr[i][j];
            }
        }
        resultDim = PROTECT(safe_R_vector(INTSXP, 2, uw_token));
        INTEGER(resultDim)[0] = numRows;
        INTEGER(resultDim)[1] = numCols;
        setAttrib(result, R_DimSymbol, resultDim);
    }
    UNPROTECT(4);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostEvaluateObjectImportances_R(
    SEXP modelParam,
    SEXP poolParam,
    SEXP trainPoolParam,
    SEXP topSizeParam,
    SEXP ostrTypeParam,
    SEXP updateMethodParam,
    SEXP threadCountParam
) {
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    R_API_BEGIN();
    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));
    TPoolHandle trainPool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(trainPoolParam));
    SEXP ostrTypeParam_char = PROTECT(safe_R_asChar(ostrTypeParam, uw_token));
    SEXP updateMethodParam_char = PROTECT(safe_R_asChar(updateMethodParam, uw_token));
    TString ostrType = CHAR(ostrTypeParam_char);
    TString updateMethod = CHAR(updateMethodParam_char);
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
    SEXP result = PROTECT(safe_R_vector(REALSXP, resultSize, uw_token));
    double *ptr_result = REAL(result);
    size_t k = 0;
    for (size_t i = 0; i < dstrResult.Indices.size(); ++i) {
        for (size_t j = 0; j < dstrResult.Indices[0].size(); ++j) {
            ptr_result[k++] = dstrResult.Indices[i][j];
        }
    }
    for (size_t i = 0; i < dstrResult.Scores.size(); ++i) {
        for (size_t j = 0; j < dstrResult.Scores[0].size(); ++j) {
            ptr_result[k++] = dstrResult.Scores[i][j];
        }
    }
    UNPROTECT(4);
    return result;
    R_API_END();
}

EXPORT_FUNCTION CatBoostIsNullHandle_R(SEXP handleParam) {
    return ScalarLogical(!R_ExternalPtrAddr(handleParam));
}

EXPORT_FUNCTION CatBoostEvalMetrics_R(
        SEXP modelParam,
        SEXP poolParam,
        SEXP metricsParam,
        SEXP treeCountStartParam,
        SEXP treeCountEndParam,
        SEXP evalPeriodParam,
        SEXP threadCountParam,
        SEXP tmpDirParam,
        SEXP resultDirParam) {
    SEXP uw_token = PROTECT(R_MakeUnwindCont());
    size_t n_protected = 1;
    R_API_BEGIN()

    auto treeCountStart = asInteger(treeCountStartParam);
    auto treeCountEnd = asInteger(treeCountEndParam);
    auto evalPeriod = asInteger(evalPeriodParam);

    CB_ENSURE(treeCountStart >= 0, "Tree start index should be greater or equal zero");
    CB_ENSURE(treeCountStart < treeCountEnd, "Tree start index should be less than tree end index");
    CB_ENSURE(evalPeriod <= (treeCountEnd - treeCountStart), "Eval period should be less or equal than number of trees");
    CB_ENSURE(evalPeriod > 0, "Eval period should be more than zero");

    size_t metricsParamLen = Rf_xlength(metricsParam);
    size_t treeCount = treeCountEnd - treeCountStart;
    size_t numberOfIterations = treeCount / evalPeriod;
    if (treeCountStart + (numberOfIterations - 1) * evalPeriod != static_cast<size_t>(treeCountEnd) - 1) {
        ++numberOfIterations;
    }

    SEXP result = PROTECT(safe_R_vector(VECSXP, metricsParamLen, uw_token));
    SEXP metricNames = PROTECT(safe_R_vector(STRSXP, metricsParamLen, uw_token));
    n_protected += 2;

    for (size_t metricIdx = 0; metricIdx < metricsParamLen; ++metricIdx) {
        SET_VECTOR_ELT(result, metricIdx, safe_R_vector(REALSXP, numberOfIterations, uw_token));
    }

    TFullModelHandle model = reinterpret_cast<TFullModelHandle>(R_ExternalPtrAddr(modelParam));
    TPoolHandle pool = reinterpret_cast<TPoolHandle>(R_ExternalPtrAddr(poolParam));

    TVector<TString> metricDescriptions;
    TVector<NCatboostOptions::TLossDescription> metricLossDescriptions;
    metricDescriptions.reserve(metricsParamLen);
    metricLossDescriptions.reserve(metricsParamLen);
    for (size_t i = 0; i < metricsParamLen; ++i) {
        SEXP temp = PROTECT(safe_R_asChar(VECTOR_ELT(metricsParam, i), uw_token));
        n_protected++;
        TString metricDescription = CHAR(temp);
        metricDescriptions.push_back(metricDescription);
        metricLossDescriptions.emplace_back(NCatboostOptions::ParseLossDescription(metricDescription));
    }
    auto metrics = CreateMetrics(metricLossDescriptions, model->GetDimensionsCount());
    SEXP tmpDirParam_char = PROTECT(safe_R_asChar(tmpDirParam, uw_token));
    n_protected++;

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(UpdateThreadCount(asInteger(threadCountParam)) - 1);

    TMetricsPlotCalcer plotCalcer = CreateMetricCalcer(
        *model,
        treeCountStart,
        treeCountEnd,
        evalPeriod,
        /*processedIterationsStep=*/50,
        CHAR(tmpDirParam_char),
        metrics,
        &executor
    );

    TRestorableFastRng64 rand(0);
    auto processedDataProvider = CreateModelCompatibleProcessedDataProvider(
        *pool,
        metricLossDescriptions,
        *model,
        GetMonopolisticFreeCpuRam(),
        &rand,
        &executor
    );

    if (plotCalcer.HasAdditiveMetric()) {
        plotCalcer.ProceedDataSetForAdditiveMetrics(processedDataProvider);
    }
    if (plotCalcer.HasNonAdditiveMetric()) {
        while (!plotCalcer.AreAllIterationsProcessed()) {
            plotCalcer.ProceedDataSetForNonAdditiveMetrics(processedDataProvider);
            plotCalcer.FinishProceedDataSetForNonAdditiveMetrics();
        }
    }

    TVector<TVector<double>> metricsScore = plotCalcer.GetMetricsScore();
    SEXP resultDirParam_char = PROTECT(safe_R_asChar(resultDirParam, uw_token));
    n_protected++;
    plotCalcer.SaveResult(CHAR(resultDirParam_char), /*metricsFile=*/"", /*saveMetrics*/ false, /*saveStats=*/true).ClearTempFiles();

    auto metricsResult = CreateMetricsFromDescription(metricDescriptions, model->GetDimensionsCount());
    for (size_t metricIdx = 0; metricIdx < metricsParamLen; ++metricIdx) {
        TString metricName = metricsResult[metricIdx]->GetDescription();
        SEXP metricScoreResult = VECTOR_ELT(result, metricIdx);
        double *ptr_metricScoreResult = REAL(metricScoreResult);
        for (size_t i = 0; i < numberOfIterations; ++i) {
            ptr_metricScoreResult[i] = metricsScore[metricIdx][i];
        }
        SET_STRING_ELT(metricNames, metricIdx, safe_R_mkChar(metricName.c_str(), uw_token));
    }

    setAttrib(result, R_NamesSymbol, metricNames);

    UNPROTECT(n_protected);
    return result;
    R_API_END();
}
}
