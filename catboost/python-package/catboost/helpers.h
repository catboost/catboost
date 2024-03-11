#pragma once

#include <Python.h>

#include <catboost/private/libs/algo/plot.h>
#include <catboost/private/libs/data_types/groupid.h>
#include <catboost/private/libs/data_types/pair.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/visitor.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/private/libs/options/plain_options_helper.h>
#include <catboost/private/libs/target/data_providers.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/train_lib/options_helper.h>

#include <library/cpp/json/json_value.h>
#include <library/cpp/threading/local_executor/tbb_local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/noncopyable.h>
#include <util/generic/ptr.h>
#include <util/generic/xrange.h>

#include <future>
#include <type_traits>


struct TTrainTestSplitParams;

namespace NCatboostOptions {
    struct TDatasetReadingParams;
}


class TGilGuard : public TNonCopyable {
public:
    TGilGuard()
        : State_(PyGILState_Ensure())
    { }

    ~TGilGuard() {
        PyGILState_Release(State_);
    }
private:
    PyGILState_STATE State_;
};

void ProcessException();
void SetPythonInterruptHandler();
void ResetPythonInterruptHandler();
void ThrowCppExceptionWithMessage(const TString& message);

TVector<TVector<double>> EvalMetrics(
    const TFullModel& model,
    const NCB::TDataProvider& srcData,
    const TVector<TString>& metricsDescription,
    int begin,
    int end,
    int evalPeriod,
    int threadCount,
    const TString& resultDir,
    const TString& tmpDir
);

TVector<TString> GetMetricNames(const TFullModel& model, const TVector<TString>& metricsDescription);

TVector<double> EvalMetricsForUtils(
    TConstArrayRef<TVector<float>> label,
    const TVector<TVector<double>>& approx,
    const TString& metricName,
    const TVector<float>& weight,
    const TVector<TGroupId>& groupId,
    const TVector<float>& groupWeight,
    const TVector<TSubgroupId>& subgroupId,
    const TVector<TPair>& pairs,
    int threadCount
);

inline TVector<NCatboostOptions::TLossDescription> CreateMetricLossDescriptions(
    const TVector<TString>& metricDescriptions) {

    CB_ENSURE(!metricDescriptions.empty(), "No metrics in metric descriptions");

    TVector<NCatboostOptions::TLossDescription> result;
    for (const auto& metricDescription : metricDescriptions) {
        result.emplace_back(NCatboostOptions::ParseLossDescription(metricDescription));
    }

    return result;
}

#if PY_MAJOR_VERSION < 3
inline const char* PyUnicode_AsUTF8AndSize(PyObject *unicode, Py_ssize_t *size) {
    return nullptr;
}
#endif


class TMetricsPlotCalcerPythonWrapper {
public:
    TMetricsPlotCalcerPythonWrapper(const TVector<TString>& metricDescriptions,
                                    const TFullModel& model,
                                    int begin,
                                    int end,
                                    int evalPeriod,
                                    int threadCount,
                                    const TString& tmpDir,
                                    bool deleteTempDirOnExit = false)
    : Rand(0)
    , MetricLossDescriptions(CreateMetricLossDescriptions(metricDescriptions))
    , Metrics(CreateMetrics(MetricLossDescriptions, model.GetDimensionsCount()))
    , MetricPlotCalcer(CreateMetricCalcer(
            model,
            begin,
            end,
            evalPeriod,
            /*processedIterationsStep=*/-1,
            tmpDir,
            Metrics,
            &Executor)) {
        Executor.RunAdditionalThreads(threadCount - 1);
        MetricPlotCalcer.SetDeleteTmpDirOnExit(deleteTempDirOnExit);
    }

    ~TMetricsPlotCalcerPythonWrapper() {
        MetricPlotCalcer.ClearTempFiles();
    }

    void AddPool(const NCB::TDataProvider& srcData) {
        auto processedDataProvider = NCB::CreateModelCompatibleProcessedDataProvider(
            srcData,
            MetricLossDescriptions,
            MetricPlotCalcer.GetModel(),
            NCB::GetMonopolisticFreeCpuRam(),
            &Rand,
            &Executor
        );

        if (MetricPlotCalcer.HasAdditiveMetric()) {
            MetricPlotCalcer.ProceedDataSetForAdditiveMetrics(processedDataProvider);
        }
        if (MetricPlotCalcer.HasNonAdditiveMetric()) {
            MetricPlotCalcer.ProceedDataSetForNonAdditiveMetrics(processedDataProvider);
        }

    }

    TVector<const IMetric*> GetMetricRawPtrs() const {
        TVector<const IMetric*> ptrs;
        for (const auto& metric : Metrics) {
            ptrs.push_back(metric.Get());
        }
        return ptrs;
    }

    TVector<TVector<double>> ComputeScores()  {
        if (MetricPlotCalcer.HasNonAdditiveMetric()) {
            MetricPlotCalcer.FinishProceedDataSetForNonAdditiveMetrics();
        }
        return MetricPlotCalcer.GetMetricsScore();
    }

private:
    TRestorableFastRng64 Rand;
    NPar::TLocalExecutor Executor;
    TVector<NCatboostOptions::TLossDescription> MetricLossDescriptions;
    TVector<THolder<IMetric>> Metrics;
    TMetricsPlotCalcer MetricPlotCalcer;
};

NJson::TJsonValue GetTrainingOptions(
    const NJson::TJsonValue& plainJsonParams,
    const NCB::TDataMetaInfo& trainDataMetaInfo,
    const TMaybe<NCB::TDataMetaInfo>& testDataMetaInfo
);

class TPythonStreamWrapper : public IInputStream {

public:
    using TReadFunction = std::function<size_t(char*, size_t, PyObject*, TString*)>;

    TPythonStreamWrapper(TReadFunction func, PyObject* stream): ReadFunc(func), Stream(stream) {}

protected:
    size_t DoRead(void *buf, size_t len) override {
        TString errStr;
        size_t result = ReadFunc(static_cast<char*>(buf), len, Stream, &errStr);

        CB_ENSURE(result != static_cast<size_t>(-1), errStr);
        return result;
    }

private:
    TReadFunction ReadFunc;
    PyObject* Stream;
};


template <class TFloatOrInteger>
void AsyncSetDataFromCythonMemoryViewCOrder(
    ui32 objCount,
    const TFloatOrInteger* data,
    size_t objStride,    // dim 0
    size_t elementStride,   // dim 1
    bool hasSeparateEmbeddingFeaturesData,
    TConstArrayRef<ui32> mainDataFeatureIdxToDstFeatureIdx,
    TConstArrayRef<bool> isCatFeature,  // can be empty, it means no categorical data
    NCB::IRawObjectsOrderDataVisitor* builderVisitor,
    NPar::ILocalExecutor* localExecutor,
    std::future<void>* result
) {
    *result = std::move(
        std::async(
            [=]() {
                if (isCatFeature) {
                    NPar::ParallelFor(
                        *localExecutor,
                        0,
                        objCount,
                        [=] (ui32 objIdx) {
                            const TFloatOrInteger* dataPtr = data + objIdx * objStride;
                            const auto value = *dataPtr;
                            for (auto featureIdx : mainDataFeatureIdxToDstFeatureIdx) {
                                if (isCatFeature[featureIdx]) {
                                    const auto isFloat
                                        = std::is_same<TFloatOrInteger, float>::value
                                            || std::is_same<TFloatOrInteger, double>::value;
                                    CB_ENSURE(
                                        !isFloat,
                                        "Invalid value for cat_feature[" << objIdx << "," << featureIdx << "]="
                                         << value << " cat_features must be integer or string. Real numbers"
                                         "and NaNs should be converted to strings."
                                    );
                                    const auto catValue = ToString(value);
                                    builderVisitor->AddCatFeature(objIdx, featureIdx, catValue);
                                } else {
                                    builderVisitor->AddFloatFeature(objIdx, featureIdx, value);
                                }
                                dataPtr += elementStride;
                            }
                        }
                    );
                } else {
                    if ((elementStride == 1) && std::is_same<TFloatOrInteger, float>::value) {
                        size_t featureCount = mainDataFeatureIdxToDstFeatureIdx.size();
                        NPar::ParallelFor(
                            *localExecutor,
                            0,
                            objCount,
                            [=] (ui32 objIdx) {
                                const TFloatOrInteger* dataPtr = data + objIdx * objStride;
                                builderVisitor->AddAllFloatFeatures(
                                    objIdx,
                                    TConstArrayRef<float>(dataPtr, featureCount)
                                );
                            }
                        );
                    } else {
                        NPar::ParallelFor(
                            *localExecutor,
                            0,
                            objCount,
                            [=] (ui32 objIdx) {
                                const TFloatOrInteger* dataPtr = data + objIdx * objStride;
                                for (auto featureIdx : mainDataFeatureIdxToDstFeatureIdx) {
                                    builderVisitor->AddFloatFeature(objIdx, featureIdx, *dataPtr);
                                    dataPtr += elementStride;
                                }
                            }
                        );
                    }
                }
            }
        )
    );
}


template <typename TFloatOrInteger>
void SetDataFromScipyCsrSparse(
    TConstArrayRef<ui32> indptr,
    TConstArrayRef<TFloatOrInteger> data,
    TConstArrayRef<ui32> indices,
    bool hasSeparateEmbeddingFeaturesData,
    TConstArrayRef<ui32> mainDataFeatureIdxToDstFeatureIdx,
    TConstArrayRef<bool> isCatFeature,
    NCB::IRawObjectsOrderDataVisitor* builderVisitor,
    NPar::ILocalExecutor* localExecutor
) {
    CB_ENSURE_INTERNAL(indptr.size() > 1, "Empty sparse arrays should be processed in Python for speed");
    const auto objCount = indptr.size() - 1;

    const auto catFeatureCount = Accumulate(isCatFeature, 0);
    if (catFeatureCount == 0) {
        const ui32 featureCount = isCatFeature.size();
        NPar::ParallelFor(
            *localExecutor,
            0,
            objCount,
            [=] (ui32 objIdx) {
                const auto nonzeroBegin = indptr[objIdx];
                const auto nonzeroEnd = indptr[objIdx + 1];

                TVector<ui32> dstIndices;
                if (hasSeparateEmbeddingFeaturesData) {
                    dstIndices.yresize(nonzeroEnd - nonzeroBegin);
                    for (auto dstIdx : xrange(nonzeroEnd - nonzeroBegin)) {
                        dstIndices[dstIdx] = mainDataFeatureIdxToDstFeatureIdx[indices[nonzeroBegin + dstIdx]];
                    }
                } else {
                    dstIndices.assign(indices.data() + nonzeroBegin, indices.data() + nonzeroEnd);
                }

                const auto features = MakeConstPolymorphicValuesSparseArrayWithArrayIndex(
                    featureCount,
                    NCB::TMaybeOwningConstArrayHolder<ui32>::CreateOwning(std::move(dstIndices)),
                    NCB::TMaybeOwningConstArrayHolder<TFloatOrInteger>::CreateOwning(TVector<TFloatOrInteger>{data.data() + nonzeroBegin, data.data() + nonzeroEnd}),
                    /*ordered*/ true,
                    /*defaultValue*/ 0.0f);
                builderVisitor->AddAllFloatFeatures(objIdx, features);
        });
        return;
    }
    NPar::ParallelFor(
        *localExecutor,
        0,
        objCount,
        [=] (ui32 objIdx) {
            const auto nonzeroBegin = indptr[objIdx];
            const auto nonzeroEnd = indptr[objIdx + 1];
            for (auto nonzeroIdx : xrange(nonzeroBegin, nonzeroEnd, 1)) {
                const auto featureIdx = mainDataFeatureIdxToDstFeatureIdx[indices[nonzeroIdx]];
                const auto value = data[nonzeroIdx];
                if (isCatFeature[featureIdx]) {
                    const auto isFloat = std::is_same<TFloatOrInteger, float>::value || std::is_same<TFloatOrInteger, double>::value;
                    CB_ENSURE(
                        !isFloat,
                        "Invalid value for cat_feature[" << objIdx << "," << featureIdx << "]=" << value <<
                        " cat_features must be integer or string. Real numbers and NaNs should be converted to strings.");
                    const auto catValue = ToString(value);
                    builderVisitor->AddCatFeature(objIdx, featureIdx, catValue);
                } else {
                    builderVisitor->AddFloatFeature(objIdx, featureIdx, value);
                }
            }
        }
    );
}

size_t GetNumPairs(const NCB::TDataProvider& dataProvider) noexcept;
TConstArrayRef<TPair> GetUngroupedPairs(const NCB::TDataProvider& dataProvider);


void TrainEvalSplit(
    const NCB::TDataProvider& srcDataProvider,
    NCB::TDataProviderPtr* trainDataProvider,
    NCB::TDataProviderPtr* evalDataProvider,
    const TTrainTestSplitParams& splitParams,
    bool saveEvalDataset,
    int threadCount,
    ui64 cpuUsedRamLimit
);


TAtomicSharedPtr<NPar::TTbbLocalExecutor<false>> GetCachedLocalExecutor(int threadsCount);

size_t GetMultiQuantileApproxSize(const TString& lossFunctionDescription);

void GetNumFeatureValuesSample(
    const TFullModel& model,
    const NCatboostOptions::TDatasetReadingParams& datasetReadingParams,
    int threadCount,
    const TVector<ui32>& sampleIndicesVector,
    const TVector<TString>& sampleIdsVector,
    TVector<TArrayRef<float>>* numFeaturesColumns
);


TMetricsAndTimeLeftHistory GetTrainingMetrics(const TFullModel& model);
