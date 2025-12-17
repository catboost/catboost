#include "helpers.h"

#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/data/sampler.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/helpers/matrix.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/polymorphic_type_containers.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/options/dataset_reading_params.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/plain_options_helper.h>
#include <catboost/private/libs/options/split_params.h>
#include <catboost/private/libs/target/data_providers.h>

#include <library/cpp/json/json_reader.h>

#include <util/system/guard.h>
#include <util/system/info.h>
#include <util/system/mutex.h>
#include <util/generic/scope.h>
#include <util/string/cast.h>

#if defined(HAVE_CUDA)
#include <catboost/cuda/cuda_lib/helpers.h>
#include <catboost/cuda/cuda_lib/memcpy_tracker.h>
#include <catboost/cuda/data/gpu_input_provider.h>
#include <catboost/cuda/gpu_data/kernel/gpu_input_utils.cuh>

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data/model_dataset_compatibility.h>
#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/model/cuda/evaluator.cuh>
#include <catboost/libs/model/evaluation_interface.h>

#include <cuda_runtime_api.h>
#else
TVector<TVector<double>> ApplyModelMultiGpuInput(
    const TFullModel& model,
    const NCB::TDataProvider& srcData,
    bool verbose,
    EPredictionType predictionType,
    int begin,
    int end,
    int threadCount,
    const TString& devices
) {
    Y_UNUSED(model);
    Y_UNUSED(srcData);
    Y_UNUSED(verbose);
    Y_UNUSED(predictionType);
    Y_UNUSED(begin);
    Y_UNUSED(end);
    Y_UNUSED(threadCount);
    Y_UNUSED(devices);
    ythrow TCatBoostException() << "GPU prediction for GPU-resident inputs requires CatBoost built with CUDA";
}

void ApplyModelMultiGpuInputToDevice(
    const TFullModel& model,
    const NCB::TDataProvider& srcData,
    bool verbose,
    EPredictionType predictionType,
    int begin,
    int end,
    ui64 dstDevicePtr,
    ui32 dstSize,
    const TString& devices
) {
    Y_UNUSED(model);
    Y_UNUSED(srcData);
    Y_UNUSED(verbose);
    Y_UNUSED(predictionType);
    Y_UNUSED(begin);
    Y_UNUSED(end);
    Y_UNUSED(dstDevicePtr);
    Y_UNUSED(dstSize);
    Y_UNUSED(devices);
    ythrow TCatBoostException() << "GPU prediction for GPU-resident inputs requires CatBoost built with CUDA";
}

TVector<ui32> CalcLeafIndexesMultiGpuInput(
    const TFullModel& model,
    const NCB::TDataProvider& srcData,
    bool verbose,
    int begin,
    int end,
    int threadCount
) {
    Y_UNUSED(model);
    Y_UNUSED(srcData);
    Y_UNUSED(verbose);
    Y_UNUSED(begin);
    Y_UNUSED(end);
    Y_UNUSED(threadCount);
    ythrow TCatBoostException() << "GPU prediction for GPU-resident inputs requires CatBoost built with CUDA";
}
#endif


using namespace NCB;


extern "C++" PyObject* PyCatboostExceptionType;

void ProcessException() {
    try {
        throw;
    } catch (const TCatBoostException& exc) {
        PyErr_SetString(PyCatboostExceptionType, exc.what());
    } catch (const TInterruptException& exc) {
        PyErr_SetString(PyExc_KeyboardInterrupt, exc.what());
    } catch (const std::exception& exc) {
        PyErr_SetString(PyCatboostExceptionType, exc.what());
    }
}

void PyCheckInterrupted() {
    TGilGuard guard;
    if (PyErr_CheckSignals() == -1) {
        throw TInterruptException();
    }
}

void SetPythonInterruptHandler() {
    SetInterruptHandler(PyCheckInterrupted);
}

void ResetPythonInterruptHandler() {
    ResetInterruptHandler();
}

void ThrowCppExceptionWithMessage(const TString& message) {
    ythrow TCatBoostException() << message;
}

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
) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);

    TRestorableFastRng64 rand(0);

    auto metricLossDescriptions = CreateMetricLossDescriptions(metricsDescription);
    auto metrics = CreateMetrics(metricLossDescriptions, model.GetDimensionsCount());
    TMetricsPlotCalcer plotCalcer = CreateMetricCalcer(
        model,
        begin,
        end,
        evalPeriod,
        /*processedIterationsStep=*/50,
        tmpDir,
        metrics,
        &executor
    );

    auto processedDataProvider = NCB::CreateModelCompatibleProcessedDataProvider(
        srcData,
        metricLossDescriptions,
        model,
        NCB::GetMonopolisticFreeCpuRam(),
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

    plotCalcer.SaveResult(resultDir, /*metricsFile=*/"", /*saveMetrics*/ false, /*saveStats=*/true).ClearTempFiles();
    return metricsScore;
}

TVector<TString> GetMetricNames(const TFullModel& model, const TVector<TString>& metricsDescription) {
    auto metrics = CreateMetricsFromDescription(metricsDescription, model.GetDimensionsCount());
    TVector<TString> metricNames;
    metricNames.reserve(metrics.ysize());
    for (auto& metric : metrics) {
        metricNames.push_back(metric->GetDescription());
    }
    return metricNames;
}

TVector<double> EvalMetricsForUtils(
    TConstArrayRef<TVector<float>> label,   // [dimensionIdx][objectIdx]
    const TVector<TVector<double>>& approx, // [dimensionIdx][objectIdx]
    const TString& metricName,
    const TVector<float>& weight,
    const TVector<TGroupId>& groupId,
    const TVector<float>& groupWeight,
    const TVector<TSubgroupId>& subgroupId,
    const TVector<TPair>& pairs,
    int threadCount
) {
    auto objectCount = label[0].size();
    CB_ENSURE(objectCount > 0, "Cannot evaluate metric on empty data");

    CB_ENSURE(!IsGroupwiseMetric(metricName) || !groupId.empty(), "Metric \"" << metricName << "\" requires group data");

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    const int approxDimension = approx.ysize();
    TVector<THolder<IMetric>> metrics = CreateMetricsFromDescription({metricName}, approxDimension);
    if (!weight.empty()) {
        for (auto& metric : metrics) {
            metric->UseWeights.SetDefaultValue(true);
        }
    }
    NCB::TObjectsGrouping objectGrouping = NCB::CreateObjectsGroupingFromGroupIds<TGroupId>(
        objectCount,
        groupId.empty() ? Nothing() : NCB::TMaybeData<TConstArrayRef<TGroupId>>(groupId)
    );
    if (!pairs.empty()) {
        NCB::CheckPairs(pairs, objectGrouping);
    }
    TVector<TQueryInfo> queriesInfo;
    if (!groupId.empty()) {
        queriesInfo = *NCB::MakeGroupInfos(
            objectGrouping,
            subgroupId.empty() ? Nothing() : NCB::TMaybeData<TConstArrayRef<TSubgroupId>>(subgroupId),
            groupWeight.empty() ? NCB::TWeights(groupId.size()) : NCB::TWeights(TVector<float>(groupWeight)),
            TConstArrayRef<TPair>(pairs)
        ).Get();
    }
    TVector<double> metricResults;
    metricResults.reserve(metrics.size());

    TVector<const IMetric*> metricPtrs;
    metricPtrs.reserve(metrics.size());
    for (const auto& metric : metrics) {
        metricPtrs.push_back(metric.Get());
    }

    auto stats = EvalErrorsWithCaching(
        approx,
        /*approxDelts*/{},
        /*isExpApprox*/false,
        To2DConstArrayRef<float>(label),
        weight,
        queriesInfo,
        metricPtrs,
        &executor
    );

    for (auto metricIdx : xrange(metricPtrs.size())) {
        metricResults.push_back(metricPtrs[metricIdx]->GetFinalError(stats[metricIdx]));
    }
    return metricResults;
}

NJson::TJsonValue GetTrainingOptions(
    const NJson::TJsonValue& plainJsonParams,
    const NCB::TDataMetaInfo& trainDataMetaInfo,
    const TMaybe<NCB::TDataMetaInfo>& testDataMetaInfo
) {
    NJson::TJsonValue trainOptionsJson;
    NJson::TJsonValue outputFilesOptionsJson;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &trainOptionsJson, &outputFilesOptionsJson);
    ConvertParamsToCanonicalFormat(trainDataMetaInfo, &trainOptionsJson);
    NCatboostOptions::TCatBoostOptions catboostOptions(NCatboostOptions::LoadOptions(trainOptionsJson));
    NCatboostOptions::TOutputFilesOptions outputOptions;
    outputOptions.UseBestModel.SetDefault(false);
    SetDataDependentDefaults(
        trainDataMetaInfo,
        testDataMetaInfo,
        /*continueFromModel*/ false,
        /*learningContinuation*/ false,
        &outputOptions,
        &catboostOptions
    );
    NJson::TJsonValue catboostOptionsJson;
    catboostOptions.Save(&catboostOptionsJson);
    return catboostOptionsJson;
}

size_t GetNumPairs(const NCB::TDataProvider& dataProvider) noexcept {
    size_t result = 0;
    const NCB::TMaybeData<NCB::TRawPairsData>& maybePairsData = dataProvider.RawTargetData.GetPairs();
    if (maybePairsData) {
        std::visit([&](const auto& pairs) { result = pairs.size(); }, *maybePairsData);
    }
    return result;
}

TConstArrayRef<TPair> GetUngroupedPairs(const NCB::TDataProvider& dataProvider) {
    TConstArrayRef<TPair> result;
    const NCB::TMaybeData<NCB::TRawPairsData>& maybePairsData = dataProvider.RawTargetData.GetPairs();
    if (maybePairsData) {
        CB_ENSURE(
            std::holds_alternative<TFlatPairsInfo>(*maybePairsData),
            "Cannot get ungrouped pairs: pairs data is grouped"
        );
        result = std::get<TFlatPairsInfo>(*maybePairsData);
    }
    return result;
}

void TrainEvalSplit(
    const NCB::TDataProvider& srcDataProvider,
    NCB::TDataProviderPtr* trainDataProvider,
    NCB::TDataProviderPtr* evalDataProvider,
    const TTrainTestSplitParams& splitParams,
    bool saveEvalDataset,
    int threadCount,
    ui64 cpuUsedRamLimit
) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);

    bool shuffle = splitParams.Shuffle && srcDataProvider.ObjectsData->GetOrder() != NCB::EObjectsOrder::RandomShuffled;
    NCB::TObjectsGroupingSubset postShuffleGroupingSubset;
    if (shuffle) {
        TRestorableFastRng64 rand(splitParams.PartitionRandSeed);
        postShuffleGroupingSubset = NCB::Shuffle(srcDataProvider.ObjectsGrouping, 1, &rand);
    } else {
        postShuffleGroupingSubset = NCB::GetSubset(
            srcDataProvider.ObjectsGrouping,
            NCB::TArraySubsetIndexing<ui32>(NCB::TFullSubset<ui32>(srcDataProvider.ObjectsGrouping->GetGroupCount())),
            NCB::EObjectsOrder::Ordered
        );
    }
    auto postShuffleGrouping = postShuffleGroupingSubset.GetSubsetGrouping();

    // for groups
    NCB::TArraySubsetIndexing<ui32> postShuffleTrainIndices;
    NCB::TArraySubsetIndexing<ui32> postShuffleTestIndices;

    if (splitParams.Stratified) {
        auto maybeOneDimensionalTarget = srcDataProvider.RawTargetData.GetOneDimensionalTarget();
        CB_ENSURE(maybeOneDimensionalTarget, "Cannot do stratified split without one-dimensional target data");

        auto doStratifiedSplit = [&](auto targetArrayRef) {
            typedef std::remove_const_t<typename decltype(targetArrayRef)::value_type> TDst;
            TVector<TDst> shuffledTarget;
            if (shuffle) {
                shuffledTarget = NCB::GetSubset<TDst>(targetArrayRef, postShuffleGroupingSubset.GetObjectsIndexing(), &executor);
                targetArrayRef = shuffledTarget;
            }
            NCB::StratifiedTrainTestSplit(
                *postShuffleGrouping,
                targetArrayRef,
                splitParams.TrainPart,
                &postShuffleTrainIndices,
                &postShuffleTestIndices
            );
        };

        std::visit(
            TOverloaded{
                [&](const NCB::ITypedSequencePtr<float>& floatTarget) { doStratifiedSplit(TConstArrayRef<float>(NCB::ToVector(*floatTarget))); },
                [&](const TVector<TString>& stringTarget) { doStratifiedSplit(TConstArrayRef<TString>(stringTarget)); }
            },
            **maybeOneDimensionalTarget
        );
    } else {
        NCB::TrainTestSplit(*postShuffleGrouping, splitParams.TrainPart, &postShuffleTrainIndices, &postShuffleTestIndices);
    }

    auto getSubset = [&](const NCB::TArraySubsetIndexing<ui32>& postShuffleIndexing) {
        return srcDataProvider.GetSubset(
            NCB::GetSubset(
                srcDataProvider.ObjectsGrouping,
                NCB::Compose(postShuffleGroupingSubset.GetGroupsIndexing(), postShuffleIndexing),
                shuffle ? NCB::EObjectsOrder::RandomShuffled : NCB::EObjectsOrder::Ordered
            ),
            cpuUsedRamLimit,
            &executor
        );
    };

    *trainDataProvider = getSubset(postShuffleTrainIndices);
    if (saveEvalDataset) {
        *evalDataProvider = getSubset(postShuffleTestIndices);
    }
}

TAtomicSharedPtr<NPar::TTbbLocalExecutor<false>> GetCachedLocalExecutor(int threadsCount) {
    static TMutex lock;
    static TAtomicSharedPtr<NPar::TTbbLocalExecutor<false>> cachedExecutor;

    CB_ENSURE(threadsCount == -1 || 0 < threadsCount, "threadsCount should be positive or -1");

    if (threadsCount == -1) {
        threadsCount = NSystemInfo::CachedNumberOfCpus();
    }

    with_lock (lock) {
        if (cachedExecutor && cachedExecutor->GetThreadCount() + 1 == threadsCount) {
            return cachedExecutor;
        }

        cachedExecutor.Reset();
        cachedExecutor = MakeAtomicShared<NPar::TTbbLocalExecutor<false>>(threadsCount);

        return cachedExecutor;
    }
}

size_t GetMultiQuantileApproxSize(const TString& lossFunctionDescription) {
    const auto& paramsMap = ParseLossParams(lossFunctionDescription).GetParamsMap();
    return NCatboostOptions::GetAlphaMultiQuantile(paramsMap).size();
}

void GetNumFeatureValuesSample(
    const TFullModel& model,
    const NCatboostOptions::TDatasetReadingParams& datasetReadingParams,
    int threadCount,
    const TVector<ui32>& sampleIndicesVector,
    const TVector<TString>& sampleIdsVector,
    TVector<TArrayRef<float>>* numFeaturesColumns
) {
    auto localExecutor = GetCachedLocalExecutor(threadCount).Get();

    auto sampler = GetProcessor<IDataProviderSampler, TDataProviderSampleParams>(
        datasetReadingParams.PoolPath,
        TDataProviderSampleParams {
            datasetReadingParams,
            /*OnlyFeaturesData*/ true,
            /*CpuUsedRamLimit*/ Max<ui64>(),
            localExecutor
        }
    );

    TDataProviderPtr dataProvider;
    if (!sampleIndicesVector.empty()) {
        dataProvider = sampler->SampleByIndices(sampleIndicesVector);
    } else if (!sampleIdsVector.empty()) {
        dataProvider = sampler->SampleBySampleIds(sampleIdsVector);
    } else {
        CB_ENSURE(false, "Neither indices nor sampleIds are provided");
    }
    auto objectsDataProvider = dataProvider->ObjectsData;
    auto rawObjectsDataProvider = dynamic_cast<const TRawObjectsDataProvider*>(objectsDataProvider.Get());
    CB_ENSURE(rawObjectsDataProvider, "Only non-quantized datasets are supported now");

    for (const auto& floatFeature : model.ModelTrees->GetFloatFeatures()) {
        auto values = (*rawObjectsDataProvider->GetFloatFeature(floatFeature.Position.Index))->ExtractValues(localExecutor);
        auto dst = (*numFeaturesColumns)[floatFeature.Position.FlatIndex];
        Copy(values.begin(), values.end(), dst.begin());
    }
}

TMetricsAndTimeLeftHistory GetTrainingMetrics(const TFullModel& model) {
    if (model.ModelInfo.contains("training"sv)) {
        NJson::TJsonValue trainingJson;
        ReadJsonTree(model.ModelInfo.at("training"sv), &trainingJson, /*throwOnError*/ true);
        const auto& trainingMap = trainingJson.GetMap();
        if (trainingMap.contains("metrics"sv)) {
            return TMetricsAndTimeLeftHistory::LoadMetrics(trainingMap.at("metrics"sv));
        }
    }

    return TMetricsAndTimeLeftHistory();
}

namespace {
#if defined(HAVE_CUDA)

    static TString GetPyClassModuleName(PyObject* obj) {
        if (obj == nullptr) {
            return TString();
        }
        PyObject* typeObj = reinterpret_cast<PyObject*>(Py_TYPE(obj));
        PyObject* moduleObj = PyObject_GetAttrString(typeObj, "__module__"); // new ref
        if (!moduleObj) {
            PyErr_Clear();
            return TString();
        }
        const char* module = PyUnicode_AsUTF8(moduleObj);
        TString result = module ? TString(module) : TString();
        Py_DECREF(moduleObj);
        return result;
    }

    static bool HasCudaArrayInterface(PyObject* obj) {
        if (obj == nullptr) {
            return false;
        }
        PyObject* attr = PyObject_GetAttrString(obj, "__cuda_array_interface__"); // new ref
        if (!attr) {
            PyErr_Clear();
            return false;
        }
        Py_DECREF(attr);
        return true;
    }

    static bool IsCudfDataFrame(PyObject* obj) {
        const TString moduleName = GetPyClassModuleName(obj);
        if (!moduleName.StartsWith("cudf")) {
            return false;
        }
        return PyObject_HasAttrString(obj, "columns") && PyObject_HasAttrString(obj, "dtypes");
    }

    static bool IsCudfSeries(PyObject* obj) {
        const TString moduleName = GetPyClassModuleName(obj);
        if (!moduleName.StartsWith("cudf")) {
            return false;
        }
        return PyObject_HasAttrString(obj, "dtype") && PyObject_HasAttrString(obj, "to_cupy");
    }

    static bool HasDLPack(PyObject* obj) {
        if (obj == nullptr) {
            return false;
        }
        if (!PyObject_HasAttrString(obj, "__dlpack__")) {
            return false;
        }
        return PyObject_HasAttrString(obj, "__dlpack_device__");
    }

    class TPyObjectResourceHolder final : public IResourceHolder {
    public:
        explicit TPyObjectResourceHolder(PyObject* obj)
            : Obj(obj)
        {}

        ~TPyObjectResourceHolder() {
            if (Obj) {
                TGilGuard guard;
                Py_DECREF(Obj);
                Obj = nullptr;
            }
        }

    private:
        PyObject* Obj = nullptr;
    };

    class TCudaDeviceMemoryResourceHolder final : public IResourceHolder {
    public:
        TCudaDeviceMemoryResourceHolder(i32 deviceId, void* ptr)
            : DeviceId(deviceId)
            , Ptr(ptr)
        {}

        ~TCudaDeviceMemoryResourceHolder() {
            if (Ptr) {
                if (DeviceId >= 0) {
                    cudaSetDevice(DeviceId);
                }
                cudaFree(Ptr);
                Ptr = nullptr;
            }
        }

    private:
        i32 DeviceId = -1;
        void* Ptr = nullptr;
    };

    // Minimal DLPack definitions (sufficient for parsing DLManagedTensor from a Python capsule).
    enum class EDlpackDeviceType : int {
        Cpu = 1,
        Cuda = 2,
        CudaHost = 3,
        CudaManaged = 13
    };

    enum class EDlpackDataTypeCode : ui8 {
        Int = 0,
        UInt = 1,
        Float = 2,
        BFloat = 4,
        Bool = 6
    };

    struct TDlpackDevice {
        EDlpackDeviceType device_type;
        int device_id;
    };

    struct TDlpackDataType {
        ui8 code;
        ui8 bits;
        ui16 lanes;
    };

    struct TDlpackTensor {
        void* data;
        TDlpackDevice device;
        int ndim;
        TDlpackDataType dtype;
        i64* shape;
        i64* strides; // in elements, can be nullptr
        ui64 byte_offset;
    };

    struct TDlpackManagedTensor {
        TDlpackTensor dl_tensor;
        void* manager_ctx;
        void (*deleter)(TDlpackManagedTensor* self);
    };

    static EGpuInputDType GetDTypeFromDlpack(const TDlpackDataType& dtype) {
        CB_ENSURE(dtype.lanes == 1, "Only DLPack dtype.lanes==1 is supported");
        const auto code = static_cast<EDlpackDataTypeCode>(dtype.code);
        const ui8 bits = dtype.bits;
        switch (code) {
            case EDlpackDataTypeCode::Float:
                if (bits == 32) {
                    return EGpuInputDType::Float32;
                }
                if (bits == 64) {
                    return EGpuInputDType::Float64;
                }
                break;
            case EDlpackDataTypeCode::Int:
                if (bits == 8) {
                    return EGpuInputDType::Int8;
                }
                if (bits == 16) {
                    return EGpuInputDType::Int16;
                }
                if (bits == 32) {
                    return EGpuInputDType::Int32;
                }
                if (bits == 64) {
                    return EGpuInputDType::Int64;
                }
                break;
            case EDlpackDataTypeCode::UInt:
                if (bits == 8) {
                    return EGpuInputDType::UInt8;
                }
                if (bits == 16) {
                    return EGpuInputDType::UInt16;
                }
                if (bits == 32) {
                    return EGpuInputDType::UInt32;
                }
                if (bits == 64) {
                    return EGpuInputDType::UInt64;
                }
                break;
            case EDlpackDataTypeCode::Bool:
                if (bits == 1 || bits == 8) {
                    return EGpuInputDType::Bool;
                }
                break;
            case EDlpackDataTypeCode::BFloat:
                // Treat bfloat16 as unsupported for now.
                break;
        }
        ythrow TCatBoostException() << "Unsupported DLPack dtype (code=" << static_cast<int>(dtype.code)
                                    << ", bits=" << static_cast<int>(dtype.bits)
                                    << ", lanes=" << static_cast<int>(dtype.lanes) << ")";
    }

    static EGpuInputDType GetDTypeFromCudaTypestr(const TString& typestr) {
        if (typestr.size() < 3) {
            ythrow TCatBoostException() << "Invalid __cuda_array_interface__ typestr: '" << typestr << "'";
        }
        const char kind = typestr[1];
        const int itemSize = FromString<int>(typestr.substr(2));
        switch (kind) {
            case 'f':
                if (itemSize == 4) {
                    return EGpuInputDType::Float32;
                }
                if (itemSize == 8) {
                    return EGpuInputDType::Float64;
                }
                break;
            case 'i':
                if (itemSize == 1) {
                    return EGpuInputDType::Int8;
                }
                if (itemSize == 2) {
                    return EGpuInputDType::Int16;
                }
                if (itemSize == 4) {
                    return EGpuInputDType::Int32;
                }
                if (itemSize == 8) {
                    return EGpuInputDType::Int64;
                }
                break;
            case 'u':
                if (itemSize == 1) {
                    return EGpuInputDType::UInt8;
                }
                if (itemSize == 2) {
                    return EGpuInputDType::UInt16;
                }
                if (itemSize == 4) {
                    return EGpuInputDType::UInt32;
                }
                if (itemSize == 8) {
                    return EGpuInputDType::UInt64;
                }
                break;
            case 'b':
                if (itemSize == 1) {
                    return EGpuInputDType::Bool;
                }
                break;
        }
        ythrow TCatBoostException() << "Unsupported __cuda_array_interface__ typestr: '" << typestr << "'";
    }

    struct TCudaArrayInterface {
        ui64 Data = 0;
        bool ReadOnly = false;
        TVector<ui64> Shape;
        TVector<ui64> StridesBytes;
        EGpuInputDType DType = EGpuInputDType::Float32;
        ui64 Stream = 0;
        ui64 ItemSize = 0;
    };

    struct TDlpackCudaArrayInterface {
        TCudaArrayInterface Cai;
        TIntrusivePtr<IResourceHolder> Holder; // keeps capsule alive
    };

    static TDlpackCudaArrayInterface ParseDlpackToCudaArrayInterface(PyObject* obj) {
        CB_ENSURE(obj, "DLPack object is null");

        PyObject* deviceTuple = PyObject_CallMethod(obj, "__dlpack_device__", nullptr); // new ref
        if (!deviceTuple) {
            PyErr_Clear();
            ythrow TCatBoostException() << "Failed to call __dlpack_device__()";
        }
        Y_DEFER {
            Py_DECREF(deviceTuple);
        };

        CB_ENSURE(
            PyTuple_Check(deviceTuple) && PyTuple_Size(deviceTuple) == 2,
            "__dlpack_device__ must return a (device_type, device_id) tuple"
        );
        const auto deviceType = PyLong_AsLong(PyTuple_GetItem(deviceTuple, 0));
        const auto deviceId = PyLong_AsLong(PyTuple_GetItem(deviceTuple, 1));
        CB_ENSURE(!PyErr_Occurred(), "Failed to parse __dlpack_device__ result");

        const auto dlDeviceType = static_cast<EDlpackDeviceType>(deviceType);
        CB_ENSURE(
            dlDeviceType == EDlpackDeviceType::Cuda || dlDeviceType == EDlpackDeviceType::CudaManaged,
            "Only CUDA DLPack tensors are supported (device_type=" << deviceType << ")"
        );

        PyObject* capsule = PyObject_CallMethod(obj, "__dlpack__", nullptr); // new ref
        if (!capsule) {
            PyErr_Clear();
            capsule = PyObject_CallMethod(obj, "__dlpack__", "O", Py_None); // new ref
            if (!capsule) {
                PyErr_Clear();
                ythrow TCatBoostException() << "Failed to call __dlpack__()";
            }
        }

        // Keep the capsule alive for the duration of training.
        auto holder = MakeIntrusive<TPyObjectResourceHolder>(capsule);

        CB_ENSURE(PyCapsule_CheckExact(capsule), "__dlpack__() must return a PyCapsule");
        void* ptr = PyCapsule_GetPointer(capsule, "dltensor");
        if (!ptr) {
            PyErr_Clear();
            ythrow TCatBoostException() << "__dlpack__() returned a capsule that is not a 'dltensor'";
        }

        const auto* managed = reinterpret_cast<const TDlpackManagedTensor*>(ptr);
        const auto& t = managed->dl_tensor;
        CB_ENSURE(t.data != nullptr, "DLPack tensor has null data pointer");
        CB_ENSURE(t.ndim == 1 || t.ndim == 2, "Only 1D/2D DLPack tensors are supported");
        CB_ENSURE(t.shape != nullptr, "DLPack tensor has null shape");

        TCudaArrayInterface cai;
        cai.Data = static_cast<ui64>(reinterpret_cast<uintptr_t>(t.data)) + t.byte_offset;
        cai.ReadOnly = true; // DLPack does not expose writability reliably.
        cai.DType = GetDTypeFromDlpack(t.dtype);
        cai.ItemSize = (t.dtype.bits + 7) / 8;
        cai.Stream = 0;

        cai.Shape.reserve(t.ndim);
        for (int i = 0; i < t.ndim; ++i) {
            CB_ENSURE(t.shape[i] >= 0, "Invalid DLPack shape");
            cai.Shape.push_back(SafeIntegerCast<ui64>(t.shape[i]));
        }

        cai.StridesBytes.reserve(t.ndim);
        if (t.strides) {
            for (int i = 0; i < t.ndim; ++i) {
                CB_ENSURE(t.strides[i] != 0, "Invalid DLPack stride");
                cai.StridesBytes.push_back(SafeIntegerCast<ui64>(t.strides[i]) * cai.ItemSize);
            }
        } else {
            // Default: assume C-contiguous when strides are not provided.
            if (t.ndim == 1) {
                cai.StridesBytes.push_back(cai.ItemSize);
            } else {
                cai.StridesBytes.push_back(cai.ItemSize * cai.Shape[1]);
                cai.StridesBytes.push_back(cai.ItemSize);
            }
        }

        // Ensure device id can be represented as CatBoost's i32.
        const i32 dlDeviceId = SafeIntegerCast<i32>(deviceId);
        Y_UNUSED(dlDeviceId);

        return {std::move(cai), std::move(holder)};
    }

    static TCudaArrayInterface ParseCudaArrayInterface(PyObject* obj) {
        TCudaArrayInterface result;
        PyObject* caiObj = PyObject_GetAttrString(obj, "__cuda_array_interface__"); // new ref
        CB_ENSURE(caiObj, "Object has no __cuda_array_interface__");
        CB_ENSURE(PyDict_Check(caiObj), "__cuda_array_interface__ is not a dict");

        PyObject* dataTuple = PyDict_GetItemString(caiObj, "data"); // borrowed
        CB_ENSURE(dataTuple && PyTuple_Check(dataTuple) && PyTuple_Size(dataTuple) == 2, "Invalid __cuda_array_interface__.data");
        PyObject* ptrObj = PyTuple_GetItem(dataTuple, 0); // borrowed
        CB_ENSURE(ptrObj && ptrObj != Py_None, "Invalid __cuda_array_interface__.data[0]");
        result.Data = PyLong_AsUnsignedLongLong(ptrObj);
        CB_ENSURE(result.Data != 0, "Invalid __cuda_array_interface__ data pointer");
        PyObject* roObj = PyTuple_GetItem(dataTuple, 1); // borrowed
        result.ReadOnly = (roObj && PyObject_IsTrue(roObj));

        PyObject* typestrObj = PyDict_GetItemString(caiObj, "typestr"); // borrowed
        CB_ENSURE(typestrObj && PyUnicode_Check(typestrObj), "Invalid __cuda_array_interface__.typestr");
        const char* typestrC = PyUnicode_AsUTF8(typestrObj);
        CB_ENSURE(typestrC, "Invalid __cuda_array_interface__.typestr");
        const TString typestr = typestrC;
        result.DType = GetDTypeFromCudaTypestr(typestr);
        result.ItemSize = FromString<ui64>(typestr.substr(2));

        PyObject* shapeObj = PyDict_GetItemString(caiObj, "shape"); // borrowed
        CB_ENSURE(shapeObj && PyTuple_Check(shapeObj), "Invalid __cuda_array_interface__.shape");
        const Py_ssize_t ndim = PyTuple_Size(shapeObj);
        CB_ENSURE(ndim == 1 || ndim == 2, "Only 1D/2D CUDA arrays are supported");
        result.Shape.reserve(ndim);
        for (Py_ssize_t i = 0; i < ndim; ++i) {
            PyObject* dimObj = PyTuple_GetItem(shapeObj, i); // borrowed
            CB_ENSURE(dimObj, "Invalid __cuda_array_interface__.shape");
            result.Shape.push_back(PyLong_AsUnsignedLongLong(dimObj));
        }

        PyObject* stridesObj = PyDict_GetItemString(caiObj, "strides"); // borrowed, can be None
        result.StridesBytes.reserve(ndim);
        if (!stridesObj || stridesObj == Py_None) {
            // Default: C-contiguous
            if (ndim == 1) {
                result.StridesBytes.push_back(result.ItemSize);
            } else {
                result.StridesBytes.push_back(result.ItemSize * result.Shape[1]);
                result.StridesBytes.push_back(result.ItemSize);
            }
        } else {
            CB_ENSURE(PyTuple_Check(stridesObj) && PyTuple_Size(stridesObj) == ndim, "Invalid __cuda_array_interface__.strides");
            for (Py_ssize_t i = 0; i < ndim; ++i) {
                PyObject* strideObj = PyTuple_GetItem(stridesObj, i); // borrowed
                CB_ENSURE(strideObj, "Invalid __cuda_array_interface__.strides");
                result.StridesBytes.push_back(PyLong_AsUnsignedLongLong(strideObj));
            }
        }

        PyObject* streamObj = PyDict_GetItemString(caiObj, "stream"); // borrowed, optional
        if (streamObj && streamObj != Py_None) {
            result.Stream = PyLong_AsUnsignedLongLong(streamObj);
        }

        Py_DECREF(caiObj);
        return result;
    }

    static i32 GetDeviceIdFromPointer(const void* ptr) {
        cudaPointerAttributes attrs;
        const cudaError_t status = cudaPointerGetAttributes(&attrs, ptr);
        if (status != cudaSuccess) {
            cudaGetLastError(); // clear error state
            ythrow TCatBoostException() << "Pointer does not appear to be a CUDA device pointer";
        }
#if (CUDART_VERSION >= 10000)
        return attrs.device;
#else
        return attrs.device;
#endif
    }

    static void FillColumnsFromCudaMatrix(
        const TCudaArrayInterface& cai,
        TVector<TGpuInputColumnDesc>* columns
    ) {
        CB_ENSURE(cai.Shape.size() == 2, "Expected 2D CUDA array");
        const ui32 objectCount = SafeIntegerCast<ui32>(cai.Shape[0]);
        const ui32 featureCount = SafeIntegerCast<ui32>(cai.Shape[1]);

        CB_ENSURE(cai.StridesBytes.size() == 2, "Invalid strides for 2D CUDA array");
        const ui64 rowStride = cai.StridesBytes[0];
        const ui64 colStride = cai.StridesBytes[1];

        columns->clear();
        columns->reserve(featureCount);
        const i32 deviceId = GetDeviceIdFromPointer(reinterpret_cast<const void*>(cai.Data));

        for (ui32 featureIdx = 0; featureIdx < featureCount; ++featureIdx) {
            TGpuInputColumnDesc desc;
            desc.Data = cai.Data + featureIdx * colStride;
            desc.StrideBytes = rowStride;
            desc.FullObjectCount = objectCount;
            desc.DType = cai.DType;
            desc.DeviceId = deviceId;
            desc.Stream = cai.Stream;
            columns->push_back(desc);
        }
    }

    static TGpuInputColumnDesc MakeColumnFromCudaVector(
        const TCudaArrayInterface& cai
    ) {
        CB_ENSURE(cai.Shape.size() == 1, "Expected 1D CUDA array");
        CB_ENSURE(cai.StridesBytes.size() == 1, "Invalid strides for 1D CUDA array");

        TGpuInputColumnDesc desc;
        desc.Data = cai.Data;
        desc.StrideBytes = cai.StridesBytes[0];
        desc.FullObjectCount = SafeIntegerCast<ui32>(cai.Shape[0]);
        desc.DType = cai.DType;
        desc.DeviceId = GetDeviceIdFromPointer(reinterpret_cast<const void*>(cai.Data));
        desc.Stream = cai.Stream;
        return desc;
    }

    static void FillColumnsFromCudfDataFrame(
        PyObject* df,
        TVector<TGpuInputColumnDesc>* columns,
        TVector<TIntrusivePtr<IResourceHolder>>* holders
    );

    static void FillColumnsFromCudfSeries(
        PyObject* series,
        TVector<TGpuInputColumnDesc>* columns,
        TVector<TIntrusivePtr<IResourceHolder>>* holders
    );

    static void FillColumnsFromCudaArray(
        PyObject* obj,
        TVector<TGpuInputColumnDesc>* columns,
        TVector<TIntrusivePtr<IResourceHolder>>* holders
    ) {
        if (HasCudaArrayInterface(obj)) {
            const TCudaArrayInterface cai = ParseCudaArrayInterface(obj);
            if (cai.Shape.size() == 1) {
                columns->clear();
                columns->push_back(MakeColumnFromCudaVector(cai));
            } else {
                FillColumnsFromCudaMatrix(cai, columns);
            }
            return;
        }
        if (HasDLPack(obj)) {
            const auto dlpack = ParseDlpackToCudaArrayInterface(obj);
            if (dlpack.Cai.Shape.size() == 1) {
                columns->clear();
                columns->push_back(MakeColumnFromCudaVector(dlpack.Cai));
            } else {
                FillColumnsFromCudaMatrix(dlpack.Cai, columns);
            }
            if (dlpack.Holder) {
                holders->push_back(dlpack.Holder);
            }
            return;
        }
        if (IsCudfDataFrame(obj)) {
            FillColumnsFromCudfDataFrame(obj, columns, holders);
            return;
        }
        if (IsCudfSeries(obj)) {
            FillColumnsFromCudfSeries(obj, columns, holders);
            return;
        }
        ythrow TCatBoostException() << "Unsupported GPU input type (expected CuPy ndarray, DLPack tensor, or cuDF DataFrame/Series)";
    }

    static bool IsCudfCategoricalSeries(PyObject* series) {
        PyObject* dtypeObj = PyObject_GetAttrString(series, "dtype"); // new ref
        if (!dtypeObj) {
            PyErr_Clear();
            return false;
        }
        PyObject* dtypeStrObj = PyObject_Str(dtypeObj); // new ref
        Py_DECREF(dtypeObj);
        if (!dtypeStrObj) {
            PyErr_Clear();
            return false;
        }
        const char* dtypeStr = PyUnicode_AsUTF8(dtypeStrObj);
        const bool isCategorical = dtypeStr && (strcmp(dtypeStr, "category") == 0);
        Py_DECREF(dtypeStrObj);
        return isCategorical;
    }

    static ui32 CalcNullCatHash() {
        return CalcCatFeatureHash("nan");
    }

    static TGpuInputColumnDesc MakeColumnFromCudfCategoricalSeries(
        PyObject* series,
        TVector<TIntrusivePtr<IResourceHolder>>* holders
    ) {
        PyObject* catObj = PyObject_GetAttrString(series, "cat"); // new ref
        CB_ENSURE(catObj, "Failed to get cudf.Series.cat");
        Y_DEFER {
            Py_DECREF(catObj);
        };

        PyObject* codesObj = PyObject_GetAttrString(catObj, "codes"); // new ref
        CB_ENSURE(codesObj, "Failed to get cudf.Series.cat.codes");
        Y_DEFER {
            Py_XDECREF(codesObj);
        };

        const TCudaArrayInterface codesCai = ParseCudaArrayInterface(codesObj);
        CB_ENSURE(codesCai.Shape.size() == 1, "cuDF categorical codes must be 1D");
        CB_ENSURE(codesCai.StridesBytes.size() == 1, "Invalid strides for cuDF categorical codes");

        // Keep codes Series alive until inference completes.
        holders->push_back(MakeIntrusive<TPyObjectResourceHolder>(codesObj));
        codesObj = nullptr;

        PyObject* categoriesObj = PyObject_GetAttrString(catObj, "categories"); // new ref
        CB_ENSURE(categoriesObj, "Failed to get cudf.Series.cat.categories");
        Y_DEFER {
            Py_DECREF(categoriesObj);
        };

        PyObject* pandasCats = PyObject_CallMethod(categoriesObj, "to_pandas", nullptr); // new ref
        CB_ENSURE(pandasCats, "Failed to convert cuDF categorical categories to pandas");
        Y_DEFER {
            Py_DECREF(pandasCats);
        };

        PyObject* catsListObj = PyObject_CallMethod(pandasCats, "tolist", nullptr); // new ref
        CB_ENSURE(catsListObj, "Failed to convert categories to list");
        Y_DEFER {
            Py_DECREF(catsListObj);
        };

        PyObject* catsSeq = PySequence_Fast(catsListObj, "Categories must be a sequence"); // new ref
        CB_ENSURE(catsSeq, "Failed to iterate categories");
        Y_DEFER {
            Py_DECREF(catsSeq);
        };

        const Py_ssize_t catCountPy = PySequence_Fast_GET_SIZE(catsSeq);
        CB_ENSURE(catCountPy >= 0, "Invalid categories size");
        const ui32 catCount = SafeIntegerCast<ui32>(catCountPy);

        TVector<ui32> hashes;
        hashes.yresize(catCount);

        PyObject** items = PySequence_Fast_ITEMS(catsSeq);
        for (ui32 i = 0; i < catCount; ++i) {
            PyObject* item = items[i]; // borrowed
            PyObject* itemStrObj = PyObject_Str(item); // new ref
            CB_ENSURE(itemStrObj, "Failed to stringify categorical value");
            Y_DEFER {
                Py_DECREF(itemStrObj);
            };
            Py_ssize_t len = 0;
            const char* s = PyUnicode_AsUTF8AndSize(itemStrObj, &len);
            CB_ENSURE(s, "Failed to convert categorical value to UTF-8");
            hashes[i] = CalcCatFeatureHash(TStringBuf(s, static_cast<size_t>(len)));
        }

        const i32 deviceId = GetDeviceIdFromPointer(reinterpret_cast<const void*>(codesCai.Data));
        CUDA_SAFE_CALL(cudaSetDevice(deviceId));

        const ui32 dictAllocCount = (catCount > 0) ? catCount : 1u;
        void* dictPtr = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&dictPtr, static_cast<size_t>(dictAllocCount) * sizeof(ui32)));
        auto dictHolder = MakeIntrusive<TCudaDeviceMemoryResourceHolder>(deviceId, dictPtr);

        if (catCount > 0) {
            const size_t bytes = static_cast<size_t>(catCount) * sizeof(ui32);
            NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(
                /*dst*/ dictPtr,
                /*src*/ hashes.data(),
                /*bytes*/ bytes,
                cudaMemcpyHostToDevice
            );
            CUDA_SAFE_CALL(cudaMemcpyAsync(dictPtr, hashes.data(), bytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
        }

        holders->push_back(dictHolder);

        TGpuInputColumnDesc desc;
        desc.Data = codesCai.Data;
        desc.StrideBytes = codesCai.StridesBytes[0];
        desc.FullObjectCount = SafeIntegerCast<ui32>(codesCai.Shape[0]);
        desc.DType = codesCai.DType;
        desc.DeviceId = deviceId;
        desc.Stream = codesCai.Stream;
        desc.CatHashDictDevicePtr = static_cast<ui64>(reinterpret_cast<uintptr_t>(dictPtr));
        desc.CatHashDictSize = catCount;
        desc.CatHashNullValue = CalcNullCatHash();
        return desc;
    }

    static void FillColumnsFromCudfDataFrame(
        PyObject* df,
        TVector<TGpuInputColumnDesc>* columns,
        TVector<TIntrusivePtr<IResourceHolder>>* holders
    ) {
        PyObject* columnsObj = PyObject_GetAttrString(df, "columns"); // new ref
        CB_ENSURE(columnsObj, "Failed to get cudf.DataFrame.columns");
        PyObject* iter = PyObject_GetIter(columnsObj); // new ref
        Py_DECREF(columnsObj);
        CB_ENSURE(iter, "Failed to iterate cudf.DataFrame.columns");

        TVector<TGpuInputColumnDesc> result;
        ui32 objectCount = 0;
        while (PyObject* colName = PyIter_Next(iter)) { // new ref
            PyObject* series = PyObject_GetItem(df, colName); // new ref
            Py_DECREF(colName);
            CB_ENSURE(series, "Failed to get cudf.DataFrame column");

            TGpuInputColumnDesc desc;
            if (IsCudfCategoricalSeries(series)) {
                desc = MakeColumnFromCudfCategoricalSeries(series, holders);
            } else {
                const TCudaArrayInterface cai = ParseCudaArrayInterface(series);
                CB_ENSURE(cai.Shape.size() == 1, "cuDF column must be 1D");
                CB_ENSURE(cai.StridesBytes.size() == 1, "Invalid strides for 1D CUDA array");
                desc.Data = cai.Data;
                desc.StrideBytes = cai.StridesBytes[0];
                desc.FullObjectCount = SafeIntegerCast<ui32>(cai.Shape[0]);
                desc.DType = cai.DType;
                desc.DeviceId = GetDeviceIdFromPointer(reinterpret_cast<const void*>(cai.Data));
                desc.Stream = cai.Stream;
            }
            Py_DECREF(series);

            const ui32 colObjectCount = desc.FullObjectCount;
            if (result.empty()) {
                objectCount = colObjectCount;
            } else {
                CB_ENSURE(colObjectCount == objectCount, "cuDF DataFrame columns have inconsistent row counts");
            }

            result.push_back(desc);
        }
        Py_DECREF(iter);

        columns->swap(result);
    }

    static void FillColumnsFromCudfSeries(
        PyObject* series,
        TVector<TGpuInputColumnDesc>* columns,
        TVector<TIntrusivePtr<IResourceHolder>>* holders
    ) {
        TVector<TGpuInputColumnDesc> result;
        result.reserve(1);
        if (IsCudfCategoricalSeries(series)) {
            result.push_back(MakeColumnFromCudfCategoricalSeries(series, holders));
        } else {
            ythrow TCatBoostException() << "Unsupported cuDF Series dtype for GPU input: expected numeric (CUDA array interface) or 'category'";
        }
        columns->swap(result);
    }

    template <class T>
    static float ReadNumericFromBuffer(const char* ptr) {
        T v;
        memcpy(&v, ptr, sizeof(T));
        return static_cast<float>(v);
    }

    static float ReadScalarFromBuffer(const Py_buffer& view, const char* basePtr) {
        CB_ENSURE(view.format && view.format[0] && !view.format[1], "Unsupported buffer format");
        const char fmt = view.format[0];
        switch (fmt) {
            case 'f':
                CB_ENSURE(view.itemsize == 4, "Unexpected itemsize for float32");
                return ReadNumericFromBuffer<float>(basePtr);
            case 'd':
                CB_ENSURE(view.itemsize == 8, "Unexpected itemsize for float64");
                return ReadNumericFromBuffer<double>(basePtr);
            case 'b':
                CB_ENSURE(view.itemsize == 1, "Unexpected itemsize for int8");
                return ReadNumericFromBuffer<i8>(basePtr);
            case 'B':
                CB_ENSURE(view.itemsize == 1, "Unexpected itemsize for uint8");
                return ReadNumericFromBuffer<ui8>(basePtr);
            case 'h':
                CB_ENSURE(view.itemsize == 2, "Unexpected itemsize for int16");
                return ReadNumericFromBuffer<i16>(basePtr);
            case 'H':
                CB_ENSURE(view.itemsize == 2, "Unexpected itemsize for uint16");
                return ReadNumericFromBuffer<ui16>(basePtr);
            case 'i':
                CB_ENSURE(view.itemsize == 4, "Unexpected itemsize for int32");
                return ReadNumericFromBuffer<i32>(basePtr);
            case 'I':
                CB_ENSURE(view.itemsize == 4, "Unexpected itemsize for uint32");
                return ReadNumericFromBuffer<ui32>(basePtr);
            case 'l':
                CB_ENSURE(view.itemsize == 8, "Unexpected itemsize for int64");
                return ReadNumericFromBuffer<i64>(basePtr);
            case 'L':
                CB_ENSURE(view.itemsize == 8, "Unexpected itemsize for uint64");
                return ReadNumericFromBuffer<ui64>(basePtr);
            case '?':
                CB_ENSURE(view.itemsize == 1, "Unexpected itemsize for bool");
                return ReadNumericFromBuffer<bool>(basePtr) ? 1.0f : 0.0f;
            default:
                ythrow TCatBoostException() << "Unsupported buffer format '" << fmt << "'";
        }
    }

    static TVector<TVector<float>> CopyTargetsFromCpu(
        PyObject* label,
        ui32 objectCount,
        ui32 targetCount
    ) {
        if (!label || label == Py_None || targetCount == 0) {
            return {};
        }

        Py_buffer view;
        CB_ENSURE(
            PyObject_GetBuffer(label, &view, PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) == 0,
            "Failed to get buffer for label"
        );
        Y_DEFER {
            PyBuffer_Release(&view);
        };

        CB_ENSURE(view.ndim == 1 || view.ndim == 2, "Label must be 1D or 2D");
        const ui32 rows = SafeIntegerCast<ui32>(view.shape[0]);
        CB_ENSURE(rows == objectCount, "Label row count mismatch");

        const ui32 cols = (view.ndim == 1) ? 1u : SafeIntegerCast<ui32>(view.shape[1]);
        CB_ENSURE(cols == targetCount, "Label target dimension mismatch");

        TVector<TVector<float>> result;
        result.resize(targetCount);
        for (auto t : xrange(targetCount)) {
            result[t].yresize(objectCount);
        }

        const auto* base = static_cast<const char*>(view.buf);
        const Py_ssize_t stride0 = view.strides[0];
        const Py_ssize_t stride1 = (view.ndim == 1) ? 0 : view.strides[1];

        for (ui32 i = 0; i < objectCount; ++i) {
            const char* rowPtr = base + i * stride0;
            if (view.ndim == 1) {
                result[0][i] = ReadScalarFromBuffer(view, rowPtr);
            } else {
                for (ui32 t = 0; t < targetCount; ++t) {
                    result[t][i] = ReadScalarFromBuffer(view, rowPtr + t * stride1);
                }
            }
        }
        return result;
    }

    static TVector<float> CopyWeightsFromCpu(PyObject* weight, ui32 objectCount) {
        if (!weight || weight == Py_None) {
            return {};
        }

        Py_buffer view;
        CB_ENSURE(
            PyObject_GetBuffer(weight, &view, PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) == 0,
            "Failed to get buffer for weight"
        );
        Y_DEFER {
            PyBuffer_Release(&view);
        };

        CB_ENSURE(view.ndim == 1, "Weight must be 1D");
        const ui32 size = SafeIntegerCast<ui32>(view.shape[0]);
        CB_ENSURE(size == objectCount, "Weight size mismatch");

        TVector<float> result;
        result.yresize(objectCount);
        const auto* base = static_cast<const char*>(view.buf);
        const Py_ssize_t stride0 = view.strides[0];
        for (ui32 i = 0; i < objectCount; ++i) {
            result[i] = ReadScalarFromBuffer(view, base + i * stride0);
        }
        return result;
    }

#endif // HAVE_CUDA
} // namespace

    TDataProviderPtr CreateGpuDataProvider(
        const TDataMetaInfo& metaInfo,
        PyObject* data,
        PyObject* label,
    PyObject* weight,
    int /*threadCount*/
    ) {
#if !defined(HAVE_CUDA)
    Y_UNUSED(metaInfo);
    Y_UNUSED(data);
    Y_UNUSED(label);
    Y_UNUSED(weight);
    ythrow TCatBoostException() << "GPU input support requires CatBoost built with CUDA";
    #else
        CB_ENSURE(metaInfo.FeaturesLayout, "FeaturesLayout must be set");

        TGpuInputData gpuInput;
        FillColumnsFromCudaArray(data, &gpuInput.Columns, &gpuInput.ResourceHolders);

        gpuInput.FeatureCount = gpuInput.Columns.size();

        const ui32 expectedFeatureCount = metaInfo.FeaturesLayout->GetExternalFeatureCount();
    CB_ENSURE(
        gpuInput.FeatureCount == expectedFeatureCount,
        "Feature count mismatch: data has " << gpuInput.FeatureCount << " columns, but feature layout expects "
                                            << expectedFeatureCount
    );

        const ui32 objectCount = gpuInput.FeatureCount ? gpuInput.Columns[0].FullObjectCount : 0;

        const bool labelOnGpu = (label != nullptr) && (HasCudaArrayInterface(label) || HasDLPack(label) || IsCudfDataFrame(label));
        const bool weightOnGpu = (weight != nullptr) && (HasCudaArrayInterface(weight) || HasDLPack(weight) || IsCudfDataFrame(weight));

        if (labelOnGpu) {
            FillColumnsFromCudaArray(label, &gpuInput.Targets, &gpuInput.ResourceHolders);
            gpuInput.TargetCount = gpuInput.Targets.size();
            CB_ENSURE(gpuInput.TargetCount > 0, "GPU label has no target columns");
            CB_ENSURE(gpuInput.Targets[0].FullObjectCount == objectCount, "GPU label size mismatch");
            for (const auto& t : gpuInput.Targets) {
                CB_ENSURE(t.FullObjectCount == objectCount, "GPU label columns have inconsistent row counts");
            }
        }

        if (weightOnGpu) {
            TVector<TGpuInputColumnDesc> weightColumns;
            FillColumnsFromCudaArray(weight, &weightColumns, &gpuInput.ResourceHolders);
            CB_ENSURE(weightColumns.size() == 1, "GPU weight must be 1D");
            gpuInput.HasWeights = true;
            gpuInput.Weights = weightColumns[0];
            CB_ENSURE(gpuInput.Weights.FullObjectCount == objectCount, "GPU weight size mismatch");
        }

        TObjectsGroupingPtr objectsGrouping = MakeIntrusive<TObjectsGrouping>(objectCount);

        TCommonObjectsData commonData;
        commonData.PrepareForInitialization(metaInfo, objectCount, /*prevTailCount*/ 0);
    commonData.Order = EObjectsOrder::Ordered;

    auto objectsData = MakeIntrusive<TGpuRawObjectsDataProvider>(
        objectsGrouping,
        std::move(commonData),
        std::move(gpuInput),
            /*skipCheck*/ true
        );

        TRawTargetData rawTargetData;
        {
            TDataMetaInfo rawTargetMetaInfo(metaInfo);
            if (labelOnGpu) {
                rawTargetMetaInfo.TargetCount = 0;
                rawTargetMetaInfo.TargetType = ERawTargetType::None;
            }
            rawTargetData.PrepareForInitialization(rawTargetMetaInfo, objectCount, /*prevTailSize*/ 0);
        }
        if (!labelOnGpu && (metaInfo.TargetCount > 0)) {
            auto targets = CopyTargetsFromCpu(label, objectCount, metaInfo.TargetCount);
            rawTargetData.TargetType = metaInfo.TargetType;
            for (auto t : xrange(metaInfo.TargetCount)) {
                rawTargetData.Target[t] = MakeTypeCastArrayHolderFromVector<float, float>(targets[t]);
            }
        }
        {
            if (!weightOnGpu) {
                auto weights = CopyWeightsFromCpu(weight, objectCount);
                if (weights.empty()) {
                    rawTargetData.SetTrivialWeights(objectCount);
                } else {
                    rawTargetData.Weights = TWeights<float>(std::move(weights));
                    rawTargetData.GroupWeights = TWeights<float>(objectCount);
                }
            }
        }

    TRawTargetDataProvider rawTargetDataProvider(
        objectsGrouping,
        std::move(rawTargetData),
        /*skipCheck*/ true,
        /*forceUnitAutoPairWeights*/ false,
        /*localExecutor*/ Nothing()
    );

    TDataMetaInfo resultMetaInfo(metaInfo);
    resultMetaInfo.ObjectCount = objectCount;

    return MakeIntrusive<TDataProvider>(
        std::move(resultMetaInfo),
        std::move(objectsData),
        objectsGrouping,
        std::move(rawTargetDataProvider)
    );
#endif
}

void ResetCudaMemcpyTrackerConfig() {
#if !defined(HAVE_CUDA)
    ythrow TCatBoostException() << "Cuda memcpy tracking requires CatBoost built with CUDA";
#else
    NCudaLib::TMemcpyTracker::Instance().ResetConfig();
#endif
}

void ResetCudaMemcpyTrackerStats() {
#if !defined(HAVE_CUDA)
    ythrow TCatBoostException() << "Cuda memcpy tracking requires CatBoost built with CUDA";
#else
    NCudaLib::TMemcpyTracker::Instance().ResetStats();
#endif
}

TCudaMemcpyTrackerStats GetCudaMemcpyTrackerStats() {
#if !defined(HAVE_CUDA)
    ythrow TCatBoostException() << "Cuda memcpy tracking requires CatBoost built with CUDA";
#else
    const auto stats = NCudaLib::TMemcpyTracker::Instance().GetStats();
    return {
        stats.HostToHostBytes,
        stats.HostToDeviceBytes,
        stats.DeviceToHostBytes,
        stats.DeviceToDeviceBytes,
        stats.UnknownBytes
    };
#endif
}

void TestCudaMemcpyTrackerDeviceToHost(ui64 bytes) {
#if !defined(HAVE_CUDA)
    ythrow TCatBoostException() << "Cuda memcpy tracking requires CatBoost built with CUDA";
#else
    NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(
        /*dst*/ nullptr,
        /*src*/ nullptr,
        /*bytes*/ static_cast<size_t>(bytes),
        cudaMemcpyDeviceToHost
    );
#endif
}

#if defined(HAVE_CUDA)
namespace {
    static inline cudaStream_t GetCudaStreamFromCudaArrayInterface(ui64 stream) {
        if (stream == 0) {
            return 0;
        }
        if (stream == 1) {
            return cudaStreamPerThread;
        }
        return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
    }

    static inline ui32 AlignUpToWarp(ui32 size) {
        return ((size + 31u) / 32u) * 32u;
    }

    static inline void FixupTreeEnd(size_t treeCount_, int treeBegin, int* treeEnd) {
        const int treeCount = SafeIntegerCast<int>(treeCount_);
        if (treeBegin == 0 && *treeEnd == 0) {
            *treeEnd = treeCount;
        }
        CB_ENSURE(0 <= treeBegin && treeBegin <= treeCount, "Out of range treeBegin=" << treeBegin);
        CB_ENSURE(0 <= *treeEnd && *treeEnd <= treeCount, "Out of range treeEnd=" << *treeEnd);
        CB_ENSURE(treeBegin < *treeEnd, "Empty tree range [" << treeBegin << ", " << *treeEnd << ")");
    }

    static void ValidateGpuInputSingleDevice(const NCB::TGpuInputData& gpuInput, i32* deviceId) {
        CB_ENSURE(!gpuInput.Columns.empty(), "GPU input has no feature columns");
        *deviceId = gpuInput.Columns[0].DeviceId;
        CB_ENSURE(*deviceId >= 0, "Invalid GPU device id for input");
        for (const auto& col : gpuInput.Columns) {
            CB_ENSURE(col.DeviceId == *deviceId, "All GPU input columns must reside on the same device");
        }
    }

    struct TZeroCopyFloatMatrixView {
        TGPUDataInput::EFeatureLayout Layout = TGPUDataInput::EFeatureLayout::ColumnFirst;
        ui32 Stride = 0; // in elements
        const float* Base = nullptr;
    };

    static bool TryMakeZeroCopyFloatMatrixView(
        const NCB::TGpuInputData& gpuInput,
        ui32 objectCount,
        ui32 floatFeatureCount,
        const TVector<i32>& floatIndexToDataFlatIdx,
        TZeroCopyFloatMatrixView* view
    ) {
        if (objectCount == 0 || floatFeatureCount == 0) {
            return false;
        }
        if (gpuInput.Columns.size() < floatFeatureCount) {
            return false;
        }

        for (ui32 floatIdx = 0; floatIdx < floatFeatureCount; ++floatIdx) {
            const i32 dataFlatIdx = floatIndexToDataFlatIdx[floatIdx];
            if (dataFlatIdx < 0) {
                continue;
            }
            if (static_cast<ui32>(dataFlatIdx) != floatIdx) {
                return false;
            }
        }

        const ui64 basePtr = gpuInput.Columns[0].Data;
        const ui64 rowStrideBytes = gpuInput.Columns[0].StrideBytes;
        if (basePtr == 0 || rowStrideBytes == 0 || (rowStrideBytes % sizeof(float)) != 0) {
            return false;
        }

        bool rowFirst = true;
        for (ui32 flatIdx = 0; flatIdx < floatFeatureCount; ++flatIdx) {
            const auto& col = gpuInput.Columns[flatIdx];
            if (col.Data != basePtr + static_cast<ui64>(flatIdx) * sizeof(float)) {
                rowFirst = false;
                break;
            }
            if (col.StrideBytes != rowStrideBytes) {
                rowFirst = false;
                break;
            }
            if (col.DType != NCB::EGpuInputDType::Float32) {
                rowFirst = false;
                break;
            }
            if (col.FullObjectCount != objectCount) {
                rowFirst = false;
                break;
            }
        }
        if (rowFirst) {
            view->Layout = TGPUDataInput::EFeatureLayout::RowFirst;
            view->Stride = SafeIntegerCast<ui32>(rowStrideBytes / sizeof(float));
            view->Base = reinterpret_cast<const float*>(static_cast<uintptr_t>(basePtr));
            return view->Stride > 0;
        }

        if (floatFeatureCount < 2) {
            return false;
        }

        const ui64 colStrideBytes = gpuInput.Columns[1].Data - basePtr;
        if (colStrideBytes == 0 || (colStrideBytes % sizeof(float)) != 0) {
            return false;
        }

        bool colFirst = true;
        for (ui32 flatIdx = 0; flatIdx < floatFeatureCount; ++flatIdx) {
            const auto& col = gpuInput.Columns[flatIdx];
            if (col.StrideBytes != sizeof(float)) {
                colFirst = false;
                break;
            }
            if (col.Data != basePtr + static_cast<ui64>(flatIdx) * colStrideBytes) {
                colFirst = false;
                break;
            }
            if (col.DType != NCB::EGpuInputDType::Float32) {
                colFirst = false;
                break;
            }
            if (col.FullObjectCount != objectCount) {
                colFirst = false;
                break;
            }
        }
        if (colFirst) {
            view->Layout = TGPUDataInput::EFeatureLayout::ColumnFirst;
            view->Stride = SafeIntegerCast<ui32>(colStrideBytes / sizeof(float));
            view->Base = reinterpret_cast<const float*>(static_cast<uintptr_t>(basePtr));
            return view->Stride >= objectCount;
        }
        return false;
    }

    static TVector<i32> ParseCudaDevicesForGpuInputApply(const TString& devices, i32 inputDeviceId) {
        if (devices.empty()) {
            return {inputDeviceId};
        }
        int devCount = 0;
        CUDA_SAFE_CALL(cudaGetDeviceCount(&devCount));
        CB_ENSURE(devCount > 0, "No CUDA devices available");

        TSet<ui32> enabledDevices;
        if (devices == "-1") {
            for (int dev = 0; dev < devCount; ++dev) {
                enabledDevices.insert(static_cast<ui32>(dev));
            }
        } else {
            enabledDevices = NHelpers::ParseRangeString(devices, static_cast<ui32>(devCount));
        }
        CB_ENSURE(!enabledDevices.empty(), "No CUDA devices selected for prediction");
        CB_ENSURE(
            enabledDevices.find(static_cast<ui32>(inputDeviceId)) != enabledDevices.end(),
            "Input device must be included in `devices` for GPU-resident prediction"
        );

        TVector<i32> result;
        result.reserve(enabledDevices.size());
        for (ui32 dev : enabledDevices) {
            result.push_back(static_cast<i32>(dev));
        }
        Sort(result.begin(), result.end());
        return result;
    }

    static void EnablePeerAccessOrFail(i32 deviceA, i32 deviceB) {
        int canAB = 0;
        int canBA = 0;
        CUDA_SAFE_CALL(cudaDeviceCanAccessPeer(&canAB, deviceA, deviceB));
        CUDA_SAFE_CALL(cudaDeviceCanAccessPeer(&canBA, deviceB, deviceA));
        CB_ENSURE(
            canAB && canBA,
            "P2P access is required for multi-GPU prediction with GPU-resident inputs (devices " << deviceA << " and " << deviceB << ")"
        );

        auto enable = [] (i32 from, i32 to) {
            CUDA_SAFE_CALL(cudaSetDevice(from));
            const cudaError_t err = cudaDeviceEnablePeerAccess(to, 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                CUDA_SAFE_CALL(cudaGetLastError());
                return;
            }
            CUDA_SAFE_CALL(err);
        };

        enable(deviceA, deviceB);
        enable(deviceB, deviceA);
    }

    static TVector<i32> BuildFloatIndexToDataFlatIdx(
        const TFullModel& model,
        const NCB::TObjectsDataProvider& objectsData,
        const NCB::TGpuInputData& gpuInput,
        ui32* floatFeatureCount
    ) {
        THashMap<ui32, ui32> columnReorderMap;
        CheckModelAndDatasetCompatibility(model, objectsData, &columnReorderMap);

        const auto applyData = model.ModelTrees->GetApplyData();
        *floatFeatureCount = SafeIntegerCast<ui32>(applyData->MinimalSufficientFloatFeaturesVectorSize);

        TVector<i32> floatIndexToDataFlatIdx(*floatFeatureCount, -1);
        for (const auto& floatFeature : model.ModelTrees->GetFloatFeatures()) {
            if (!floatFeature.UsedInModel()) {
                continue;
            }
            const ui32 modelFlatIdx = floatFeature.Position.FlatIndex;
            const auto it = columnReorderMap.find(modelFlatIdx);
            CB_ENSURE(it != columnReorderMap.end(), "Model/dataset compatibility check failed for float feature");
            const ui32 dataFlatIdx = it->second;
            CB_ENSURE(dataFlatIdx < gpuInput.Columns.size(), "GPU input column index out of range");

            const ui32 floatIdx = floatFeature.Position.Index;
            CB_ENSURE(floatIdx < *floatFeatureCount, "Float feature index out of range");
            floatIndexToDataFlatIdx[floatIdx] = SafeIntegerCast<i32>(dataFlatIdx);
        }

        return floatIndexToDataFlatIdx;
    }

    static inline bool IsGpuInputIntegerDType(NCB::EGpuInputDType dtype) {
        switch (dtype) {
            case NCB::EGpuInputDType::Int8:
            case NCB::EGpuInputDType::Int16:
            case NCB::EGpuInputDType::Int32:
            case NCB::EGpuInputDType::Int64:
            case NCB::EGpuInputDType::UInt8:
            case NCB::EGpuInputDType::UInt16:
            case NCB::EGpuInputDType::UInt32:
            case NCB::EGpuInputDType::UInt64:
                return true;
            default:
                return false;
        }
    }

    static TVector<i32> BuildCatPackedIndexToDataFlatIdx(
        const TFullModel& model,
        const NCB::TObjectsDataProvider& objectsData,
        const NCB::TGpuInputData& gpuInput,
        ui32* catFeatureCount
    ) {
        THashMap<ui32, ui32> columnReorderMap;
        CheckModelAndDatasetCompatibility(model, objectsData, &columnReorderMap);

        const auto applyData = model.ModelTrees->GetApplyData();
        *catFeatureCount = SafeIntegerCast<ui32>(applyData->UsedCatFeaturesCount);

        TVector<i32> catPackedIndexToDataFlatIdx(*catFeatureCount, -1);
        ui32 packedIdx = 0;
        for (const auto& catFeature : model.ModelTrees->GetCatFeatures()) {
            if (!catFeature.UsedInModel()) {
                continue;
            }
            const ui32 modelFlatIdx = catFeature.Position.FlatIndex;
            const auto it = columnReorderMap.find(modelFlatIdx);
            CB_ENSURE(it != columnReorderMap.end(), "Model/dataset compatibility check failed for categorical feature");
            const ui32 dataFlatIdx = it->second;
            CB_ENSURE(dataFlatIdx < gpuInput.Columns.size(), "GPU input column index out of range");
            CB_ENSURE(packedIdx < *catFeatureCount, "Packed categorical feature index out of range");
            catPackedIndexToDataFlatIdx[packedIdx] = SafeIntegerCast<i32>(dataFlatIdx);
            ++packedIdx;
        }
        CB_ENSURE(packedIdx == *catFeatureCount, "Categorical feature count mismatch");

        return catPackedIndexToDataFlatIdx;
    }
}

TVector<TVector<double>> ApplyModelMultiGpuInput(
    const TFullModel& model,
    const NCB::TDataProvider& srcData,
    bool verbose,
    EPredictionType predictionType,
    int begin,
    int end,
    int threadCount,
    const TString& devices
) {
    Y_UNUSED(verbose);

    const auto* gpuObjects = dynamic_cast<const NCB::TGpuRawObjectsDataProvider*>(srcData.ObjectsData.Get());
    CB_ENSURE(gpuObjects, "Expected GPU-backed pool for GPU prediction");

    // Keep constraints aligned with existing GPU evaluator backend for now.
    CB_ENSURE(!model.HasTextFeatures(), "Model contains text features, GPU evaluation impossible");
    CB_ENSURE(!model.HasEmbeddingFeatures(), "Model contains embedding features, GPU evaluation impossible");
    CB_ENSURE(model.IsOblivious(), "Model is not oblivious, GPU evaluation impossible");
    CB_ENSURE(model.ModelTrees->GetApplyData()->UsedEstimatedFeaturesCount == 0, "Model contains estimated features, GPU evaluation impossible");
    const ui32 approxDimension = SafeIntegerCast<ui32>(model.GetDimensionsCount());
    CB_ENSURE(approxDimension > 0, "Model has zero dimensions, GPU evaluation impossible");

    CB_ENSURE(model.GetEvaluatorType() == EFormulaEvaluatorType::GPU, "Model evaluator type must be GPU");
    const auto& gpuInput = gpuObjects->GetData();
    i32 deviceId = -1;
    ValidateGpuInputSingleDevice(gpuInput, &deviceId);
    CUDA_SAFE_CALL(cudaSetDevice(deviceId));

    const ui32 objectCount = SafeIntegerCast<ui32>(gpuObjects->GetObjectCount());
    TVector<double> rawApprox(static_cast<size_t>(objectCount) * approxDimension);
    if (objectCount == 0) {
        return TVector<TVector<double>>(approxDimension);
    }

    const TVector<i32> devicesToUse = ParseCudaDevicesForGpuInputApply(devices, deviceId);

    if (devicesToUse.size() > 1) {
        const size_t outSize = rawApprox.size();
        TCudaVec<double> outDevice(outSize, NCuda::EMemoryType::Device);
        ApplyModelMultiGpuInputToDevice(
            model,
            srcData,
            /*verbose*/ verbose,
            predictionType,
            begin,
            end,
            static_cast<ui64>(reinterpret_cast<uintptr_t>(outDevice.Get())),
            SafeIntegerCast<ui32>(outSize),
            devices
        );
        CUDA_SAFE_CALL(cudaMemcpy(rawApprox.data(), outDevice.Get(), sizeof(double) * outSize, cudaMemcpyDeviceToHost));

        TVector<TVector<double>> approxes(approxDimension);
        for (ui32 dim = 0; dim < approxDimension; ++dim) {
            approxes[dim].yresize(objectCount);
        }
        for (ui32 doc = 0; doc < objectCount; ++doc) {
            const double* src = rawApprox.data() + static_cast<size_t>(doc) * approxDimension;
            for (ui32 dim = 0; dim < approxDimension; ++dim) {
                approxes[dim][doc] = src[dim];
            }
        }

        if (predictionType == EPredictionType::InternalRawFormulaVal) {
            return approxes;
        }

        NPar::TLocalExecutor executor;
        if (threadCount > 1) {
            executor.RunAdditionalThreads(threadCount - 1);
        }
        return PrepareEvalForInternalApprox(predictionType, model, approxes, &executor);
    }

    auto evaluator = model.GetCurrentEvaluator();
    const auto* gpuEvaluator = dynamic_cast<const NCB::NModelEvaluation::IGpuModelEvaluator*>(evaluator.Get());
    CB_ENSURE(gpuEvaluator, "Current evaluator does not support pure GPU apply");

    // Compute raw approximations on GPU and convert to requested prediction type on CPU (same as ApplyModelMulti).
    model.SetPredictionType(NCB::NModelEvaluation::EPredictionType::RawFormulaVal);

    int treeEnd = end;
    FixupTreeEnd(model.GetTreeCount(), begin, &treeEnd);

    ui32 floatFeatureCount = 0;
    const auto floatIndexToDataFlatIdx = BuildFloatIndexToDataFlatIdx(
        model,
        *gpuObjects,
        gpuInput,
        &floatFeatureCount
    );

    ui32 catFeatureCount = 0;
    const auto catPackedIndexToDataFlatIdx = BuildCatPackedIndexToDataFlatIdx(
        model,
        *gpuObjects,
        gpuInput,
        &catFeatureCount
    );

    if (devicesToUse.size() == 1 && catFeatureCount == 0) {
        TZeroCopyFloatMatrixView view;
        if (TryMakeZeroCopyFloatMatrixView(gpuInput, objectCount, floatFeatureCount, floatIndexToDataFlatIdx, &view)) {
            for (ui32 flatIdx = 0; flatIdx < floatFeatureCount; ++flatIdx) {
                const cudaStream_t inputStream = GetCudaStreamFromCudaArrayInterface(gpuInput.Columns[flatIdx].Stream);
                if (inputStream != 0) {
                    CUDA_SAFE_CALL(cudaStreamSynchronize(inputStream));
                }
            }

            TGPUDataInput dataInput;
            dataInput.FloatFeatureLayout = view.Layout;
            dataInput.ObjectCount = 0;
            dataInput.FloatFeatureCount = floatFeatureCount;
            dataInput.CatFeatureCount = 0;
            dataInput.Stride = view.Stride;
            dataInput.CatFeatureLayout = TGPUDataInput::EFeatureLayout::ColumnFirst;

            // Match apply.cpp: NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE * 64 (128 * 64 = 8192).
            const ui32 maxBlockSize = Min<ui32>(objectCount, 8192u);
            for (ui32 blockStart = 0; blockStart < objectCount; blockStart += maxBlockSize) {
                const ui32 blockSize = Min<ui32>(objectCount - blockStart, maxBlockSize);
                const float* blockBase = (view.Layout == TGPUDataInput::EFeatureLayout::RowFirst)
                    ? (view.Base + static_cast<ui64>(blockStart) * view.Stride)
                    : (view.Base + blockStart);
                const size_t blockBufSize = (view.Layout == TGPUDataInput::EFeatureLayout::RowFirst)
                    ? static_cast<size_t>(blockSize) * view.Stride
                    : static_cast<size_t>(floatFeatureCount) * view.Stride;
                dataInput.FlatFloatsVector = TConstArrayRef<float>(blockBase, blockBufSize);
                dataInput.ObjectCount = static_cast<i32>(blockSize);
                gpuEvaluator->CalcOnDevice(
                    dataInput,
                    /*treeStart*/ begin,
                    /*treeEnd*/ treeEnd,
                    MakeArrayRef(
                        rawApprox.data() + static_cast<size_t>(blockStart) * approxDimension,
                        static_cast<size_t>(blockSize) * approxDimension
                    )
                );
            }
            CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));

            TVector<TVector<double>> approxes(approxDimension);
            for (ui32 dim = 0; dim < approxDimension; ++dim) {
                approxes[dim].yresize(objectCount);
            }
            for (ui32 doc = 0; doc < objectCount; ++doc) {
                const double* src = rawApprox.data() + static_cast<size_t>(doc) * approxDimension;
                for (ui32 dim = 0; dim < approxDimension; ++dim) {
                    approxes[dim][doc] = src[dim];
                }
            }

            if (predictionType == EPredictionType::InternalRawFormulaVal) {
                return approxes;
            }

            NPar::TLocalExecutor executor;
            if (threadCount > 1) {
                executor.RunAdditionalThreads(threadCount - 1);
            }
            return PrepareEvalForInternalApprox(predictionType, model, approxes, &executor);
        }
    }

    // Match apply.cpp: NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE * 64 (128 * 64 = 8192).
    const ui32 maxBlockSize = Min<ui32>(objectCount, 8192u);
    const ui32 stride = AlignUpToWarp(maxBlockSize);
    const size_t packedSize = static_cast<size_t>(stride) * floatFeatureCount;
    TCudaVec<float> packed(packedSize, NCuda::EMemoryType::Device);
    const size_t hashedCatSize = static_cast<size_t>(stride) * catFeatureCount;
    TCudaVec<ui32> hashedCats(hashedCatSize, NCuda::EMemoryType::Device);

    TGPUDataInput dataInput;
    dataInput.FloatFeatureLayout = TGPUDataInput::EFeatureLayout::ColumnFirst;
    dataInput.ObjectCount = 0;
    dataInput.FloatFeatureCount = floatFeatureCount;
    dataInput.CatFeatureCount = catFeatureCount;
    dataInput.Stride = stride;
    dataInput.FlatFloatsVector = packed.AsArrayRef();
    dataInput.CatFeatureLayout = TGPUDataInput::EFeatureLayout::ColumnFirst;
    dataInput.HashedFlatCatFeatures = hashedCats.AsArrayRef();

    for (ui32 blockStart = 0; blockStart < objectCount; blockStart += maxBlockSize) {
        const ui32 blockSize = Min<ui32>(objectCount - blockStart, maxBlockSize);

        for (ui32 floatIdx = 0; floatIdx < floatFeatureCount; ++floatIdx) {
            const i32 dataFlatIdx = floatIndexToDataFlatIdx[floatIdx];
            if (dataFlatIdx < 0) {
                continue;
            }
            const auto& col = gpuInput.Columns[static_cast<ui32>(dataFlatIdx)];
            CB_ENSURE(col.Data != 0, "GPU input column pointer is null");
            CB_ENSURE(col.StrideBytes > 0, "GPU input column stride is invalid");

            const cudaStream_t inputStream = GetCudaStreamFromCudaArrayInterface(col.Stream);
            if (inputStream != 0) {
                CUDA_SAFE_CALL(cudaStreamSynchronize(inputStream));
            }

            const auto* srcBase = reinterpret_cast<const char*>(static_cast<uintptr_t>(col.Data));
            const void* src = srcBase + static_cast<ui64>(blockStart) * col.StrideBytes;
            float* dst = packed.Get() + static_cast<size_t>(floatIdx) * stride;
            NKernel::CopyStridedGpuInputToFloat(
                src,
                col.StrideBytes,
                blockSize,
                static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(col.DType)),
                dst,
                /*stream*/ cudaStreamPerThread
            );
        }

        for (ui32 catPackedIdx = 0; catPackedIdx < catFeatureCount; ++catPackedIdx) {
            const i32 dataFlatIdx = catPackedIndexToDataFlatIdx[catPackedIdx];
            CB_ENSURE(dataFlatIdx >= 0, "GPU input categorical column mapping is missing");

            const auto& col = gpuInput.Columns[static_cast<ui32>(dataFlatIdx)];
            CB_ENSURE(col.Data != 0, "GPU input categorical column pointer is null");
            CB_ENSURE(col.StrideBytes > 0, "GPU input categorical column stride is invalid");
            CB_ENSURE(
                IsGpuInputIntegerDType(col.DType),
                "GPU categorical features currently support only integer dtypes"
            );

            const cudaStream_t inputStream = GetCudaStreamFromCudaArrayInterface(col.Stream);
            if (inputStream != 0) {
                CUDA_SAFE_CALL(cudaStreamSynchronize(inputStream));
            }

            const auto* srcBase = reinterpret_cast<const char*>(static_cast<uintptr_t>(col.Data));
            const void* src = srcBase + static_cast<ui64>(blockStart) * col.StrideBytes;
            ui32* dst = hashedCats.Get() + static_cast<size_t>(catPackedIdx) * stride;
            if (col.CatHashDictDevicePtr != 0) {
                NKernel::MapStridedCatCodesToCatHash(
                    src,
                    col.StrideBytes,
                    blockSize,
                    static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(col.DType)),
                    reinterpret_cast<const ui32*>(static_cast<uintptr_t>(col.CatHashDictDevicePtr)),
                    col.CatHashDictSize,
                    col.CatHashNullValue,
                    dst,
                    /*stream*/ cudaStreamPerThread
                );
            } else {
                NKernel::HashStridedGpuInputToCatHash(
                    src,
                    col.StrideBytes,
                    blockSize,
                    static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(col.DType)),
                    dst,
                    /*stream*/ cudaStreamPerThread
                );
            }
        }
        CUDA_SAFE_CALL(cudaGetLastError());

        dataInput.ObjectCount = static_cast<i32>(blockSize);
        gpuEvaluator->CalcOnDevice(
            dataInput,
            /*treeStart*/ begin,
            /*treeEnd*/ treeEnd,
            MakeArrayRef(
                rawApprox.data() + static_cast<size_t>(blockStart) * approxDimension,
                static_cast<size_t>(blockSize) * approxDimension
            )
        );
    }
    CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));

    TVector<TVector<double>> approxes(approxDimension);
    for (ui32 dim = 0; dim < approxDimension; ++dim) {
        approxes[dim].yresize(objectCount);
    }
    for (ui32 doc = 0; doc < objectCount; ++doc) {
        const double* src = rawApprox.data() + static_cast<size_t>(doc) * approxDimension;
        for (ui32 dim = 0; dim < approxDimension; ++dim) {
            approxes[dim][doc] = src[dim];
        }
    }

    if (predictionType == EPredictionType::InternalRawFormulaVal) {
        return approxes;
    }

    NPar::TLocalExecutor executor;
    if (threadCount > 1) {
        executor.RunAdditionalThreads(threadCount - 1);
    }
    return PrepareEvalForInternalApprox(predictionType, model, approxes, &executor);
}

void ApplyModelMultiGpuInputToDevice(
    const TFullModel& model,
    const NCB::TDataProvider& srcData,
    bool verbose,
    EPredictionType predictionType,
    int begin,
    int end,
    ui64 dstDevicePtr,
    ui32 dstSize,
    const TString& devices
) {
    Y_UNUSED(verbose);
    Y_UNUSED(predictionType);
    Y_UNUSED(devices);

    const auto* gpuObjects = dynamic_cast<const NCB::TGpuRawObjectsDataProvider*>(srcData.ObjectsData.Get());
    CB_ENSURE(gpuObjects, "Expected GPU-backed pool for GPU prediction");

    // Keep constraints aligned with existing GPU evaluator backend for now.
    CB_ENSURE(!model.HasTextFeatures(), "Model contains text features, GPU evaluation impossible");
    CB_ENSURE(!model.HasEmbeddingFeatures(), "Model contains embedding features, GPU evaluation impossible");
    CB_ENSURE(model.IsOblivious(), "Model is not oblivious, GPU evaluation impossible");
    CB_ENSURE(model.ModelTrees->GetApplyData()->UsedEstimatedFeaturesCount == 0, "Model contains estimated features, GPU evaluation impossible");
    const ui32 approxDimension = SafeIntegerCast<ui32>(model.GetDimensionsCount());
    CB_ENSURE(approxDimension > 0, "Model has zero dimensions, GPU evaluation impossible");

    CB_ENSURE(model.GetEvaluatorType() == EFormulaEvaluatorType::GPU, "Model evaluator type must be GPU");
    const auto& gpuInput = gpuObjects->GetData();
    i32 deviceId = -1;
    ValidateGpuInputSingleDevice(gpuInput, &deviceId);

    int treeEnd = end;
    FixupTreeEnd(model.GetTreeCount(), begin, &treeEnd);

    const ui32 objectCount = SafeIntegerCast<ui32>(gpuObjects->GetObjectCount());
    CB_ENSURE(dstDevicePtr != 0 || objectCount == 0, "Destination device pointer is null");
    CB_ENSURE(dstSize == static_cast<ui32>(static_cast<size_t>(objectCount) * approxDimension), "Destination buffer size mismatch");
    if (objectCount == 0) {
        return;
    }

    CUDA_SAFE_CALL(cudaSetDevice(deviceId));

    const TVector<i32> devicesToUse = ParseCudaDevicesForGpuInputApply(devices, deviceId);

    cudaPointerAttributes dstAttrs;
    CUDA_SAFE_CALL(cudaPointerGetAttributes(&dstAttrs, reinterpret_cast<void*>(static_cast<uintptr_t>(dstDevicePtr))));
#if (CUDART_VERSION >= 10000)
    CB_ENSURE(dstAttrs.type == cudaMemoryTypeDevice, "Destination pointer is not a device pointer");
    CB_ENSURE(dstAttrs.device == deviceId, "Destination pointer device does not match input device");
#else
    CB_ENSURE(dstAttrs.memoryType == cudaMemoryTypeDevice, "Destination pointer is not a device pointer");
    CB_ENSURE(dstAttrs.device == deviceId, "Destination pointer device does not match input device");
#endif

    if (devicesToUse.size() > 1) {
        for (const i32 dev : devicesToUse) {
            if (dev == deviceId) {
                continue;
            }
            EnablePeerAccessOrFail(deviceId, dev);
        }
    }

    auto evaluator = model.GetCurrentEvaluator();
    const auto* gpuEvaluator = dynamic_cast<const NCB::NModelEvaluation::IGpuModelEvaluator*>(evaluator.Get());
    CB_ENSURE(gpuEvaluator, "Current evaluator does not support pure GPU apply");

    model.SetPredictionType(NCB::NModelEvaluation::EPredictionType::RawFormulaVal);

    ui32 floatFeatureCount = 0;
    const auto floatIndexToDataFlatIdx = BuildFloatIndexToDataFlatIdx(
        model,
        *gpuObjects,
        gpuInput,
        &floatFeatureCount
    );

    ui32 catFeatureCount = 0;
    const auto catPackedIndexToDataFlatIdx = BuildCatPackedIndexToDataFlatIdx(
        model,
        *gpuObjects,
        gpuInput,
        &catFeatureCount
    );

    if (devicesToUse.size() > 1) {
        const ui32 maxBlockSize = Min<ui32>(objectCount, 8192u);
        const ui32 stride = AlignUpToWarp(maxBlockSize);

        const size_t floatBufSize = static_cast<size_t>(stride) * floatFeatureCount;
        const size_t catBufSize = static_cast<size_t>(stride) * catFeatureCount;

        TCudaVec<float> packedPrimary(floatBufSize, NCuda::EMemoryType::Device);
        TCudaVec<ui32> hashedCatsPrimary(catBufSize, NCuda::EMemoryType::Device);

        struct TDeviceBuffers {
            NCB::NModelEvaluation::TModelEvaluatorPtr Evaluator;
            const NCB::NModelEvaluation::IGpuModelEvaluator* GpuEvaluator = nullptr;
            TCudaVec<float> Packed;
            TCudaVec<ui32> HashedCats;
            TCudaVec<double> Results;
        };

        THashMap<i32, TDeviceBuffers> buffersByDevice;

        for (const i32 dev : devicesToUse) {
            if (dev == deviceId) {
                continue;
            }
            CUDA_SAFE_CALL(cudaSetDevice(dev));
            TDeviceBuffers bufs;
            bufs.Evaluator = NCB::NModelEvaluation::CreateEvaluator(EFormulaEvaluatorType::GPU, model);
            bufs.Evaluator->SetPredictionType(NCB::NModelEvaluation::EPredictionType::RawFormulaVal);
            bufs.GpuEvaluator = dynamic_cast<const NCB::NModelEvaluation::IGpuModelEvaluator*>(bufs.Evaluator.Get());
            CB_ENSURE(bufs.GpuEvaluator, "Current evaluator does not support pure GPU apply");
            bufs.Packed = TCudaVec<float>(floatBufSize, NCuda::EMemoryType::Device);
            bufs.HashedCats = TCudaVec<ui32>(catBufSize, NCuda::EMemoryType::Device);
            bufs.Results = TCudaVec<double>(static_cast<size_t>(maxBlockSize) * approxDimension, NCuda::EMemoryType::Device);
            buffersByDevice.emplace(dev, std::move(bufs));
        }
        CUDA_SAFE_CALL(cudaSetDevice(deviceId));

        auto* dst = reinterpret_cast<double*>(static_cast<uintptr_t>(dstDevicePtr));

        const ui32 devicesCount = SafeIntegerCast<ui32>(devicesToUse.size());
        const ui32 shardSize = (objectCount + devicesCount - 1) / devicesCount;

        for (ui32 devIdx = 0; devIdx < devicesCount; ++devIdx) {
            const ui32 shardStart = devIdx * shardSize;
            if (shardStart >= objectCount) {
                break;
            }
            const ui32 shardEnd = Min<ui32>(objectCount, shardStart + shardSize);
            const ui32 shardCount = shardEnd - shardStart;
            if (shardCount == 0) {
                continue;
            }
            const i32 dev = devicesToUse[devIdx];

            for (ui32 blockStart = shardStart; blockStart < shardEnd; blockStart += maxBlockSize) {
                const ui32 blockSize = Min<ui32>(shardEnd - blockStart, maxBlockSize);

                CUDA_SAFE_CALL(cudaSetDevice(deviceId));

                for (ui32 floatIdx = 0; floatIdx < floatFeatureCount; ++floatIdx) {
                    const i32 dataFlatIdx = floatIndexToDataFlatIdx[floatIdx];
                    if (dataFlatIdx < 0) {
                        continue;
                    }
                    const auto& col = gpuInput.Columns[static_cast<ui32>(dataFlatIdx)];
                    CB_ENSURE(col.Data != 0, "GPU input column pointer is null");
                    CB_ENSURE(col.StrideBytes > 0, "GPU input column stride is invalid");

                    const cudaStream_t inputStream = GetCudaStreamFromCudaArrayInterface(col.Stream);
                    if (inputStream != 0) {
                        CUDA_SAFE_CALL(cudaStreamSynchronize(inputStream));
                    }

                    const auto* srcBase = reinterpret_cast<const char*>(static_cast<uintptr_t>(col.Data));
                    const void* src = srcBase + static_cast<ui64>(blockStart) * col.StrideBytes;
                    float* dstPacked = packedPrimary.Get() + static_cast<size_t>(floatIdx) * stride;
                    NKernel::CopyStridedGpuInputToFloat(
                        src,
                        col.StrideBytes,
                        blockSize,
                        static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(col.DType)),
                        dstPacked,
                        /*stream*/ cudaStreamPerThread
                    );
                }

                for (ui32 catPackedIdx = 0; catPackedIdx < catFeatureCount; ++catPackedIdx) {
                    const i32 dataFlatIdx = catPackedIndexToDataFlatIdx[catPackedIdx];
                    CB_ENSURE(dataFlatIdx >= 0, "GPU input categorical column mapping is missing");

                    const auto& col = gpuInput.Columns[static_cast<ui32>(dataFlatIdx)];
                    CB_ENSURE(col.Data != 0, "GPU input categorical column pointer is null");
                    CB_ENSURE(col.StrideBytes > 0, "GPU input categorical column stride is invalid");
                    CB_ENSURE(
                        IsGpuInputIntegerDType(col.DType),
                        "GPU categorical features currently support only integer dtypes"
                    );

                    const cudaStream_t inputStream = GetCudaStreamFromCudaArrayInterface(col.Stream);
                    if (inputStream != 0) {
                        CUDA_SAFE_CALL(cudaStreamSynchronize(inputStream));
                    }

                    const auto* srcBase = reinterpret_cast<const char*>(static_cast<uintptr_t>(col.Data));
                    const void* src = srcBase + static_cast<ui64>(blockStart) * col.StrideBytes;
                    ui32* dstCats = hashedCatsPrimary.Get() + static_cast<size_t>(catPackedIdx) * stride;
                    if (col.CatHashDictDevicePtr != 0) {
                        NKernel::MapStridedCatCodesToCatHash(
                            src,
                            col.StrideBytes,
                            blockSize,
                            static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(col.DType)),
                            reinterpret_cast<const ui32*>(static_cast<uintptr_t>(col.CatHashDictDevicePtr)),
                            col.CatHashDictSize,
                            col.CatHashNullValue,
                            dstCats,
                            /*stream*/ cudaStreamPerThread
                        );
                    } else {
                        NKernel::HashStridedGpuInputToCatHash(
                            src,
                            col.StrideBytes,
                            blockSize,
                            static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(col.DType)),
                            dstCats,
                            /*stream*/ cudaStreamPerThread
                        );
                    }
                }
                CUDA_SAFE_CALL(cudaGetLastError());
                CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));

                TGPUDataInput dataInput;
                dataInput.FloatFeatureLayout = TGPUDataInput::EFeatureLayout::ColumnFirst;
                dataInput.ObjectCount = static_cast<i32>(blockSize);
                dataInput.FloatFeatureCount = floatFeatureCount;
                dataInput.CatFeatureCount = catFeatureCount;
                dataInput.Stride = stride;
                dataInput.CatFeatureLayout = TGPUDataInput::EFeatureLayout::ColumnFirst;

                if (dev == deviceId) {
                    dataInput.FlatFloatsVector = packedPrimary.AsArrayRef();
                    dataInput.HashedFlatCatFeatures = hashedCatsPrimary.AsArrayRef();
                    gpuEvaluator->CalcOnDevice(
                        dataInput,
                        /*treeStart*/ begin,
                        /*treeEnd*/ treeEnd,
                        TArrayRef<double>(
                            dst + static_cast<size_t>(blockStart) * approxDimension,
                            static_cast<size_t>(blockSize) * approxDimension
                        )
                    );
                    CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
                } else {
                    auto& bufs = buffersByDevice.at(dev);
                    CUDA_SAFE_CALL(cudaMemcpyPeer(bufs.Packed.Get(), dev, packedPrimary.Get(), deviceId, sizeof(float) * floatBufSize));
                    CUDA_SAFE_CALL(cudaMemcpyPeer(bufs.HashedCats.Get(), dev, hashedCatsPrimary.Get(), deviceId, sizeof(ui32) * catBufSize));

                    CUDA_SAFE_CALL(cudaSetDevice(dev));
                    dataInput.FlatFloatsVector = bufs.Packed.AsArrayRef();
                    dataInput.HashedFlatCatFeatures = bufs.HashedCats.AsArrayRef();
                    bufs.GpuEvaluator->CalcOnDevice(
                        dataInput,
                        /*treeStart*/ begin,
                        /*treeEnd*/ treeEnd,
                        TArrayRef<double>(
                            bufs.Results.Get(),
                            static_cast<size_t>(blockSize) * approxDimension
                        )
                    );
                    CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));

                    CUDA_SAFE_CALL(cudaSetDevice(deviceId));
                    CUDA_SAFE_CALL(cudaMemcpyPeer(
                        dst + static_cast<size_t>(blockStart) * approxDimension,
                        deviceId,
                        bufs.Results.Get(),
                        dev,
                        sizeof(double) * static_cast<size_t>(blockSize) * approxDimension
                    ));
                }
            }
        }

        CUDA_SAFE_CALL(cudaSetDevice(deviceId));
        CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }

    // Match apply.cpp: NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE * 64 (128 * 64 = 8192).
    const ui32 maxBlockSize = Min<ui32>(objectCount, 8192u);
    const ui32 stride = AlignUpToWarp(maxBlockSize);
    const size_t packedSize = static_cast<size_t>(stride) * floatFeatureCount;
    TCudaVec<float> packed(packedSize, NCuda::EMemoryType::Device);
    const size_t hashedCatSize = static_cast<size_t>(stride) * catFeatureCount;
    TCudaVec<ui32> hashedCats(hashedCatSize, NCuda::EMemoryType::Device);

    TGPUDataInput dataInput;
    dataInput.FloatFeatureLayout = TGPUDataInput::EFeatureLayout::ColumnFirst;
    dataInput.ObjectCount = 0;
    dataInput.FloatFeatureCount = floatFeatureCount;
    dataInput.CatFeatureCount = catFeatureCount;
    dataInput.Stride = stride;
    dataInput.FlatFloatsVector = packed.AsArrayRef();
    dataInput.CatFeatureLayout = TGPUDataInput::EFeatureLayout::ColumnFirst;
    dataInput.HashedFlatCatFeatures = hashedCats.AsArrayRef();

    auto* dst = reinterpret_cast<double*>(static_cast<uintptr_t>(dstDevicePtr));

    if (devicesToUse.size() == 1 && catFeatureCount == 0) {
        TZeroCopyFloatMatrixView view;
        if (TryMakeZeroCopyFloatMatrixView(gpuInput, objectCount, floatFeatureCount, floatIndexToDataFlatIdx, &view)) {
            for (ui32 flatIdx = 0; flatIdx < floatFeatureCount; ++flatIdx) {
                const cudaStream_t inputStream = GetCudaStreamFromCudaArrayInterface(gpuInput.Columns[flatIdx].Stream);
                if (inputStream != 0) {
                    CUDA_SAFE_CALL(cudaStreamSynchronize(inputStream));
                }
            }

            TGPUDataInput dataInputView;
            dataInputView.FloatFeatureLayout = view.Layout;
            dataInputView.ObjectCount = 0;
            dataInputView.FloatFeatureCount = floatFeatureCount;
            dataInputView.CatFeatureCount = 0;
            dataInputView.Stride = view.Stride;
            dataInputView.CatFeatureLayout = TGPUDataInput::EFeatureLayout::ColumnFirst;

            for (ui32 blockStart = 0; blockStart < objectCount; blockStart += maxBlockSize) {
                const ui32 blockSize = Min<ui32>(objectCount - blockStart, maxBlockSize);
                const float* blockBase = (view.Layout == TGPUDataInput::EFeatureLayout::RowFirst)
                    ? (view.Base + static_cast<ui64>(blockStart) * view.Stride)
                    : (view.Base + blockStart);
                const size_t blockBufSize = (view.Layout == TGPUDataInput::EFeatureLayout::RowFirst)
                    ? static_cast<size_t>(blockSize) * view.Stride
                    : static_cast<size_t>(floatFeatureCount) * view.Stride;
                dataInputView.FlatFloatsVector = TConstArrayRef<float>(blockBase, blockBufSize);
                dataInputView.ObjectCount = static_cast<i32>(blockSize);
                gpuEvaluator->CalcOnDevice(
                    dataInputView,
                    /*treeStart*/ begin,
                    /*treeEnd*/ treeEnd,
                    TArrayRef<double>(
                        dst + static_cast<size_t>(blockStart) * approxDimension,
                        static_cast<size_t>(blockSize) * approxDimension
                    )
                );
            }
            CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
            return;
        }
    }

    for (ui32 blockStart = 0; blockStart < objectCount; blockStart += maxBlockSize) {
        const ui32 blockSize = Min<ui32>(objectCount - blockStart, maxBlockSize);

        for (ui32 floatIdx = 0; floatIdx < floatFeatureCount; ++floatIdx) {
            const i32 dataFlatIdx = floatIndexToDataFlatIdx[floatIdx];
            if (dataFlatIdx < 0) {
                continue;
            }
            const auto& col = gpuInput.Columns[static_cast<ui32>(dataFlatIdx)];
            CB_ENSURE(col.Data != 0, "GPU input column pointer is null");
            CB_ENSURE(col.StrideBytes > 0, "GPU input column stride is invalid");

            const cudaStream_t inputStream = GetCudaStreamFromCudaArrayInterface(col.Stream);
            if (inputStream != 0) {
                CUDA_SAFE_CALL(cudaStreamSynchronize(inputStream));
            }

            const auto* srcBase = reinterpret_cast<const char*>(static_cast<uintptr_t>(col.Data));
            const void* src = srcBase + static_cast<ui64>(blockStart) * col.StrideBytes;
            float* tmp = packed.Get() + static_cast<size_t>(floatIdx) * stride;
            NKernel::CopyStridedGpuInputToFloat(
                src,
                col.StrideBytes,
                blockSize,
                static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(col.DType)),
                tmp,
                /*stream*/ cudaStreamPerThread
            );
        }

        for (ui32 catPackedIdx = 0; catPackedIdx < catFeatureCount; ++catPackedIdx) {
            const i32 dataFlatIdx = catPackedIndexToDataFlatIdx[catPackedIdx];
            CB_ENSURE(dataFlatIdx >= 0, "GPU input categorical column mapping is missing");

            const auto& col = gpuInput.Columns[static_cast<ui32>(dataFlatIdx)];
            CB_ENSURE(col.Data != 0, "GPU input categorical column pointer is null");
            CB_ENSURE(col.StrideBytes > 0, "GPU input categorical column stride is invalid");
            CB_ENSURE(
                IsGpuInputIntegerDType(col.DType),
                "GPU categorical features currently support only integer dtypes"
            );

            const cudaStream_t inputStream = GetCudaStreamFromCudaArrayInterface(col.Stream);
            if (inputStream != 0) {
                CUDA_SAFE_CALL(cudaStreamSynchronize(inputStream));
            }

            const auto* srcBase = reinterpret_cast<const char*>(static_cast<uintptr_t>(col.Data));
            const void* src = srcBase + static_cast<ui64>(blockStart) * col.StrideBytes;
            ui32* dst = hashedCats.Get() + static_cast<size_t>(catPackedIdx) * stride;
            if (col.CatHashDictDevicePtr != 0) {
                NKernel::MapStridedCatCodesToCatHash(
                    src,
                    col.StrideBytes,
                    blockSize,
                    static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(col.DType)),
                    reinterpret_cast<const ui32*>(static_cast<uintptr_t>(col.CatHashDictDevicePtr)),
                    col.CatHashDictSize,
                    col.CatHashNullValue,
                    dst,
                    /*stream*/ cudaStreamPerThread
                );
            } else {
                NKernel::HashStridedGpuInputToCatHash(
                    src,
                    col.StrideBytes,
                    blockSize,
                    static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(col.DType)),
                    dst,
                    /*stream*/ cudaStreamPerThread
                );
            }
        }
        CUDA_SAFE_CALL(cudaGetLastError());

        dataInput.ObjectCount = static_cast<i32>(blockSize);
        gpuEvaluator->CalcOnDevice(
            dataInput,
            /*treeStart*/ begin,
            /*treeEnd*/ treeEnd,
            TArrayRef<double>(
                dst + static_cast<size_t>(blockStart) * approxDimension,
                static_cast<size_t>(blockSize) * approxDimension
            )
        );
    }

    CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
}

TVector<ui32> CalcLeafIndexesMultiGpuInput(
    const TFullModel& model,
    const NCB::TDataProvider& srcData,
    bool verbose,
    int begin,
    int end,
    int threadCount
) {
    Y_UNUSED(verbose);
    Y_UNUSED(threadCount);

    const auto* gpuObjects = dynamic_cast<const NCB::TGpuRawObjectsDataProvider*>(srcData.ObjectsData.Get());
    CB_ENSURE(gpuObjects, "Expected GPU-backed pool for GPU leaf index calculation");

    // Keep constraints aligned with existing GPU evaluator backend for now.
    CB_ENSURE(!model.HasTextFeatures(), "Model contains text features, GPU evaluation impossible");
    CB_ENSURE(!model.HasEmbeddingFeatures(), "Model contains embedding features, GPU evaluation impossible");
    CB_ENSURE(model.IsOblivious(), "Model is not oblivious, GPU evaluation impossible");
    CB_ENSURE(model.ModelTrees->GetApplyData()->UsedEstimatedFeaturesCount == 0, "Model contains estimated features, GPU evaluation impossible");

    CB_ENSURE(model.GetEvaluatorType() == EFormulaEvaluatorType::GPU, "Model evaluator type must be GPU");
    int treeEnd = end;
    FixupTreeEnd(model.GetTreeCount(), begin, &treeEnd);
    const ui32 treeCountToEval = SafeIntegerCast<ui32>(treeEnd - begin);

    const ui32 objectCount = SafeIntegerCast<ui32>(gpuObjects->GetObjectCount());
    TVector<ui32> leafIndexes(static_cast<size_t>(objectCount) * treeCountToEval);
    if (objectCount == 0) {
        return leafIndexes;
    }

    const auto& gpuInput = gpuObjects->GetData();
    i32 deviceId = -1;
    ValidateGpuInputSingleDevice(gpuInput, &deviceId);
    CUDA_SAFE_CALL(cudaSetDevice(deviceId));

    auto evaluator = model.GetCurrentEvaluator();
    const auto* gpuEvaluator = dynamic_cast<const NCB::NModelEvaluation::IGpuModelEvaluator*>(evaluator.Get());
    CB_ENSURE(gpuEvaluator, "Current evaluator does not support pure GPU leaf index calculation");

    ui32 floatFeatureCount = 0;
    const auto floatIndexToDataFlatIdx = BuildFloatIndexToDataFlatIdx(
        model,
        *gpuObjects,
        gpuInput,
        &floatFeatureCount
    );

    ui32 catFeatureCount = 0;
    const auto catPackedIndexToDataFlatIdx = BuildCatPackedIndexToDataFlatIdx(
        model,
        *gpuObjects,
        gpuInput,
        &catFeatureCount
    );

    if (catFeatureCount == 0) {
        TZeroCopyFloatMatrixView view;
        if (TryMakeZeroCopyFloatMatrixView(gpuInput, objectCount, floatFeatureCount, floatIndexToDataFlatIdx, &view)) {
            for (ui32 flatIdx = 0; flatIdx < floatFeatureCount; ++flatIdx) {
                const cudaStream_t inputStream = GetCudaStreamFromCudaArrayInterface(gpuInput.Columns[flatIdx].Stream);
                if (inputStream != 0) {
                    CUDA_SAFE_CALL(cudaStreamSynchronize(inputStream));
                }
            }

            TGPUDataInput dataInputView;
            dataInputView.FloatFeatureLayout = view.Layout;
            dataInputView.ObjectCount = 0;
            dataInputView.FloatFeatureCount = floatFeatureCount;
            dataInputView.CatFeatureCount = 0;
            dataInputView.Stride = view.Stride;
            dataInputView.CatFeatureLayout = TGPUDataInput::EFeatureLayout::ColumnFirst;

            // Match apply.cpp: NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE * 64 (128 * 64 = 8192).
            const ui32 maxBlockSize = Min<ui32>(objectCount, 8192u);
            for (ui32 blockStart = 0; blockStart < objectCount; blockStart += maxBlockSize) {
                const ui32 blockSize = Min<ui32>(objectCount - blockStart, maxBlockSize);
                const float* blockBase = (view.Layout == TGPUDataInput::EFeatureLayout::RowFirst)
                    ? (view.Base + static_cast<ui64>(blockStart) * view.Stride)
                    : (view.Base + blockStart);
                const size_t blockBufSize = (view.Layout == TGPUDataInput::EFeatureLayout::RowFirst)
                    ? static_cast<size_t>(blockSize) * view.Stride
                    : static_cast<size_t>(floatFeatureCount) * view.Stride;
                dataInputView.FlatFloatsVector = TConstArrayRef<float>(blockBase, blockBufSize);
                dataInputView.ObjectCount = static_cast<i32>(blockSize);
                gpuEvaluator->CalcLeafIndexesOnDevice(
                    dataInputView,
                    /*treeStart*/ begin,
                    /*treeEnd*/ treeEnd,
                    TArrayRef<ui32>(
                        leafIndexes.data() + static_cast<size_t>(blockStart) * treeCountToEval,
                        static_cast<size_t>(blockSize) * treeCountToEval
                    )
                );
            }
            CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
            return leafIndexes;
        }
    }

    // Match apply.cpp: NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE * 64 (128 * 64 = 8192).
    const ui32 maxBlockSize = Min<ui32>(objectCount, 8192u);
    const ui32 stride = AlignUpToWarp(maxBlockSize);
    const size_t packedSize = static_cast<size_t>(stride) * floatFeatureCount;
    TCudaVec<float> packed(packedSize, NCuda::EMemoryType::Device);
    const size_t hashedCatSize = static_cast<size_t>(stride) * catFeatureCount;
    TCudaVec<ui32> hashedCats(hashedCatSize, NCuda::EMemoryType::Device);

    TGPUDataInput dataInput;
    dataInput.FloatFeatureLayout = TGPUDataInput::EFeatureLayout::ColumnFirst;
    dataInput.ObjectCount = 0;
    dataInput.FloatFeatureCount = floatFeatureCount;
    dataInput.CatFeatureCount = catFeatureCount;
    dataInput.Stride = stride;
    dataInput.FlatFloatsVector = packed.AsArrayRef();
    dataInput.CatFeatureLayout = TGPUDataInput::EFeatureLayout::ColumnFirst;
    dataInput.HashedFlatCatFeatures = hashedCats.AsArrayRef();

    for (ui32 blockStart = 0; blockStart < objectCount; blockStart += maxBlockSize) {
        const ui32 blockSize = Min<ui32>(objectCount - blockStart, maxBlockSize);

        for (ui32 floatIdx = 0; floatIdx < floatFeatureCount; ++floatIdx) {
            const i32 dataFlatIdx = floatIndexToDataFlatIdx[floatIdx];
            if (dataFlatIdx < 0) {
                continue;
            }
            const auto& col = gpuInput.Columns[static_cast<ui32>(dataFlatIdx)];
            CB_ENSURE(col.Data != 0, "GPU input column pointer is null");
            CB_ENSURE(col.StrideBytes > 0, "GPU input column stride is invalid");

            const cudaStream_t inputStream = GetCudaStreamFromCudaArrayInterface(col.Stream);
            if (inputStream != 0) {
                CUDA_SAFE_CALL(cudaStreamSynchronize(inputStream));
            }

            const auto* srcBase = reinterpret_cast<const char*>(static_cast<uintptr_t>(col.Data));
            const void* src = srcBase + static_cast<ui64>(blockStart) * col.StrideBytes;
            float* dst = packed.Get() + static_cast<size_t>(floatIdx) * stride;
            NKernel::CopyStridedGpuInputToFloat(
                src,
                col.StrideBytes,
                blockSize,
                static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(col.DType)),
                dst,
                /*stream*/ cudaStreamPerThread
            );
        }

        for (ui32 catPackedIdx = 0; catPackedIdx < catFeatureCount; ++catPackedIdx) {
            const i32 dataFlatIdx = catPackedIndexToDataFlatIdx[catPackedIdx];
            CB_ENSURE(dataFlatIdx >= 0, "GPU input categorical column mapping is missing");

            const auto& col = gpuInput.Columns[static_cast<ui32>(dataFlatIdx)];
            CB_ENSURE(col.Data != 0, "GPU input categorical column pointer is null");
            CB_ENSURE(col.StrideBytes > 0, "GPU input categorical column stride is invalid");
            CB_ENSURE(
                IsGpuInputIntegerDType(col.DType),
                "GPU categorical features currently support only integer dtypes"
            );

            const cudaStream_t inputStream = GetCudaStreamFromCudaArrayInterface(col.Stream);
            if (inputStream != 0) {
                CUDA_SAFE_CALL(cudaStreamSynchronize(inputStream));
            }

            const auto* srcBase = reinterpret_cast<const char*>(static_cast<uintptr_t>(col.Data));
            const void* src = srcBase + static_cast<ui64>(blockStart) * col.StrideBytes;
            ui32* dst = hashedCats.Get() + static_cast<size_t>(catPackedIdx) * stride;
            if (col.CatHashDictDevicePtr != 0) {
                NKernel::MapStridedCatCodesToCatHash(
                    src,
                    col.StrideBytes,
                    blockSize,
                    static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(col.DType)),
                    reinterpret_cast<const ui32*>(static_cast<uintptr_t>(col.CatHashDictDevicePtr)),
                    col.CatHashDictSize,
                    col.CatHashNullValue,
                    dst,
                    /*stream*/ cudaStreamPerThread
                );
            } else {
                NKernel::HashStridedGpuInputToCatHash(
                    src,
                    col.StrideBytes,
                    blockSize,
                    static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(col.DType)),
                    dst,
                    /*stream*/ cudaStreamPerThread
                );
            }
        }
        CUDA_SAFE_CALL(cudaGetLastError());

        dataInput.ObjectCount = static_cast<i32>(blockSize);
        gpuEvaluator->CalcLeafIndexesOnDevice(
            dataInput,
            /*treeStart*/ begin,
            /*treeEnd*/ treeEnd,
            TArrayRef<ui32>(
                leafIndexes.data() + static_cast<size_t>(blockStart) * treeCountToEval,
                static_cast<size_t>(blockSize) * treeCountToEval
            )
        );
    }

    CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
    return leafIndexes;
}
#endif
