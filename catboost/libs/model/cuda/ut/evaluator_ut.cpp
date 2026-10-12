#include <catboost/libs/model/cuda/evaluator.cuh>
#include <catboost/libs/model/evaluation_interface.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/model_interface/c_api.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/yexception.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <memory>
#include <thread>

namespace {

    using EGpuPredictionType = NCB::NModelEvaluation::EPredictionType;
    constexpr size_t MAX_BATCH = 129;

    void InitCuda() {
        int count = 0;
        CUDA_SAFE_CALL(cudaGetDeviceCount(&count));
        UNIT_ASSERT(count > 0);
        CUDA_SAFE_CALL(cudaSetDevice(0));
    }

    TFullModel MakeFloatModel() {
        TObliviousTreeBuilder builder(
            {TFloatFeature(false, 0, 0, {}, ""), TFloatFeature(false, 1, 1, {}, "")},
            {}, {}, {}, 1);
        builder.AddTree(
            {TModelSplit(TFloatSplit(0, 0.0f)), TModelSplit(TFloatSplit(1, 0.5f))},
            TVector<double>{-2.0, 3.0, 1.0, 7.0}, {});
        TFullModel model;
        builder.Build(model.ModelTrees.GetMutable());
        model.UpdateDynamicData();
        model.SetScaleAndBias({2.0, {0.25}});
        return model;
    }

    void InitSingleSplitModel(TGPUCatboostEvaluationContext& context) {
        auto& model = context.GPUModelData;
        constexpr auto device = NCuda::EMemoryType::Device;
        model.TreeSplits = TCudaVec<TGPURepackedBin>({{0, 1, 0}}, device);
        model.BordersOffsets = TCudaVec<ui32>({0}, device);
        model.BordersCount = TCudaVec<ui32>({1}, device);
        model.FlatBordersVector = TCudaVec<float>({0.0f}, device);
        model.TreeSizes = TCudaVec<ui32>({1}, device);
        model.TreeStartOffsets = TCudaVec<ui32>({0}, device);
        model.TreeFirstLeafOffsets = TCudaVec<ui32>({0}, device);
        model.ModelLeafs = TCudaVec<float>({-2.0f, 3.0f}, device);
        model.FloatFeatureForBucketIdx = TCudaVec<ui32>({0}, device);
        model.UsedInModel = {true};
        model.ApproxDimension = 1;
        model.Bias = TCudaVec<double>({0.25}, device);
        model.Scale = 2.0;
        context.Stream = TCudaStream::NewStream();
    }

    TGPUDataInput MakeInput(const TCudaVec<float>& values, size_t count) {
        TGPUDataInput input;
        input.ObjectCount = static_cast<i32>(count);
        input.FloatFeatureCount = 1;
        input.Stride = 1;
        input.FloatFeatureLayout = TGPUDataInput::EFeatureLayout::RowFirst;
        input.FlatFloatsVector = values.Slice(0, count);
        return input;
    }

    void EvalSingleSplit(
        TGPUCatboostEvaluationContext& context, const TCudaVec<float>& values,
        size_t count, TCudaVec<double>& results) {
        context.EvalDataCache.PrepareCopyBufs(count, count);
        context.EvalData(MakeInput(values, count), 0, 1, results.Slice(0, count), EGpuPredictionType::RawFormulaVal);
    }

} // namespace

Y_UNIT_TEST_SUITE(TGpuWorkspace) {
    Y_UNIT_TEST(QuantizedDimensionsGrowAndRetainCapacityAcrossEmptyBatches) {
        InitCuda();
        TCudaQuantizedData data;
        data.SetDimensions(3, 0);
        UNIT_ASSERT_VALUES_EQUAL(data.GetObjectsCount(), 0u);
        UNIT_ASSERT_VALUES_EQUAL(data.BinarizedFeaturesBuffer.Get(), nullptr);

        data.SetDimensions(3, 127);
        auto* first = data.BinarizedFeaturesBuffer.Get();
        const auto firstCapacity = data.BinarizedFeaturesBuffer.Size();
        UNIT_ASSERT_UNEQUAL(first, nullptr);
        UNIT_ASSERT_VALUES_EQUAL(data.GetObjectsCount(), 127u);
        data.SetDimensions(3, 128);
        UNIT_ASSERT_VALUES_EQUAL(data.BinarizedFeaturesBuffer.Get(), first);
        UNIT_ASSERT_VALUES_EQUAL(data.BinarizedFeaturesBuffer.Size(), firstCapacity);
        UNIT_ASSERT_VALUES_EQUAL(data.GetObjectsCount(), 128u);

        data.SetDimensions(3, 129);
        auto* grown = data.BinarizedFeaturesBuffer.Get();
        const auto grownCapacity = data.BinarizedFeaturesBuffer.Size();
        UNIT_ASSERT_UNEQUAL(grown, first);
        UNIT_ASSERT(grownCapacity > firstCapacity);
        UNIT_ASSERT_VALUES_EQUAL(data.GetObjectsCount(), 129u);
        for (const ui32 count : {128u, 127u, 1u, 0u}) {
            data.SetDimensions(3, count);
            UNIT_ASSERT_VALUES_EQUAL(data.GetObjectsCount(), count);
            UNIT_ASSERT_VALUES_EQUAL(data.BinarizedFeaturesBuffer.Get(), grown);
            UNIT_ASSERT_VALUES_EQUAL(data.BinarizedFeaturesBuffer.Size(), grownCapacity);
        }
    }

    Y_UNIT_TEST(EvalDataReusesScratchAndRefreshesObjectCount) {
        InitCuda();
        TGPUCatboostEvaluationContext context;
        InitSingleSplitModel(context);
        TVector<float> hostValues(MAX_BATCH);
        for (size_t i = 0; i < hostValues.size(); ++i) {
            hostValues[i] = i % 2 == 0 ? -1.0f : 1.0f;
        }
        const TCudaVec<float> values(hostValues, NCuda::EMemoryType::Device);
        TCudaVec<double> results(MAX_BATCH + 1, NCuda::EMemoryType::Host);
        EvalSingleSplit(context, values, MAX_BATCH, results);
        auto* quantized = context.QuantizedDataCache.BinarizedFeaturesBuffer.Get();
        auto* resultScratch = context.EvalDataCache.ResultsDoubleBuf.Get();
        UNIT_ASSERT_UNEQUAL(quantized, nullptr);

        for (const size_t count : {127u, 128u, 129u, 1u, 129u}) {
            std::fill(results.Get(), results.Get() + results.Size(), -999.0);
            EvalSingleSplit(context, values, count, results);
            UNIT_ASSERT_VALUES_EQUAL(context.QuantizedDataCache.GetObjectsCount(), count);
            UNIT_ASSERT_VALUES_EQUAL(context.QuantizedDataCache.BinarizedFeaturesBuffer.Get(), quantized);
            UNIT_ASSERT_VALUES_EQUAL(context.EvalDataCache.ResultsDoubleBuf.Get(), resultScratch);
            for (size_t i = 0; i < count; ++i) {
                UNIT_ASSERT_VALUES_EQUAL(results.Get()[i], hostValues[i] > 0.0f ? 6.25 : -3.75);
            }
            UNIT_ASSERT_VALUES_EQUAL(results.Get()[count], -999.0);
        }
    }

    Y_UNIT_TEST(CopiedWarmContextSharesModelAndStreamButOwnsFreshScratch) {
        InitCuda();
        TGPUCatboostEvaluationContext original;
        InitSingleSplitModel(original);
        const TCudaVec<float> values(TVector<float>(MAX_BATCH, 1.0f), NCuda::EMemoryType::Device);
        TCudaVec<double> results(MAX_BATCH, NCuda::EMemoryType::Host);
        EvalSingleSplit(original, values, MAX_BATCH, results);

        TGPUCatboostEvaluationContext copy(original);
        UNIT_ASSERT_VALUES_EQUAL(copy.GPUModelData.ModelLeafs.Get(), original.GPUModelData.ModelLeafs.Get());
        UNIT_ASSERT_VALUES_EQUAL(copy.GPUModelData.TreeSplits.Get(), original.GPUModelData.TreeSplits.Get());
        UNIT_ASSERT(copy.Stream == original.Stream);
        UNIT_ASSERT_VALUES_EQUAL(copy.QuantizedDataCache.GetObjectsCount(), 0u);
        UNIT_ASSERT_VALUES_EQUAL(copy.QuantizedDataCache.BinarizedFeaturesBuffer.Get(), nullptr);
        UNIT_ASSERT_VALUES_EQUAL(copy.EvalDataCache.CopyDataBufHost.Get(), nullptr);
        UNIT_ASSERT_VALUES_EQUAL(copy.EvalDataCache.CopyDataBufDevice.Get(), nullptr);
        UNIT_ASSERT_VALUES_EQUAL(copy.EvalDataCache.ResultsFloatBuf.Get(), nullptr);
        UNIT_ASSERT_VALUES_EQUAL(copy.EvalDataCache.ResultsDoubleBuf.Get(), nullptr);

        std::fill(results.Get(), results.Get() + MAX_BATCH, -999.0);
        copy.EvalQuantizedData(&original.QuantizedDataCache, 0, 1, results.Slice(0, MAX_BATCH), EGpuPredictionType::RawFormulaVal);
        copy.Stream.Synchronize();
        UNIT_ASSERT_VALUES_EQUAL(copy.EvalDataCache.CopyDataBufHost.Get(), nullptr);
        UNIT_ASSERT_VALUES_EQUAL(copy.EvalDataCache.CopyDataBufDevice.Get(), nullptr);
        for (size_t i = 0; i < MAX_BATCH; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(results.Get()[i], 6.25);
        }

        EvalSingleSplit(copy, values, 127, results);
        UNIT_ASSERT_VALUES_EQUAL(copy.QuantizedDataCache.GetObjectsCount(), 127u);
        UNIT_ASSERT_VALUES_EQUAL(original.QuantizedDataCache.GetObjectsCount(), MAX_BATCH);
        UNIT_ASSERT_UNEQUAL(copy.QuantizedDataCache.BinarizedFeaturesBuffer.Get(), original.QuantizedDataCache.BinarizedFeaturesBuffer.Get());
        UNIT_ASSERT_UNEQUAL(copy.EvalDataCache.CopyDataBufHost.Get(), original.EvalDataCache.CopyDataBufHost.Get());
        UNIT_ASSERT_UNEQUAL(copy.EvalDataCache.CopyDataBufDevice.Get(), original.EvalDataCache.CopyDataBufDevice.Get());
        UNIT_ASSERT_UNEQUAL(copy.EvalDataCache.ResultsFloatBuf.Get(), original.EvalDataCache.ResultsFloatBuf.Get());
        UNIT_ASSERT_UNEQUAL(copy.EvalDataCache.ResultsDoubleBuf.Get(), original.EvalDataCache.ResultsDoubleBuf.Get());
        for (size_t i = 0; i < 127; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(results.Get()[i], 6.25);
        }
    }

    Y_UNIT_TEST(WarmEvalWaitsForPinnedHostResultsBeforeReturning) {
        InitCuda();
        TGPUCatboostEvaluationContext context;
        InitSingleSplitModel(context);
        const TCudaVec<float> values(TVector<float>(MAX_BATCH, 1.0f), NCuda::EMemoryType::Device);
        TCudaVec<double> results(MAX_BATCH, NCuda::EMemoryType::Host);
        EvalSingleSplit(context, values, MAX_BATCH, results);
        std::fill(results.Get(), results.Get() + results.Size(), -999.0);
        // Keep the warmed stream busy so accidental asynchronous return is observable.
        CUDA_SAFE_CALL(cudaLaunchHostFunc(context.Stream.GetStream(), [](void*) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }, nullptr));
        EvalSingleSplit(context, values, MAX_BATCH, results);
        UNIT_ASSERT(cudaStreamQuery(context.Stream.GetStream()) == cudaSuccess);
        for (size_t i = 0; i < MAX_BATCH; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(results.Get()[i], 6.25);
        }
    }

    Y_UNIT_TEST(CApiPreservesErrorContractAndReusesScratchAfterUnsupportedPrediction) {
        InitCuda();
        const auto cpuModel = MakeFloatModel();
        const auto serializedModel = SerializeModel(cpuModel);
        const std::unique_ptr<ModelCalcerHandle, decltype(&ModelCalcerDelete)> model(
            ModelCalcerCreate(), ModelCalcerDelete);
        UNIT_ASSERT(model);
        UNIT_ASSERT_C(LoadFullModelFromBuffer(model.get(), serializedModel.data(), serializedModel.size()), GetErrorString());
        const size_t featuresCount = GetFloatFeaturesCount(model.get());
        UNIT_ASSERT_VALUES_EQUAL(featuresCount, 2u);
        TVector<TVector<float>> rows(MAX_BATCH, TVector<float>(featuresCount));
        TVector<const float*> rowPtrs(MAX_BATCH);
        for (size_t i = 0; i < MAX_BATCH; ++i) {
            for (size_t f = 0; f < featuresCount; ++f) {
                rows[i][f] = (i + f) % 2 == 0 ? -1.0f : 1.0f;
            }
            rowPtrs[i] = rows[i].data();
        }
        UNIT_ASSERT_C(EnableGPUEvaluation(model.get(), 0), GetErrorString());
        TCudaVec<double> result(MAX_BATCH, NCuda::EMemoryType::Host);
        const auto calculate = [&] {
            return CalcModelPredictionFlat(
                model.get(), MAX_BATCH, rowPtrs.data(), featuresCount, result.Get(), MAX_BATCH);
        };
        UNIT_ASSERT_C(calculate(), GetErrorString());
        UNIT_ASSERT_C(SetPredictionType(model.get(), APT_EXPONENT), GetErrorString());
        UNIT_ASSERT(!calculate());
        UNIT_ASSERT_C(SetPredictionType(model.get(), APT_RAW_FORMULA_VAL), GetErrorString());
        for (auto& row : rows) {
            for (auto& value : row) {
                value = -value;
            }
        }
        TVector<double> expected(MAX_BATCH);
        cpuModel.CalcFlat(rows, expected);
        std::fill(result.Get(), result.Get() + MAX_BATCH, std::numeric_limits<double>::quiet_NaN());
        UNIT_ASSERT_C(calculate(), GetErrorString());
        for (size_t i = 0; i < MAX_BATCH; ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL(result.Get()[i], expected[i], 1e-6);
        }
    }

    Y_UNIT_TEST(FailedEvalWaitsForGpuWorkBeforeReturning) {
        InitCuda();
        TGPUCatboostEvaluationContext context;
        InitSingleSplitModel(context);
        const TCudaVec<float> values(TVector<float>(MAX_BATCH, 1.0f), NCuda::EMemoryType::Device);
        TCudaVec<double> results(MAX_BATCH, NCuda::EMemoryType::Host);
        EvalSingleSplit(context, values, MAX_BATCH, results);
        CUDA_SAFE_CALL(cudaLaunchHostFunc(context.Stream.GetStream(), [](void*) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }, nullptr));
        UNIT_ASSERT_EXCEPTION(
            context.EvalData(MakeInput(values, MAX_BATCH), 0, 1, results.Slice(0, MAX_BATCH), EGpuPredictionType::Exponent),
            yexception);
        // Observe completion before cleanup can hide outstanding work on the error path.
        const auto status = cudaStreamQuery(context.Stream.GetStream());
        context.Stream.Synchronize();
        UNIT_ASSERT_C(status == cudaSuccess, cudaGetErrorString(status));
    }

    Y_UNIT_TEST(FlatTransposedAndClonedGpuEvaluatorsMatchCpuAcrossBatches) {
        InitCuda();
        auto model = MakeFloatModel();
        const auto cpu = model.GetCurrentEvaluator();
        model.SetEvaluatorType(EFormulaEvaluatorType::GPU);
        const auto gpu = model.GetCurrentEvaluator();
        const size_t featuresCount = model.ModelTrees->GetFlatFeatureVectorExpectedSize();
        UNIT_ASSERT_VALUES_EQUAL(featuresCount, 2u);
        TCudaVec<double> result(MAX_BATCH, NCuda::EMemoryType::Host);
        auto clone = gpu->Clone();
        for (const size_t count : {127u, 128u, 129u, 1u, 129u, 127u}) {
            TVector<TVector<float>> rows(count, TVector<float>(featuresCount));
            TVector<TVector<float>> columns(featuresCount, TVector<float>(count));
            for (size_t i = 0; i < count; ++i) {
                for (size_t f = 0; f < featuresCount; ++f) {
                    rows[i][f] = static_cast<float>((i + f + count) % 7) - 2.0f;
                    columns[f][i] = rows[i][f];
                }
            }
            TVector<double> expected(count);
            cpu->CalcFlat(rows, expected);
            for (const auto* evaluator : std::array<const NCB::NModelEvaluation::IModelEvaluator*, 2>{gpu.Get(), clone.Get()}) {
                std::fill(result.Get(), result.Get() + count, std::numeric_limits<double>::quiet_NaN());
                evaluator->CalcFlat(rows, result.Slice(0, count));
                for (size_t i = 0; i < count; ++i) {
                    UNIT_ASSERT_DOUBLES_EQUAL(result.Get()[i], expected[i], 1e-6);
                }
                std::fill(result.Get(), result.Get() + count, std::numeric_limits<double>::quiet_NaN());
                evaluator->CalcFlatTransposed(columns, result.Slice(0, count));
                for (size_t i = 0; i < count; ++i) {
                    UNIT_ASSERT_DOUBLES_EQUAL(result.Get()[i], expected[i], 1e-6);
                }
            }
            clone = gpu->Clone();
        }
    }

} // Y_UNIT_TEST_SUITE(TGpuWorkspace)
