#include "doc_parallel_dataset_builder.h"
#include "compressed_index_builder.h"
#include "feature_layout_doc_parallel.h"
#include "dataset_helpers.h"
#include "estimated_features_calcer.h"

#include <catboost/cuda/data/gpu_input_provider.h>
#include <catboost/cuda/gpu_data/kernel/gpu_input_utils.cuh>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>

#include <cuda_runtime_api.h>

template <typename T>
static TVector<T> Flatten2D(TVector<TVector<T>>&& src) {
    if (src.empty()) {
        return TVector<T>();
    }
    TVector<T> result;
    result.reserve(src.size() * src[0].size());
    for (const auto& v : src) {
        result.insert(result.end(), v.begin(), v.end());
    }
    return result;
}

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

    static void FillTargetsAndWeightsFromGpuInput(
        const NCB::TGpuInputTargets& gpuTargets,
        const ui32 objectCount,
        const NCatboostCuda::TDataPermutation& loadBalancingPermutation,
        const NCudaLib::TStripeMapping& dstMapping,
        NCudaLib::TCudaBuffer<float, NCudaLib::TStripeMapping>* targets,
        NCudaLib::TCudaBuffer<float, NCudaLib::TStripeMapping>* weights
    ) {
        CB_ENSURE(targets, "targets buffer is null");
        CB_ENSURE(weights, "weights buffer is null");
        CB_ENSURE(gpuTargets.TargetCount > 0, "GPU targets are empty");
        CB_ENSURE(gpuTargets.Targets.size() == gpuTargets.TargetCount, "GPU targets size mismatch");
        CB_ENSURE(objectCount > 0, "Object count is zero");

        const i32 deviceId = gpuTargets.Targets[0].DeviceId;
        CB_ENSURE(deviceId >= 0, "Invalid device id for GPU target");
        for (const auto& t : gpuTargets.Targets) {
            CB_ENSURE(t.DeviceId == deviceId, "All GPU target columns must reside on the same device");
            CB_ENSURE(t.FullObjectCount == objectCount, "GPU target size mismatch");
        }

        NCudaLib::SetDevice(deviceId);
        const cudaStream_t caiStream = GetCudaStreamFromCudaArrayInterface(gpuTargets.Targets[0].Stream);

        TSingleBuffer<float> targetsSingle;
        targetsSingle.Reset(NCudaLib::TSingleMapping(deviceId, objectCount), gpuTargets.TargetCount);

        TSingleBuffer<float> weightsSingle;
        weightsSingle.Reset(NCudaLib::TSingleMapping(deviceId, objectCount));

        // Ensure handle-based buffers are materialized before using raw CUDA runtime calls.
        NCudaLib::GetCudaManager().WaitComplete(NCudaLib::TDevicesListBuilder::SingleDevice(deviceId));

        for (ui32 targetIdx = 0; targetIdx < gpuTargets.TargetCount; ++targetIdx) {
            const auto& col = gpuTargets.Targets[targetIdx];
            const void* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(col.Data));
            NKernel::CopyStridedGpuInputToFloat(
                src,
                col.StrideBytes,
                objectCount,
                static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(col.DType)),
                targetsSingle.At(deviceId).GetColumn(targetIdx),
                caiStream
            );
        }
        CUDA_SAFE_CALL(cudaStreamSynchronize(caiStream));

        if (gpuTargets.HasWeights) {
            CB_ENSURE(gpuTargets.Weights.DeviceId == deviceId, "GPU weights must reside on the same device as GPU targets");
            CB_ENSURE(gpuTargets.Weights.FullObjectCount == objectCount, "GPU weight size mismatch");
            const cudaStream_t weightStream = GetCudaStreamFromCudaArrayInterface(gpuTargets.Weights.Stream);
            const void* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(gpuTargets.Weights.Data));
            NKernel::CopyStridedGpuInputToFloat(
                src,
                gpuTargets.Weights.StrideBytes,
                objectCount,
                static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(gpuTargets.Weights.DType)),
                weightsSingle.At(deviceId).Get(),
                weightStream
            );
            CUDA_SAFE_CALL(cudaStreamSynchronize(weightStream));
        } else {
            FillBuffer(weightsSingle, 1.0f);
        }

        TSingleBuffer<ui32> order;
        order.Reset(NCudaLib::TSingleMapping(deviceId, objectCount));
        loadBalancingPermutation.WriteOrder(order);

        TSingleBuffer<float> targetsPermuted;
        targetsPermuted.Reset(NCudaLib::TSingleMapping(deviceId, objectCount), gpuTargets.TargetCount);
        Gather(targetsPermuted, targetsSingle, order);

        TSingleBuffer<float> weightsPermuted;
        weightsPermuted.Reset(NCudaLib::TSingleMapping(deviceId, objectCount));
        Gather(weightsPermuted, weightsSingle, order);

        targets->Reset(dstMapping, gpuTargets.TargetCount);
        weights->Reset(dstMapping);

        NCudaLib::Reshard(targetsPermuted, *targets, /*stream*/ 0);
        NCudaLib::Reshard(weightsPermuted, *weights, /*stream*/ 0);
    }
}

NCatboostCuda::TDocParallelDataSetsHolder NCatboostCuda::TDocParallelDataSetBuilder::BuildDataSet(const ui32 permutationCount,
                                                                                                  NPar::ILocalExecutor* localExecutor) {
    TDocParallelDataSetsHolder dataSetsHolder(DataProvider,
                                              FeaturesManager,
                                              LinkedTest);

    TSharedCompressedIndexBuilder<TDataSetLayout> compressedIndexBuilder(*dataSetsHolder.CompressedIndex,
                                                                         localExecutor);

    dataSetsHolder.PermutationDataSets.resize(permutationCount);

    //
    TDataPermutation learnLoadBalancingPermutation = dataSetsHolder.LearnDocPerDevicesSplit->Permutation;

    TCudaBuffer<float, NCudaLib::TStripeMapping> targets;
    TCudaBuffer<float, NCudaLib::TStripeMapping> weights;

    const auto* gpuObjects = dynamic_cast<const NCB::TGpuInputQuantizedObjectsDataProvider*>(DataProvider.ObjectsData.Get());
    const bool hasGpuTargets = gpuObjects && (gpuObjects->GetGpuTargets().TargetCount > 0);

    ui32 targetCount = 0;
    if (hasGpuTargets) {
        const auto& gpuTargets = gpuObjects->GetGpuTargets();
        targetCount = gpuTargets.TargetCount;
        FillTargetsAndWeightsFromGpuInput(
            gpuTargets,
            DataProvider.GetObjectCount(),
            learnLoadBalancingPermutation,
            dataSetsHolder.LearnDocPerDevicesSplit->Mapping,
            &targets,
            &weights
        );
    } else {
        const auto cpuTargets = *DataProvider.TargetData->GetTarget();
        targetCount = cpuTargets.size();
        targets.Reset(dataSetsHolder.LearnDocPerDevicesSplit->Mapping, targetCount);
        weights.Reset(dataSetsHolder.LearnDocPerDevicesSplit->Mapping);
        targets.Write(Flatten2D(learnLoadBalancingPermutation.Gather2D(*DataProvider.TargetData->GetTarget())));
        weights.Write(learnLoadBalancingPermutation.Gather(GetWeights(*DataProvider.TargetData)));
    }

    for (ui32 permutationId = 0; permutationId < permutationCount; ++permutationId) {
        dataSetsHolder.PermutationDataSets[permutationId] = THolder<TDocParallelDataSet>(new TDocParallelDataSet(DataProvider,
                                                                                    dataSetsHolder.CompressedIndex,
                                                                                    GetPermutation(DataProvider, permutationId),
                                                                                    learnLoadBalancingPermutation,
                                                                                    dataSetsHolder.LearnDocPerDevicesSplit->SamplesGrouping,
                                                                                    TTarget<NCudaLib::TStripeMapping>(targets.ConstCopyView(),
                                                                                                                      weights.ConstCopyView(),
                                                                                                                      /*isPairWeights*/ false)));
    }

    if (LinkedTest != nullptr) {
        TCudaBuffer<float, NCudaLib::TStripeMapping> testTargets;
        TCudaBuffer<float, NCudaLib::TStripeMapping> testWeights;

        TDataPermutation testLoadBalancingPermutation = dataSetsHolder.TestDocPerDevicesSplit->Permutation;

        testTargets.Reset(dataSetsHolder.TestDocPerDevicesSplit->Mapping, targetCount);
        testWeights.Reset(dataSetsHolder.TestDocPerDevicesSplit->Mapping);

        const auto* testGpuObjects = dynamic_cast<const NCB::TGpuInputQuantizedObjectsDataProvider*>(LinkedTest->ObjectsData.Get());
        const bool testHasGpuTargets = testGpuObjects && (testGpuObjects->GetGpuTargets().TargetCount > 0);
        if (testHasGpuTargets) {
            const auto& gpuTargets = testGpuObjects->GetGpuTargets();
            CB_ENSURE(gpuTargets.TargetCount == targetCount, "Learn/test GPU target dimension mismatch");
            FillTargetsAndWeightsFromGpuInput(
                gpuTargets,
                LinkedTest->GetObjectCount(),
                testLoadBalancingPermutation,
                dataSetsHolder.TestDocPerDevicesSplit->Mapping,
                &testTargets,
                &testWeights
            );
        } else {
            testTargets.Write(Flatten2D(testLoadBalancingPermutation.Gather2D(*LinkedTest->TargetData->GetTarget())));
            testWeights.Write(testLoadBalancingPermutation.Gather(GetWeights(*LinkedTest->TargetData)));
        }

        dataSetsHolder.TestDataSet = THolder<TDocParallelDataSet>(new TDocParallelDataSet(*LinkedTest,
                                                             dataSetsHolder.CompressedIndex,
                                                             GetIdentityPermutation(*LinkedTest),
                                                             testLoadBalancingPermutation,
                                                             dataSetsHolder.TestDocPerDevicesSplit->SamplesGrouping,
                                                             TTarget<NCudaLib::TStripeMapping>(testTargets.ConstCopyView(),
                                                                                               testWeights.ConstCopyView(),
                                                                                               /*isPairWeights*/ false)));
    }

    auto allFeatures = GetLearnFeatureIds(FeaturesManager);
    TVector<ui32> permutationIndependent;
    TVector<ui32> permutationDependent;
    {
        SplitByPermutationDependence(FeaturesManager,
                                     allFeatures,
                                     permutationCount,
                                     &permutationIndependent,
                                     &permutationDependent);
    }

    auto learnMapping = targets.GetMapping();
    TAtomicSharedPtr<TDatasetPermutationOrderAndSubsetIndexing> learnGatherIndices = nullptr;
    if (!learnLoadBalancingPermutation.IsIdentity()) {
        TVector<ui32> learnGatherIndicesVec;
        learnLoadBalancingPermutation.FillOrder(learnGatherIndicesVec);
        learnGatherIndices = TDatasetPermutationOrderAndSubsetIndexing::ConstructShared(
            DataProvider.ObjectsData->GetFeaturesArraySubsetIndexing(),
            std::move(learnGatherIndicesVec)
        );
    }


    TBinarizationInfoProvider learnBinarizationInfo(FeaturesManager,
                                                    &DataProvider);

    const ui32 permutationIndependentCompressedDataSetId = compressedIndexBuilder.AddDataSet(
        learnBinarizationInfo,
        {"Learn permutation independent features"},
        learnMapping,
        permutationIndependent,
        learnGatherIndices);

    for (ui32 permutationId = 0; permutationId < permutationCount; ++permutationId) {
        auto& dataSet = *dataSetsHolder.PermutationDataSets[permutationId];

        if (permutationDependent.size()) {
            TDataSetDescription description;
            description.Name = TStringBuilder() << "Learn permutation dependent features #" << permutationId;
            dataSet.PermutationDependentFeatures = compressedIndexBuilder.AddDataSet(learnBinarizationInfo,
                                                                                     description,
                                                                                     learnMapping,
                                                                                     permutationDependent,
                                                                                     learnGatherIndices);
        }
        dataSet.PermutationIndependentFeatures = permutationIndependentCompressedDataSetId;
    }

    ui32 testDataSetId = -1;
    if (LinkedTest) {
        TDataSetDescription description;
        description.Name = "Test dataset";

        TBinarizationInfoProvider testBinarizationInfo(FeaturesManager,
                                                       LinkedTest);
        TAtomicSharedPtr<TDatasetPermutationOrderAndSubsetIndexing> testObjectsSubsetIndexing = nullptr;
        if (!learnLoadBalancingPermutation.IsIdentity()) {
            TVector<ui32> testIndicesVec;
            dataSetsHolder.TestDocPerDevicesSplit->Permutation.FillOrder(testIndicesVec);
            testObjectsSubsetIndexing = TDatasetPermutationOrderAndSubsetIndexing::ConstructShared(
                LinkedTest->ObjectsData->GetFeaturesArraySubsetIndexing(),
                std::move(testIndicesVec)
            );
        }
        auto testMapping = dataSetsHolder.TestDataSet->GetSamplesMapping();
        testDataSetId = compressedIndexBuilder.AddDataSet(
            testBinarizationInfo,
            description,
            testMapping,
            allFeatures,
            testObjectsSubsetIndexing
        );

        dataSetsHolder.TestDataSet->PermutationIndependentFeatures = testDataSetId;
    }

    compressedIndexBuilder.PrepareToWrite();

    {
        TFloatAndOneHotFeaturesWriter<TDocParallelLayout> floatFeaturesWriter(FeaturesManager,
                                                                              compressedIndexBuilder,
                                                                              DataProvider,
                                                                              permutationIndependentCompressedDataSetId,
                                                                              /*skipExclusiveBundles=*/ false);
        floatFeaturesWriter.Write(permutationIndependent);
    }

    if (LinkedTest) {
        TFloatAndOneHotFeaturesWriter<TDocParallelLayout> floatFeaturesWriter(FeaturesManager,
                                                                              compressedIndexBuilder,
                                                                              *LinkedTest,
                                                                              testDataSetId,
                                                                              /*skipExclusiveBundles=*/ true);
        floatFeaturesWriter.Write(permutationIndependent);
    }

    if (FeaturesManager.GetCtrsCount() + FeaturesManager.GetEstimatedFeatureCount() > 0) {
        WriteCtrsAndEstimatedFeatures(
            dataSetsHolder,
            permutationIndependentCompressedDataSetId,
            testDataSetId,
            permutationCount,
            permutationIndependent,
            permutationDependent,
            &compressedIndexBuilder,
            localExecutor);
    }
    compressedIndexBuilder.Finish();

    return dataSetsHolder;
}

void NCatboostCuda::TDocParallelDataSetBuilder::WriteCtrsAndEstimatedFeatures(
    const NCatboostCuda::TDocParallelDataSetsHolder& dataSetsHolder,
    ui32 permutationIndependentCompressedDataSetId,
    ui32 testDataSetId,
    ui32 permutationCount,
    const TVector<ui32>& permutationIndependent,
    const TVector<ui32>& permutationDependent,
    NCatboostCuda::TSharedCompressedIndexBuilder<NCatboostCuda::TDocParallelLayout>* compressedIndexBuilder,
    NPar::ILocalExecutor* localExecutor
) {
    auto ctrsTarget = BuildCtrTarget(FeaturesManager,
                                        DataProvider,
                                        LinkedTest);

    TMirrorBuffer<ui32> ctrEstimationOrder;
    ctrEstimationOrder.Reset(NCudaLib::TMirrorMapping(DataProvider.GetObjectCount()));
    MakeSequence(ctrEstimationOrder);

    TMirrorBuffer<ui32> testIndices;
    if (LinkedTest) {
        testIndices.Reset(NCudaLib::TMirrorMapping(LinkedTest->GetObjectCount()));
        MakeSequence(testIndices);
    }
    //CTRs
    const auto writeCtrs = [&] (bool useTest, ui32 trainDatasetId, ui32 testDatasetId, const TVector<ui32>& featureIds) {
        TBatchedBinarizedCtrsCalcer ctrsCalcer(FeaturesManager,
                                                *ctrsTarget,
                                                DataProvider,
                                                ctrEstimationOrder,
                                                useTest ? LinkedTest : nullptr,
                                                useTest ? &testIndices : nullptr,
                                                localExecutor);

        TCtrsWriter<TDocParallelLayout> ctrsWriter(FeaturesManager,
                                                    *compressedIndexBuilder,
                                                    ctrsCalcer,
                                                    trainDatasetId,
                                                    testDatasetId);
        ctrsWriter.Write(featureIds);
    };
    writeCtrs(LinkedTest != nullptr, permutationIndependentCompressedDataSetId, testDataSetId, permutationIndependent);
    //TODO: ideally should be combined with CTRs
    const auto writeEstimatedFeatures = [&] (bool useTest, ui32 trainDatasetId, const auto& permutation, const TVector<ui32>& featureIds) {
        TEstimatorsExecutor estimatorsExecutor(FeaturesManager,
                                                Estimators,
                                                permutation,
                                                localExecutor);

        TMaybe<ui32> testId;
        if (useTest) {
            testId = testDataSetId;
        }
        TEstimatedFeaturesWriter<TDocParallelLayout> writer(FeaturesManager,
                                                            *compressedIndexBuilder,
                                                            estimatorsExecutor,
                                                            trainDatasetId,
                                                            testId);
        writer.Write(featureIds);
    };
    writeEstimatedFeatures(LinkedTest != nullptr, permutationIndependentCompressedDataSetId, GetIdentityPermutation(DataProvider), permutationIndependent);

    if (!permutationDependent.empty()) {
        for (ui32 permutationId = 0; permutationId < permutationCount; ++permutationId) {
            auto& ds = *dataSetsHolder.PermutationDataSets[permutationId];
            const TDataPermutation& ctrsEstimationPermutation = ds.GetCtrsEstimationPermutation();
            ctrsEstimationPermutation.WriteOrder(ctrEstimationOrder);

            const bool useTest = (permutationId == 0 && LinkedTest);

            writeCtrs(useTest, ds.PermutationDependentFeatures, testDataSetId, permutationDependent);
            writeEstimatedFeatures(useTest, ds.PermutationDependentFeatures, ctrsEstimationPermutation, permutationDependent);
            CATBOOST_INFO_LOG << "Ctr computation for " << permutationId << " is finished" << Endl;
        }
    }
}
