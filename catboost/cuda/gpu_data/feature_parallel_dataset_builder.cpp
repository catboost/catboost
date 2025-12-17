#include "feature_parallel_dataset_builder.h"
#include "compressed_index_builder.h"
#include "feature_layout_feature_parallel.h"
#include "dataset_helpers.h"

#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/cuda/data/gpu_input_provider.h>
#include <catboost/cuda/gpu_data/kernel/gpu_input_utils.cuh>
#include <catboost/cuda/gpu_data/kernel/gpu_input_targets.cuh>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>

#include <cuda_runtime_api.h>

namespace NCatboostCuda {
    TFeatureParallelDataSetsHolder TFeatureParallelDataSetHoldersBuilder::BuildDataSet(const ui32 permutationCount,
                                                                                       NPar::ILocalExecutor* localExecutor) {
        TFeatureParallelDataSetsHolder dataSetsHolder(DataProvider,
                                                      FeaturesManager);

        Y_ASSERT(dataSetsHolder.CompressedIndex);
        TSharedCompressedIndexBuilder<TDataSetLayout> compressedIndexBuilder(*dataSetsHolder.CompressedIndex,
                                                                             localExecutor);

        dataSetsHolder.CtrTargets = BuildCtrTarget(FeaturesManager,
                                                   DataProvider,
                                                   LinkedTest);
        auto& ctrsTarget = *dataSetsHolder.CtrTargets;

        {
            dataSetsHolder.LearnCatFeaturesDataSet = MakeHolder<TCompressedCatFeatureDataSet>(CatFeaturesStorage);
            BuildCompressedCatFeatures(DataProvider,
                                       *dataSetsHolder.LearnCatFeaturesDataSet,
                                       localExecutor);

            if (LinkedTest) {
                dataSetsHolder.TestCatFeaturesDataSet = MakeHolder<TCompressedCatFeatureDataSet>(CatFeaturesStorage);
                BuildCompressedCatFeatures(*LinkedTest,
                                           *dataSetsHolder.TestCatFeaturesDataSet,
                                           localExecutor);
            }
        }

        TAtomicSharedPtr<TPermutationScope> permutationIndependentScope = new TPermutationScope;

        dataSetsHolder.PermutationDataSets.resize(permutationCount);

        const auto learnWeights = NCB::GetWeights(*DataProvider.TargetData);

        bool isTrivialLearnWeights = AreEqualTo(learnWeights, 1.0f);
        {
            const auto learnMapping = NCudaLib::TMirrorMapping(ctrsTarget.LearnSlice.Size());

            const auto* gpuObjects = dynamic_cast<const NCB::TGpuInputQuantizedObjectsDataProvider*>(DataProvider.ObjectsData.Get());
            const bool hasGpuTargets = gpuObjects && (gpuObjects->GetGpuTargets().TargetCount > 0);

            if (hasGpuTargets) {
                // Weight values live on GPU; treat as non-trivial to ensure correct permutation handling.
                isTrivialLearnWeights = false;

                const auto& gpuTargets = gpuObjects->GetGpuTargets();
                CB_ENSURE(gpuTargets.TargetCount == 1, "Feature-parallel training supports only one-dimensional GPU target");
                const auto& targetColumn = gpuTargets.Targets[0];
                const i32 deviceId = targetColumn.DeviceId;
                CB_ENSURE(deviceId >= 0, "Invalid device id for GPU target");
                NCudaLib::SetDevice(deviceId);

                auto getCudaStreamFromCudaArrayInterface = [] (ui64 stream) -> cudaStream_t {
                    if (stream == 0) {
                        return 0;
                    }
                    if (stream == 1) {
                        return cudaStreamPerThread;
                    }
                    return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
                };
                const cudaStream_t caiStream = getCudaStreamFromCudaArrayInterface(targetColumn.Stream);

                const ui32 learnSize = static_cast<ui32>(ctrsTarget.LearnSlice.Size());
                CB_ENSURE(targetColumn.FullObjectCount == learnSize, "GPU target size mismatch");

                TSingleBuffer<float> targetSingle;
                targetSingle.Reset(NCudaLib::TSingleMapping(deviceId, learnSize));
                TSingleBuffer<float> weightsSingle;
                weightsSingle.Reset(NCudaLib::TSingleMapping(deviceId, learnSize));

                NCudaLib::GetCudaManager().WaitComplete(NCudaLib::TDevicesListBuilder::SingleDevice(deviceId));

                const void* srcTarget = reinterpret_cast<const void*>(static_cast<uintptr_t>(targetColumn.Data));
                NKernel::CopyStridedGpuInputToFloat(
                    srcTarget,
                    targetColumn.StrideBytes,
                    learnSize,
                    static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(targetColumn.DType)),
                    targetSingle.At(deviceId).Get(),
                    caiStream
                );

                if (gpuTargets.HasWeights) {
                    const auto& weightColumn = gpuTargets.Weights;
                    CB_ENSURE(weightColumn.DeviceId == deviceId, "GPU weights must reside on the same device as GPU target");
                    CB_ENSURE(weightColumn.FullObjectCount == learnSize, "GPU weight size mismatch");
                    const cudaStream_t weightStream = getCudaStreamFromCudaArrayInterface(weightColumn.Stream);
                    const void* srcWeight = reinterpret_cast<const void*>(static_cast<uintptr_t>(weightColumn.Data));
                    NKernel::CopyStridedGpuInputToFloat(
                        srcWeight,
                        weightColumn.StrideBytes,
                        learnSize,
                        static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(weightColumn.DType)),
                        weightsSingle.At(deviceId).Get(),
                        weightStream
                    );
                    CUDA_SAFE_CALL(cudaStreamSynchronize(weightStream));
                } else {
                    NKernel::FillLearnTestWeights(
                        weightsSingle.At(deviceId).Get(),
                        /*learnSize*/ learnSize,
                        /*totalSize*/ learnSize,
                        /*learnValue*/ 1.0f,
                        /*testValue*/ 1.0f,
                        caiStream
                    );
                }

                CUDA_SAFE_CALL(cudaStreamSynchronize(caiStream));

                dataSetsHolder.DirectTarget.Reset(learnMapping);
                dataSetsHolder.DirectWeights.Reset(learnMapping);
                NCudaLib::Reshard(targetSingle, dataSetsHolder.DirectTarget, /*stream*/ 0);
                NCudaLib::Reshard(weightsSingle, dataSetsHolder.DirectWeights, /*stream*/ 0);
            } else {
                if (isTrivialLearnWeights == ctrsTarget.IsTrivialWeights()) {
                    dataSetsHolder.DirectWeights = ctrsTarget.Weights.SliceView(ctrsTarget.LearnSlice);
                } else {
                    dataSetsHolder.DirectWeights.Reset(learnMapping);
                    dataSetsHolder.DirectWeights.Write(learnWeights);
                }
                if (isTrivialLearnWeights && ctrsTarget.IsTrivialWeights()) {
                    dataSetsHolder.DirectTarget = ctrsTarget.WeightedTarget.SliceView(ctrsTarget.LearnSlice);
                } else {
                    dataSetsHolder.DirectTarget.Reset(learnMapping);
                    dataSetsHolder.DirectTarget.Write(*DataProvider.TargetData->GetOneDimensionalTarget());
                }
            }
        }

        for (ui32 permutationId = 0; permutationId < permutationCount; ++permutationId) {
            TDataPermutation permutation = NCatboostCuda::GetPermutation(DataProvider,
                                                                         permutationId,
                                                                         DataProviderPermutationBlockSize);

            const auto targetsMapping = NCudaLib::TMirrorMapping(ctrsTarget.LearnSlice.Size());

            TMirrorBuffer<ui32> indices;
            indices.Reset(targetsMapping);
            permutation.WriteOrder(indices);
            TMirrorBuffer<ui32> inverseIndices;

            TMirrorBuffer<float> targets;
            TMirrorBuffer<float> weights;

            if (permutation.IsIdentity()) {
                inverseIndices = indices.CopyView();
                targets = dataSetsHolder.DirectTarget.CopyView();
            } else {
                inverseIndices.Reset(targetsMapping);
                permutation.WriteInversePermutation(inverseIndices);

                targets.Reset(dataSetsHolder.DirectTarget.GetMapping());
                Gather(targets, dataSetsHolder.DirectTarget, indices);
            }

            if (isTrivialLearnWeights) {
                weights = dataSetsHolder.DirectWeights.CopyView();
            } else {
                weights.Reset(dataSetsHolder.DirectTarget.GetMapping());
                Gather(weights, dataSetsHolder.DirectWeights, indices);
            }

            dataSetsHolder.PermutationDataSets[permutationId] = THolder<TFeatureParallelDataSet>(new TFeatureParallelDataSet(DataProvider,
                                                                                                                             dataSetsHolder.CompressedIndex,
                                                                                                                             permutationIndependentScope,
                                                                                                                             new TPermutationScope(),
                                                                                                                             *dataSetsHolder.LearnCatFeaturesDataSet,
                                                                                                                             dataSetsHolder.GetCtrTargets(),
                                                                                                                             TTarget<NCudaLib::TMirrorMapping>(targets.AsConstBuf(),
                                                                                                                                                               weights.AsConstBuf(),
                                                                                                                                                               indices.AsConstBuf(),
                                                                                                                                                               /*isPairWeights*/ false),
                                                                                                                             std::move(inverseIndices),
                                                                                                                             std::move(permutation)));
        }

        if (LinkedTest != nullptr) {
            BuildTestTargetAndIndices(dataSetsHolder, ctrsTarget);
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

        auto learnMapping = dataSetsHolder.PermutationDataSets[0]->GetSamplesMapping();

        TBinarizationInfoProvider learnBinarizationInfo(FeaturesManager,
                                                        &DataProvider);

        const ui32 permutationIndependentCompressedDataSetId = compressedIndexBuilder.AddDataSet(learnBinarizationInfo,
                                                                                                 {"Learn permutation-independent features"},
                                                                                                 learnMapping,
                                                                                                 permutationIndependent);

        for (ui32 permutationId = 0; permutationId < permutationCount; ++permutationId) {
            auto& dataSet = *dataSetsHolder.PermutationDataSets[permutationId];
            if (permutationDependent.size()) {
                const auto& permutation = dataSet.GetCtrsEstimationPermutation();
                TVector<ui32> gatherIndices;
                permutation.FillOrder(gatherIndices);
                auto composedSubsetIndexing = TDatasetPermutationOrderAndSubsetIndexing::ConstructShared(
                    DataProvider.ObjectsData->GetFeaturesArraySubsetIndexing(),
                    std::move(gatherIndices)
                );
                TDataSetDescription description;
                description.Name = TStringBuilder() << "Learn permutation dependent features #" << permutationId;
                dataSet.PermutationDependentFeatures = compressedIndexBuilder.AddDataSet(
                    learnBinarizationInfo,
                    description,
                    learnMapping,
                    permutationDependent,
                    composedSubsetIndexing
                );
            }
            dataSet.PermutationIndependentFeatures = permutationIndependentCompressedDataSetId;
        }

        ui32 testDataSetId = -1;
        if (LinkedTest) {
            TDataSetDescription description;
            description.Name = "Test dataset";
            TBinarizationInfoProvider testBinarizationInfo(FeaturesManager,
                                                           LinkedTest);

            auto testMapping = dataSetsHolder.TestDataSet->GetSamplesMapping();
            testDataSetId = compressedIndexBuilder.AddDataSet(testBinarizationInfo,
                                                              description,
                                                              testMapping,
                                                              allFeatures);

            dataSetsHolder.TestDataSet->PermutationIndependentFeatures = testDataSetId;
        }

        compressedIndexBuilder.PrepareToWrite();

        {
            TFloatAndOneHotFeaturesWriter<TFeatureParallelLayout> floatFeaturesWriter(FeaturesManager,
                                                                                      compressedIndexBuilder,
                                                                                      DataProvider,
                                                                                      permutationIndependentCompressedDataSetId,
                                                                                      /*skipExclusiveBundles=*/ false);
            floatFeaturesWriter.Write(permutationIndependent);
        }

        if (LinkedTest) {
            TFloatAndOneHotFeaturesWriter<TFeatureParallelLayout> floatFeaturesWriter(FeaturesManager,
                                                                                      compressedIndexBuilder,
                                                                                      *LinkedTest,
                                                                                      testDataSetId,
                                                                                      /*skipExclusiveBundles=*/ true);
            floatFeaturesWriter.Write(permutationIndependent);
        }

        {
            TBatchedBinarizedCtrsCalcer ctrsCalcer(FeaturesManager,
                                                   ctrsTarget,
                                                   DataProvider,
                                                   dataSetsHolder.PermutationDataSets[0]->GetTarget().GetIndices(),
                                                   LinkedTest,
                                                   LinkedTest ? &dataSetsHolder.TestDataSet->GetTarget().GetIndices() : nullptr,
                                                   localExecutor);

            TCtrsWriter<TFeatureParallelLayout> ctrsWriter(FeaturesManager,
                                                           compressedIndexBuilder,
                                                           ctrsCalcer,
                                                           permutationIndependentCompressedDataSetId,
                                                           testDataSetId);
            ctrsWriter.Write(permutationIndependent);
        }
        {
            TEstimatorsExecutor estimatorsExecutor(FeaturesManager,
                                                   Estimators,
                                                   dataSetsHolder.PermutationDataSets[0]->CtrsEstimationPermutation,
                                                   localExecutor
            );

            TMaybe<ui32> testId;
            if (LinkedTest) {
                testId = testDataSetId;
            }
            TEstimatedFeaturesWriter<TFeatureParallelLayout> writer(FeaturesManager,
                                                                    compressedIndexBuilder,
                                                                    estimatorsExecutor,
                                                                    permutationIndependentCompressedDataSetId,
                                                                    testId);
            writer.Write(permutationIndependent);
        }

        if (!permutationDependent.empty()) {
            for (ui32 permutationId = 0; permutationId < permutationCount; ++permutationId) {
                auto& ds = *dataSetsHolder.PermutationDataSets[permutationId];
                //link common datasets
                if (permutationId > 0) {
                    ds.PermutationIndependentFeatures = permutationIndependentCompressedDataSetId;
                }

                const NCB::TTrainingDataProvider* linkedTest = permutationId == 0 ? LinkedTest : nullptr;
                const TMirrorBuffer<const ui32>* testIndices = (permutationId == 0 && linkedTest)
                                                               ? &dataSetsHolder.TestDataSet->GetTarget().GetIndices()
                                                               : nullptr;

                {
                    TBatchedBinarizedCtrsCalcer ctrsCalcer(FeaturesManager,
                                                           ctrsTarget,
                                                           DataProvider,
                                                           ds.GetIndices(),
                                                           linkedTest,
                                                           testIndices,
                                                           localExecutor);

                    TCtrsWriter<TFeatureParallelLayout> ctrsWriter(FeaturesManager,
                                                                   compressedIndexBuilder,
                                                                   ctrsCalcer,
                                                                   ds.PermutationDependentFeatures,
                                                                   testDataSetId);
                    ctrsWriter.Write(permutationDependent);
                }
                CATBOOST_DEBUG_LOG << "Ctr computation for permutation #" << permutationId << " is finished" << Endl;
                {
                    TEstimatorsExecutor estimatorsExecutor(FeaturesManager,
                                                           Estimators,
                                                           dataSetsHolder.PermutationDataSets[permutationId]->CtrsEstimationPermutation,
                                                           localExecutor
                    );

                    TMaybe<ui32> testId;
                    if (LinkedTest && permutationId == 0) {
                        testId = testDataSetId;
                    }
                    TEstimatedFeaturesWriter<TFeatureParallelLayout> writer(FeaturesManager,
                                                                            compressedIndexBuilder,
                                                                            estimatorsExecutor,
                                                                            ds.PermutationDependentFeatures,
                                                                            testId);
                    writer.Write(permutationDependent);
                    CATBOOST_DEBUG_LOG << "Feature estimators for permutation #" << permutationId << " is finished" << Endl;
                }
            }
        }
        compressedIndexBuilder.Finish();

        return dataSetsHolder;
    }

    void TFeatureParallelDataSetHoldersBuilder::BuildTestTargetAndIndices(TFeatureParallelDataSetsHolder& dataSetsHolder,
                                                                          const TCtrTargets<NCudaLib::TMirrorMapping>& ctrsTarget) {
        const auto testMapping = NCudaLib::TMirrorMapping(ctrsTarget.TestSlice.Size());

        TMirrorBuffer<ui32> indices;
        indices.Reset(testMapping);
        MakeSequence(indices);
        TMirrorBuffer<ui32> inverseIndices = indices.CopyView();

        auto targets = TMirrorBuffer<float>::CopyMapping(indices);
        auto weights = TMirrorBuffer<float>::CopyMapping(indices);

        const auto* gpuObjects = dynamic_cast<const NCB::TGpuInputQuantizedObjectsDataProvider*>(LinkedTest->ObjectsData.Get());
        const bool hasGpuTargets = gpuObjects && (gpuObjects->GetGpuTargets().TargetCount > 0);
        if (hasGpuTargets) {
            const auto& gpuTargets = gpuObjects->GetGpuTargets();
            CB_ENSURE(gpuTargets.TargetCount == 1, "Feature-parallel training supports only one-dimensional GPU target");
            const auto& targetColumn = gpuTargets.Targets[0];
            const i32 deviceId = targetColumn.DeviceId;
            CB_ENSURE(deviceId >= 0, "Invalid device id for GPU target");
            NCudaLib::SetDevice(deviceId);

            auto getCudaStreamFromCudaArrayInterface = [] (ui64 stream) -> cudaStream_t {
                if (stream == 0) {
                    return 0;
                }
                if (stream == 1) {
                    return cudaStreamPerThread;
                }
                return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
            };
            const cudaStream_t caiStream = getCudaStreamFromCudaArrayInterface(targetColumn.Stream);

            const ui32 testSize = static_cast<ui32>(ctrsTarget.TestSlice.Size());
            CB_ENSURE(targetColumn.FullObjectCount == testSize, "GPU test target size mismatch");

            TSingleBuffer<float> targetSingle;
            targetSingle.Reset(NCudaLib::TSingleMapping(deviceId, testSize));
            TSingleBuffer<float> weightsSingle;
            weightsSingle.Reset(NCudaLib::TSingleMapping(deviceId, testSize));

            NCudaLib::GetCudaManager().WaitComplete(NCudaLib::TDevicesListBuilder::SingleDevice(deviceId));

            const void* srcTarget = reinterpret_cast<const void*>(static_cast<uintptr_t>(targetColumn.Data));
            NKernel::CopyStridedGpuInputToFloat(
                srcTarget,
                targetColumn.StrideBytes,
                testSize,
                static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(targetColumn.DType)),
                targetSingle.At(deviceId).Get(),
                caiStream
            );

            if (gpuTargets.HasWeights) {
                const auto& weightColumn = gpuTargets.Weights;
                CB_ENSURE(weightColumn.DeviceId == deviceId, "GPU weights must reside on the same device as GPU target");
                CB_ENSURE(weightColumn.FullObjectCount == testSize, "GPU test weight size mismatch");
                const cudaStream_t weightStream = getCudaStreamFromCudaArrayInterface(weightColumn.Stream);
                const void* srcWeight = reinterpret_cast<const void*>(static_cast<uintptr_t>(weightColumn.Data));
                NKernel::CopyStridedGpuInputToFloat(
                    srcWeight,
                    weightColumn.StrideBytes,
                    testSize,
                    static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(weightColumn.DType)),
                    weightsSingle.At(deviceId).Get(),
                    weightStream
                );
                CUDA_SAFE_CALL(cudaStreamSynchronize(weightStream));
            } else {
                NKernel::FillLearnTestWeights(
                    weightsSingle.At(deviceId).Get(),
                    /*learnSize*/ testSize,
                    /*totalSize*/ testSize,
                    /*learnValue*/ 1.0f,
                    /*testValue*/ 1.0f,
                    caiStream
                );
            }

            CUDA_SAFE_CALL(cudaStreamSynchronize(caiStream));

            NCudaLib::Reshard(targetSingle, targets, /*stream*/ 0);
            NCudaLib::Reshard(weightsSingle, weights, /*stream*/ 0);
        } else {
            targets.Write(*LinkedTest->TargetData->GetOneDimensionalTarget());
            weights.Write(GetWeights(*LinkedTest->TargetData));
        }

        dataSetsHolder.TestDataSet.Reset(new TFeatureParallelDataSet(*LinkedTest,
                                                                     dataSetsHolder.CompressedIndex,
                                                                     new TPermutationScope(),
                                                                     new TPermutationScope(),
                                                                     *dataSetsHolder.TestCatFeaturesDataSet,
                                                                     dataSetsHolder.GetCtrTargets(),
                                                                     TTarget<NCudaLib::TMirrorMapping>(targets.AsConstBuf(),
                                                                                                       weights.AsConstBuf(),
                                                                                                       indices.AsConstBuf(),
                                                                                                       /*isPairWeights*/ false),
                                                                     std::move(inverseIndices),
                                                                     GetPermutation(*LinkedTest, 0u, 1u))

        );

        dataSetsHolder.TestDataSet->LinkedHistoryForCtrs = dataSetsHolder.PermutationDataSets[0].Get();
    }

    void TFeatureParallelDataSetHoldersBuilder::BuildCompressedCatFeatures(const NCB::TTrainingDataProvider& dataProvider,
                                                                           TCompressedCatFeatureDataSet& dataset,
                                                                           NPar::ILocalExecutor* localExecutor) {
        TCompressedCatFeatureDataSetBuilder builder(dataProvider,
                                                    FeaturesManager,
                                                    dataset,
                                                    localExecutor);
        for (ui32 catFeature : FeaturesManager.GetCatFeatureIds()) {
            if (FeaturesManager.UseForTreeCtr(catFeature)) {
                builder.Add(catFeature);
            }
        }
        builder.Finish();
    }
}
