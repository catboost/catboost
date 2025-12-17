#include "dataset_helpers.h"
#include "feature_layout_doc_parallel.h"
#include "feature_layout_feature_parallel.h"

#include <catboost/cuda/data/gpu_input_provider.h>
#include <catboost/cuda/gpu_data/kernel/gpu_input_utils.cuh>
#include <catboost/cuda/gpu_data/kernel/gpu_input_targets.cuh>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>

#include <cuda_runtime_api.h>

#include <util/generic/maybe.h>
#include <util/generic/scope.h>

THolder<NCatboostCuda::TCtrTargets<NCudaLib::TMirrorMapping>> NCatboostCuda::BuildCtrTarget(const NCatboostCuda::TBinarizedFeaturesManager& featuresManager,
                                                                                            const NCB::TTrainingDataProvider& dataProvider,
                                                                                            const NCB::TTrainingDataProvider* test) {
    const auto* gpuObjects = dynamic_cast<const NCB::TGpuInputQuantizedObjectsDataProvider*>(dataProvider.ObjectsData.Get());
    const bool hasGpuTargets = gpuObjects && (gpuObjects->GetGpuTargets().TargetCount > 0);

    if (!hasGpuTargets) {
        TVector<float> joinedTarget = Join((*dataProvider.TargetData->GetTarget())[0],
                                           test ? MakeMaybe((*test->TargetData->GetTarget())[0]) : Nothing()); // espetrov: fix for multi-target + cat features

        THolder<TCtrTargets<NCudaLib::TMirrorMapping>> ctrsTargetPtr;
        ctrsTargetPtr = MakeHolder<TCtrTargets<NCudaLib::TMirrorMapping>>();
        auto& ctrsTarget = *ctrsTargetPtr;
        ctrsTarget.BinarizedTarget = BuildBinarizedTarget(featuresManager,
                                                          joinedTarget);

        ctrsTarget.WeightedTarget.Reset(NCudaLib::TMirrorMapping(joinedTarget.size()));
        ctrsTarget.Weights.Reset(NCudaLib::TMirrorMapping(joinedTarget.size()));

        ctrsTarget.LearnSlice = TSlice(0, dataProvider.GetObjectCount());
        ctrsTarget.TestSlice = TSlice(dataProvider.GetObjectCount(), joinedTarget.size());

        TVector<float> ctrWeights;
        ctrWeights.resize(joinedTarget.size(), 1.0f);

        TVector<float> ctrWeightedTargets(joinedTarget.begin(), joinedTarget.end());

        double totalWeight = 0;
        for (ui32 i = (ui32)ctrsTarget.LearnSlice.Right; i < ctrWeights.size(); ++i) {
            ctrWeights[i] = 0;
        }

        for (ui32 i = 0; i < ctrWeightedTargets.size(); ++i) {
            ctrWeightedTargets[i] *= ctrWeights[i];
            totalWeight += ctrWeights[i];
        }

        ctrsTarget.TotalWeight = (float)totalWeight;
        ctrsTarget.WeightedTarget.Write(ctrWeightedTargets);
        ctrsTarget.Weights.Write(ctrWeights);

        CB_ENSURE(ctrsTarget.IsTrivialWeights());

        if (!dataProvider.ObjectsGrouping->IsTrivial() && featuresManager.GetCatFeatureOptions().CtrHistoryUnit == ECtrHistoryUnit::Group) {
            const ui64 groupCountLearn = dataProvider.ObjectsGrouping->GetGroupCount();
            TVector<ui32> groupIds;
            groupIds.reserve(joinedTarget.size());

            for (ui32 groupId = 0; groupId < groupCountLearn; ++groupId) {
                ui32 groupSize = dataProvider.ObjectsGrouping->GetGroup(groupId).GetSize();
                for (ui32 j  = 0; j < groupSize; ++j) {
                    groupIds.push_back(groupId);
                }
            }
            const ui64 groupCountTest = test ? test->ObjectsGrouping->GetGroupCount() : 0;

            for (ui32 groupId = 0; groupId < groupCountTest; ++groupId) {
                ui32 groupSize = test->ObjectsGrouping->GetGroup(groupId).GetSize();
                for (ui32 j = 0; j < groupSize; ++j) {
                    groupIds.push_back(groupId + groupCountLearn);
                }
            }


            auto tmp = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(groupIds.size()));
            tmp.Write(groupIds);
            ctrsTarget.GroupIds = tmp.ConstCopyView();
        }
        return ctrsTargetPtr;
    }

    const auto& gpuTargets = gpuObjects->GetGpuTargets();
    CB_ENSURE(gpuTargets.TargetCount == 1, "CTR target requires one-dimensional target");
    CB_ENSURE(gpuTargets.Targets.size() == 1, "GPU targets size mismatch");

    const ui32 learnSize = dataProvider.GetObjectCount();
    const ui32 totalSize = learnSize + (test ? test->GetObjectCount() : 0);

    THolder<TCtrTargets<NCudaLib::TMirrorMapping>> ctrsTargetPtr = MakeHolder<TCtrTargets<NCudaLib::TMirrorMapping>>();
    auto& ctrsTarget = *ctrsTargetPtr;

    ctrsTarget.LearnSlice = TSlice(0, learnSize);
    ctrsTarget.TestSlice = TSlice(learnSize, totalSize);
    ctrsTarget.TotalWeight = static_cast<float>(learnSize);

    const i32 deviceId = gpuTargets.Targets[0].DeviceId;
    CB_ENSURE(deviceId >= 0, "Invalid device id for GPU target");
    NCudaLib::SetDevice(deviceId);

    const auto getCudaStreamFromCudaArrayInterface = [] (ui64 stream) -> cudaStream_t {
        if (stream == 0) {
            return 0;
        }
        if (stream == 1) {
            return cudaStreamPerThread;
        }
        return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
    };

    const auto& targetColumn = gpuTargets.Targets[0];
    CB_ENSURE(targetColumn.FullObjectCount == learnSize, "GPU target size mismatch");

    const cudaStream_t caiStream = getCudaStreamFromCudaArrayInterface(targetColumn.Stream);

    // Build joined target/weights on a single device, then replicate to mirror mapping.
    TSingleBuffer<float> joinedTargetSingle;
    joinedTargetSingle.Reset(NCudaLib::TSingleMapping(deviceId, totalSize));
    TSingleBuffer<float> joinedWeightsSingle;
    joinedWeightsSingle.Reset(NCudaLib::TSingleMapping(deviceId, totalSize));
    TSingleBuffer<ui8> joinedBinarizedSingle;
    joinedBinarizedSingle.Reset(NCudaLib::TSingleMapping(deviceId, totalSize));

    // Ensure handle-based buffers are materialized before using raw CUDA runtime calls.
    NCudaLib::GetCudaManager().WaitComplete(NCudaLib::TDevicesListBuilder::SingleDevice(deviceId));

    CUDA_SAFE_CALL(cudaMemsetAsync(joinedTargetSingle.At(deviceId).Get(), 0, static_cast<size_t>(totalSize) * sizeof(float), caiStream));

    const void* srcTarget = reinterpret_cast<const void*>(static_cast<uintptr_t>(targetColumn.Data));
    NKernel::CopyStridedGpuInputToFloat(
        srcTarget,
        targetColumn.StrideBytes,
        learnSize,
        static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(targetColumn.DType)),
        joinedTargetSingle.At(deviceId).Get(),
        caiStream
    );

    NKernel::FillLearnTestWeights(
        joinedWeightsSingle.At(deviceId).Get(),
        learnSize,
        totalSize,
        /*learnValue*/ 1.0f,
        /*testValue*/ 0.0f,
        caiStream
    );

    CUDA_SAFE_CALL(cudaMemsetAsync(joinedBinarizedSingle.At(deviceId).Get(), 0, static_cast<size_t>(totalSize) * sizeof(ui8), caiStream));

    if (featuresManager.HasTargetBinarization()) {
        const auto& borders = featuresManager.GetTargetBorders();
        const ui32 borderCount = static_cast<ui32>(borders.size());
        CB_ENSURE(borderCount <= 255, "Target border count is too large for ui8 binarization");

        float* bordersDev = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&bordersDev, static_cast<size_t>(borderCount) * sizeof(float)));
        Y_DEFER { cudaFree(bordersDev); };
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            bordersDev,
            borders.data(),
            static_cast<size_t>(borderCount) * sizeof(float),
            cudaMemcpyHostToDevice,
            caiStream
        ));
        NKernel::BinarizeToUi8(
            joinedTargetSingle.At(deviceId).Get(),
            totalSize,
            bordersDev,
            borderCount,
            joinedBinarizedSingle.At(deviceId).Get(),
            caiStream
        );
    }

    CUDA_SAFE_CALL(cudaStreamSynchronize(caiStream));

    ctrsTarget.WeightedTarget.Reset(NCudaLib::TMirrorMapping(totalSize));
    ctrsTarget.Weights.Reset(NCudaLib::TMirrorMapping(totalSize));
    ctrsTarget.BinarizedTarget.Reset(NCudaLib::TMirrorMapping(totalSize));

    NCudaLib::Reshard(joinedTargetSingle, ctrsTarget.WeightedTarget, /*stream*/ 0);
    NCudaLib::Reshard(joinedWeightsSingle, ctrsTarget.Weights, /*stream*/ 0);
    NCudaLib::Reshard(joinedBinarizedSingle, ctrsTarget.BinarizedTarget, /*stream*/ 0);

    if (!dataProvider.ObjectsGrouping->IsTrivial() && featuresManager.GetCatFeatureOptions().CtrHistoryUnit == ECtrHistoryUnit::Group) {
        const ui64 groupCountLearn = dataProvider.ObjectsGrouping->GetGroupCount();
        TVector<ui32> groupIds;
        groupIds.reserve(totalSize);

        for (ui32 groupId = 0; groupId < groupCountLearn; ++groupId) {
            ui32 groupSize = dataProvider.ObjectsGrouping->GetGroup(groupId).GetSize();
            for (ui32 j  = 0; j < groupSize; ++j) {
                groupIds.push_back(groupId);
            }
        }
        const ui64 groupCountTest = test ? test->ObjectsGrouping->GetGroupCount() : 0;

        for (ui32 groupId = 0; groupId < groupCountTest; ++groupId) {
            ui32 groupSize = test->ObjectsGrouping->GetGroup(groupId).GetSize();
            for (ui32 j = 0; j < groupSize; ++j) {
                groupIds.push_back(groupId + groupCountLearn);
            }
        }


        auto tmp = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(groupIds.size()));
        tmp.Write(groupIds);
        ctrsTarget.GroupIds = tmp.ConstCopyView();
    }
    return ctrsTargetPtr;
}

TVector<ui32> NCatboostCuda::GetLearnFeatureIds(NCatboostCuda::TBinarizedFeaturesManager& featuresManager) {
    TSet<ui32> featureIdsSet;
    auto ctrTypes = featuresManager.GetKnownSimpleCtrTypes();

    for (auto floatFeature : featuresManager.GetFloatFeatureIds()) {
        if (featuresManager.GetBinCount(floatFeature) > 1) {
            featureIdsSet.insert(floatFeature);
        }
    }
    for (auto catFeature : featuresManager.GetCatFeatureIds()) {
        if (featuresManager.UseForOneHotEncoding(catFeature)) {
            if (featuresManager.GetBinCount(catFeature) > 1) {
                featureIdsSet.insert(catFeature);
            }
        }

        if (featuresManager.UseForCtr(catFeature)) {
            for (auto& ctr : ctrTypes) {
                const auto simpleCtrsForType = featuresManager.CreateSimpleCtrsForType(catFeature,
                                                                                       ctr);
                for (auto ctrFeatureId : simpleCtrsForType) {
                    featureIdsSet.insert(ctrFeatureId);
                }
            }
        }
    }
    TSet<ui32> combinationCtrIds;

    for (auto& ctr : ctrTypes) {
        auto combinationCtrs = featuresManager.CreateCombinationCtrForType(ctr);
        for (auto ctrFeatureId : combinationCtrs) {
            TFeatureTensor tensor = featuresManager.GetCtr(ctrFeatureId).FeatureTensor;
            bool hasUnknownFeatures = false;
            CB_ENSURE(tensor.GetSplits().size() == 0);

            for (auto featureId : tensor.GetCatFeatures()) {
                if (!featureIdsSet.contains(featureId)) {
                    hasUnknownFeatures = true;
                    break;
                }
            }
            for (auto binarySplit : tensor.GetSplits()) {
                if (!featureIdsSet.contains(binarySplit.FeatureId)) {
                    hasUnknownFeatures = true;
                    break;
                }
            }
            if (!hasUnknownFeatures) {
                combinationCtrIds.insert(ctrFeatureId);
            }
        }
    }
    featureIdsSet.insert(combinationCtrIds.begin(), combinationCtrIds.end());

    auto estimatedFeatures = featuresManager.GetEstimatedFeatureIds();
    featureIdsSet.insert(estimatedFeatures.begin(), estimatedFeatures.end());

    auto featureBundleIds = featuresManager.GetExclusiveFeatureBundleIds();
    featureIdsSet.insert(featureBundleIds.begin(), featureBundleIds.end());

    return TVector<ui32>(featureIdsSet.begin(), featureIdsSet.end());
}

namespace NCatboostCuda {
    TMirrorBuffer<ui8> BuildBinarizedTarget(const TBinarizedFeaturesManager& featuresManager, const TVector<float>& targets) {
        TVector<ui8> binarizedTarget;
        if (featuresManager.HasTargetBinarization()) {
            auto& borders = featuresManager.GetTargetBorders();
            binarizedTarget = NCB::BinarizeLine<ui8>(targets,
                                                     ENanMode::Forbidden,
                                                     borders);
        } else {
            binarizedTarget.resize(targets.size(), 0);
        }

        TMirrorBuffer<ui8> binarizedTargetGpu = TMirrorBuffer<ui8>::Create(NCudaLib::TMirrorMapping(binarizedTarget.size()));
        binarizedTargetGpu.Write(binarizedTarget);
        return binarizedTargetGpu;
    }

    void SplitByPermutationDependence(const TBinarizedFeaturesManager& featuresManager, const TVector<ui32>& features,
                                      const ui32 permutationCount, TVector<ui32>* permutationIndependent,
                                      TVector<ui32>* permutationDependent) {
        if (permutationCount == 1) {
            //            shortcut
            (*permutationIndependent) = features;
            return;
        }
        permutationDependent->clear();
        permutationIndependent->clear();
        for (const auto& feature : features) {
            const bool permutationDependentCtr = featuresManager.IsCtr(feature) && featuresManager.IsPermutationDependent(featuresManager.GetCtr(feature));
            const bool onlineEstimatedFeature = featuresManager.IsEstimatedFeature(feature) && featuresManager.GetEstimatedFeature(feature).EstimatorId.IsOnline;

            const bool needPermutationFlag = permutationDependentCtr || onlineEstimatedFeature;
            if (needPermutationFlag) {
                permutationDependent->push_back(feature);
            } else {
                permutationIndependent->push_back(feature);
            }
        }
    }

    template class TFloatAndOneHotFeaturesWriter<TFeatureParallelLayout>;
    template class TFloatAndOneHotFeaturesWriter<TDocParallelLayout>;

    template class TCtrsWriter<TFeatureParallelLayout>;
    template class TCtrsWriter<TDocParallelLayout>;

    template class TEstimatedFeaturesWriter<TFeatureParallelLayout>;
    template class TEstimatedFeaturesWriter<TDocParallelLayout>;


}
