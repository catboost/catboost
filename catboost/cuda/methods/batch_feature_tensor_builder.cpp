#include "batch_feature_tensor_builder.h"

#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/gpu_data/kernels.h>
#include <catboost/cuda/cuda_lib/device_subtasks_helper.h>

namespace NCatboostCuda {
    void TBatchFeatureTensorBuilder::VisitCtrBinBuilders(const TSingleBuffer<const ui32>& baseTensorIndices,
                                                         const TFeatureTensor& baseTensor,
                                                         const TVector<ui32>& catFeatureIds,
                                                         TBatchFeatureTensorBuilder::TFeatureTensorVisitor& featureTensorVisitor) {
        TSingleBuffer<ui32> currentBins;
        {
            TSingleBuffer<ui32> tmp;
            TBinBuilder::ComputeCurrentBins(baseTensorIndices, tmp, currentBins, 0);
        }

        const ui32 buildStreams = RequestStream(static_cast<ui32>(catFeatureIds.size()));
        NCudaLib::GetCudaManager().WaitComplete(); //ensure all prev command results will be visibile

        for (ui32 i = 0; i < catFeatureIds.size(); i += buildStreams) {
            //submit build tensors
            //do not merge with second part. ctrBinBuilder should be async wrt host
            {
                auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile("ctrBinsBuilder");
                for (ui32 j = 0; j < buildStreams; ++j) {
                    const ui32 featureIndex = i + j;
                    if (featureIndex < catFeatureIds.size()) {
                        const ui32 catFeatureId = catFeatureIds[featureIndex];

                        if (CatFeatures.GetStorageType() == EGpuCatFeaturesStorage::GpuRam) {
                            CtrBinBuilders[j]
                                .SetIndices(baseTensorIndices)
                                .AddCompressedBinsWithCurrentBinsCache(currentBins,
                                                                       CatFeatures.GetFeatureGpu(catFeatureId),
                                                                       FeaturesManager.GetBinCount(
                                                                           catFeatureId));
                        } else {
                            CtrBinBuilders[j]
                                .SetIndices(baseTensorIndices)
                                .AddCompressedBinsWithCurrentBinsCache(currentBins,
                                                                       CatFeatures.GetFeatureCpu(catFeatureId),
                                                                       FeaturesManager.GetBinCount(
                                                                           catFeatureId));
                        }
                    }
                }
            }

            //visit tensors
            for (ui32 j = 0; j < buildStreams; ++j) {
                const ui32 featureIndex = i + j;
                if (featureIndex < catFeatureIds.size()) {
                    const ui32 catFeatureId = catFeatureIds[featureIndex];
                    auto featureTensor = baseTensor;
                    featureTensor.AddCatFeature(catFeatureId);

                    featureTensorVisitor(featureTensor,
                                         CtrBinBuilders[j]);
                }
            }
        }
    }

    ui32 TBatchFeatureTensorBuilder::RequestStream(ui32 featuresToBuild) {
        const ui32 buildStreams = std::min<ui32>(TensorBuilderStreams, featuresToBuild);
        {
            for (ui32 i = static_cast<ui32>(BuilderStreams.size()); i < buildStreams; ++i) {
                BuilderStreams.push_back(buildStreams > 1 ? NCudaLib::GetCudaManager().RequestStream()
                                                          : NCudaLib::GetCudaManager().DefaultStream());
                CtrBinBuilders.push_back(TBinBuilder(BuilderStreams.back().GetId()));
            }
        }
        return buildStreams;
    }
}
