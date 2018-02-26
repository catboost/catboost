#pragma once

#include "histograms_helper.h"
#include "tree_ctrs_dataset.h"
#include <catboost/cuda/utils/countdown_latch.h>
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>

#include <util/generic/map.h>
#include <util/generic/hash.h>
#include <util/generic/set.h>
#include <catboost/cuda/gpu_data/kernels.h>
#include <catboost/cuda/cuda_lib/device_subtasks_helper.h>

namespace NCatboostCuda {
    template <NCudaLib::EPtrType CatFeaturesStoragePtrType>
    class TBatchFeatureTensorBuilder {
    public:
        using TBinBuilder = TCtrBinBuilder<NCudaLib::TSingleMapping>;

        TBatchFeatureTensorBuilder(const TBinarizedFeaturesManager& featuresManager,
                                   const TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>& catFeatures,
                                   ui32 tensorBuilderStreams)
            : FeaturesManager(featuresManager)
            , CatFeatures(catFeatures)
            , TensorBuilderStreams(tensorBuilderStreams)
        {
        }

        template <class TUi32,
                  class TFeatureTensorVisitor>
        void VisitCtrBinBuilders(const TSingleBuffer<TUi32>& baseTensorIndices,
                                 const TFeatureTensor& baseTensor,
                                 const TVector<ui32>& catFeatureIds,
                                 TFeatureTensorVisitor& featureTensorVisitor) {
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

                            CtrBinBuilders[j]
                                .SetIndices(baseTensorIndices)
                                .AddCompressedBinsWithCurrentBinsCache(currentBins,
                                                                       CatFeatures.GetFeature(catFeatureId),
                                                                       FeaturesManager.GetBinCount(catFeatureId));
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

    private:
        ui32 RequestStream(ui32 featuresToBuild) {
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

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>& CatFeatures;
        ui32 TensorBuilderStreams;

        TVector<TComputationStream> BuilderStreams;
        TVector<TBinBuilder> CtrBinBuilders;
    };
}
