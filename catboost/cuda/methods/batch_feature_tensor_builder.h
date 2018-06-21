#pragma once

#include <catboost/cuda/ctrs/ctr_bins_builder.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/gpu_data/cat_features_dataset.h>

namespace NCatboostCuda {
    class TBatchFeatureTensorBuilder {
    public:
        using TBinBuilder = TCtrBinBuilder<NCudaLib::TSingleMapping>;
        using TFeatureTensorVisitor = std::function<void(const TFeatureTensor& tensor, TCtrBinBuilder<NCudaLib::TSingleMapping>&)>;

        TBatchFeatureTensorBuilder(const TBinarizedFeaturesManager& featuresManager,
                                   const TCompressedCatFeatureDataSet& catFeatures,
                                   ui32 tensorBuilderStreams)
            : FeaturesManager(featuresManager)
            , CatFeatures(catFeatures)
            , TensorBuilderStreams(tensorBuilderStreams)
        {
        }

        void VisitCtrBinBuilders(const TSingleBuffer<const ui32>& baseTensorIndices,
                                 const TFeatureTensor& baseTensor,
                                 const TVector<ui32>& catFeatureIds,
                                 TFeatureTensorVisitor& featureTensorVisitor);

    private:
        ui32 RequestStream(ui32 featuresToBuild);

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const TCompressedCatFeatureDataSet& CatFeatures;
        ui32 TensorBuilderStreams;

        TVector<TComputationStream> BuilderStreams;
        TVector<TBinBuilder> CtrBinBuilders;
    };
}
