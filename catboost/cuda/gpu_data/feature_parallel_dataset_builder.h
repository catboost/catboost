#pragma once

#include "feature_parallel_dataset.h"

namespace NCatboostCuda {
    //Test dataset will be linked on first permutation (direct indexing)
    class TFeatureParallelDataSetHoldersBuilder {
    public:
        using TDataSetLayout = TFeatureParallelLayout;

        TFeatureParallelDataSetHoldersBuilder(TBinarizedFeaturesManager& featuresManager,
                                              const TDataProvider& dataProvider,
                                              const TDataProvider* linkedTest = nullptr,
                                              ui32 blockSize = 1,
                                              EGpuCatFeaturesStorage catFeaturesStorage = EGpuCatFeaturesStorage::GpuRam)
            : FeaturesManager(featuresManager)
            , DataProvider(dataProvider)
            , LinkedTest(linkedTest)
            , DataProviderPermutationBlockSize(blockSize)
            , CatFeaturesStorage(catFeaturesStorage)
        {
        }

        TFeatureParallelDataSetsHolder BuildDataSet(const ui32 permutationCount);
    private:
        void BuildTestTargetAndIndices(TFeatureParallelDataSetsHolder& dataSetsHolder,
                                       const TCtrTargets<NCudaLib::TMirrorMapping>& ctrsTarget);

        void BuildCompressedCatFeatures(const TDataProvider& dataProvider,
                                        TCompressedCatFeatureDataSet& dataset);

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const TDataProvider& DataProvider;
        const TDataProvider* LinkedTest;
        ui32 DataProviderPermutationBlockSize = 1;
        EGpuCatFeaturesStorage CatFeaturesStorage;
    };

}
