#pragma once

#include "feature_parallel_dataset.h"

#include <library/cpp/threading/local_executor/local_executor.h>

namespace NCatboostCuda {
    //Test dataset will be linked on first permutation (direct indexing)
    class TFeatureParallelDataSetHoldersBuilder {
    public:
        using TDataSetLayout = TFeatureParallelLayout;

        TFeatureParallelDataSetHoldersBuilder(TBinarizedFeaturesManager& featuresManager,
                                              const NCB::TTrainingDataProvider& dataProvider,
                                              const NCB::TFeatureEstimators& estimators,
                                              const NCB::TTrainingDataProvider* linkedTest = nullptr,
                                              ui32 blockSize = 1,
                                              EGpuCatFeaturesStorage catFeaturesStorage = EGpuCatFeaturesStorage::GpuRam)
            : FeaturesManager(featuresManager)
            , DataProvider(dataProvider)
            , Estimators(estimators)
            , LinkedTest(linkedTest)
            , DataProviderPermutationBlockSize(blockSize)
            , CatFeaturesStorage(catFeaturesStorage)
        {
        }

        TFeatureParallelDataSetsHolder BuildDataSet(ui32 permutationCount,
                                                    NPar::ILocalExecutor* localExecutor);

    private:
        void BuildTestTargetAndIndices(TFeatureParallelDataSetsHolder& dataSetsHolder,
                                       const TCtrTargets<NCudaLib::TMirrorMapping>& ctrsTarget);

        void BuildCompressedCatFeatures(const NCB::TTrainingDataProvider& dataProvider,
                                        TCompressedCatFeatureDataSet& dataset,
                                        NPar::ILocalExecutor* localExecutor);

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const NCB::TTrainingDataProvider& DataProvider;
        const NCB::TFeatureEstimators& Estimators;
        const NCB::TTrainingDataProvider* LinkedTest;
        ui32 DataProviderPermutationBlockSize = 1;
        EGpuCatFeaturesStorage CatFeaturesStorage;
    };

}
