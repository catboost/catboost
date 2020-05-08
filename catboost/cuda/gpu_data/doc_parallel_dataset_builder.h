#pragma once

#include "doc_parallel_dataset.h"
#include <catboost/libs/helpers/interrupt.h>

#include <library/cpp/threading/local_executor/local_executor.h>

namespace NCatboostCuda {
    class TDocParallelDataSetBuilder {
    public:
        using TDataSetLayout = TDocParallelLayout;

        TDocParallelDataSetBuilder(TBinarizedFeaturesManager& featuresManager,
                                   const NCB::TTrainingDataProvider& dataProvider,
                                   const NCB::TFeatureEstimators& estimators,
                                   const NCB::TTrainingDataProvider* linkedTest = nullptr)
            : FeaturesManager(featuresManager)
            , DataProvider(dataProvider)
            , Estimators(estimators)
            , LinkedTest(linkedTest)
        {
        }

        TDocParallelDataSetsHolder BuildDataSet(const ui32 permutationCount,
                                                NPar::TLocalExecutor* localExecutor);

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const NCB::TTrainingDataProvider& DataProvider;
        const NCB::TFeatureEstimators& Estimators;
        const NCB::TTrainingDataProvider* LinkedTest;
    };
}
