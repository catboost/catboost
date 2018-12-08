#pragma once

#include "doc_parallel_dataset.h"
#include <catboost/libs/helpers/interrupt.h>

namespace NCatboostCuda {
    class TDocParallelDataSetBuilder {
    public:
        using TDataSetLayout = TDocParallelLayout;

        TDocParallelDataSetBuilder(TBinarizedFeaturesManager& featuresManager,
                                   const NCB::TTrainingDataProvider& dataProvider,
                                   const NCB::TTrainingDataProvider* linkedTest = nullptr)
            : FeaturesManager(featuresManager)
            , DataProvider(dataProvider)
            , LinkedTest(linkedTest)
        {
        }

        TDocParallelDataSetsHolder BuildDataSet(const ui32 permutationCount);

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const NCB::TTrainingDataProvider& DataProvider;
        const NCB::TTrainingDataProvider* LinkedTest;
    };
}
