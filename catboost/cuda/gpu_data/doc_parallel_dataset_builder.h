#pragma once

#include "doc_parallel_dataset.h"
#include "compressed_index_builder.h"
#include "gpu_grid_creator.h"
#include "ctr_helper.h"
#include "batch_binarized_ctr_calcer.h"
#include "dataset_helpers.h"

#include <catboost/cuda/ctrs/ctr_calcers.h>
#include <catboost/cuda/ctrs/ctr.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/cuda/cuda_lib/device_subtasks_helper.h>

namespace NCatboostCuda {

    class TDocParallelDataSetBuilder {
    public:
        using TDataSetLayout = TDocParallelLayout;

        TDocParallelDataSetBuilder(TBinarizedFeaturesManager& featuresManager,
                                   const TDataProvider& dataProvider,
                                   const TDataProvider* linkedTest = nullptr)
            : FeaturesManager(featuresManager)
            , DataProvider(dataProvider)
            , LinkedTest(linkedTest)
        {
        }

        TDocParallelDataSetsHolder BuildDataSet(const ui32 permutationCount);

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const TDataProvider& DataProvider;
        const TDataProvider* LinkedTest;
    };
}
