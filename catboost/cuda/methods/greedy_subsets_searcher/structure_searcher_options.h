#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/cuda/gpu_data/bootstrap.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/libs/options/oblivious_tree_options.h>
#include <catboost/cuda/methods/weak_target_helpers.h>

namespace NCatboostCuda {
    struct TTreeStructureSearcherOptions {
        EScoreFunction ScoreFunction = EScoreFunction::Correlation;
        NCatboostOptions::TBootstrapConfig BootstrapOptions = NCatboostOptions::TBootstrapConfig(ETaskType::GPU);
        ui32 MaxLeaves = 64;
        ui32 MaxDepth = 6;
        double L2Reg = 3.0;
        EGrowingPolicy Policy = EGrowingPolicy::ObliviousTree;

        double MinLeafSize = 1;
        double RandomStrength = 0;
    };
}
