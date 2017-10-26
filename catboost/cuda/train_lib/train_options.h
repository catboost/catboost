#pragma once


#include <catboost/cuda/train_lib/application_options.h>
#include <catboost/cuda/data/binarization_config.h>
#include <catboost/cuda/data/load_config.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/methods/oblivious_tree.h>
#include <catboost/cuda/methods/boosting.h>
#include <catboost/cuda/targets/target_options.h>

namespace NCatboostCuda
{
    struct TTrainCatBoostOptions
    {
        TApplicationOptions ApplicationOptions;
        TFeatureManagerOptions FeatureManagerOptions;
        TObliviousTreeLearnerOptions TreeConfig;
        TBoostingOptions BoostingOptions;
        TOutputFilesOptions OutputFilesOptions;
        TTargetOptions TargetOptions;
    };
}


