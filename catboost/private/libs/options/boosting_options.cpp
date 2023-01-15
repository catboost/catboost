#include "boosting_options.h"
#include "json_helper.h"

#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/logging_level.h>

#include <util/generic/ymath.h>

NCatboostOptions::TBoostingOptions::TBoostingOptions(ETaskType taskType)
    : LearningRate("learning_rate", 0.03)
    , FoldLenMultiplier("fold_len_multiplier", 2.0)
    , PermutationBlockSize("fold_permutation_block", 0)
    , IterationCount("iterations", 1000)
    , PermutationCount("permutation_count", 4)
    , OverfittingDetector("od_config", TOverfittingDetectorOptions())
    , BoostingType("boosting_type", EBoostingType::Plain)
    , BoostFromAverage("boost_from_average", false)
    , ApproxOnFullHistory("approx_on_full_history", false, taskType)
    , ModelShrinkRate("model_shrink_rate", 0.0f, taskType)
    , ModelShrinkMode("model_shrink_mode", EModelShrinkMode::Constant, taskType)
    , Langevin("langevin", false)
    , DiffusionTemperature("diffusion_temperature", 0.0f)
    , PosteriorSampling("posterior_sampling", false, taskType)
    , MinFoldSize("min_fold_size", 100, taskType)
    , DataPartitionType("data_partition", EDataPartitionType::FeatureParallel, taskType)
{
}

void NCatboostOptions::TBoostingOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options,
            &LearningRate, &FoldLenMultiplier, &PermutationBlockSize, &IterationCount, &OverfittingDetector,
            &BoostingType, &BoostFromAverage, &PermutationCount, &MinFoldSize, &ApproxOnFullHistory,
            &DataPartitionType, &ModelShrinkRate, &ModelShrinkMode, &Langevin, &DiffusionTemperature, &PosteriorSampling);

    Validate();
}

void NCatboostOptions::TBoostingOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options,
            LearningRate, FoldLenMultiplier, PermutationBlockSize, IterationCount, OverfittingDetector,
            BoostingType, BoostFromAverage, PermutationCount, MinFoldSize, ApproxOnFullHistory,
            DataPartitionType, ModelShrinkRate, ModelShrinkMode, PosteriorSampling);
    if (Langevin) {
        SaveFields(options, Langevin, DiffusionTemperature);
    }
}

bool NCatboostOptions::TBoostingOptions::operator==(const TBoostingOptions& rhs) const {
    return std::tie(LearningRate, FoldLenMultiplier, PermutationBlockSize, IterationCount, OverfittingDetector,
            ApproxOnFullHistory, BoostingType, BoostFromAverage, PermutationCount,
            MinFoldSize, DataPartitionType, ModelShrinkRate, ModelShrinkMode, Langevin, DiffusionTemperature, PosteriorSampling) ==
        std::tie(rhs.LearningRate, rhs.FoldLenMultiplier, rhs.PermutationBlockSize, rhs.IterationCount,
                rhs.OverfittingDetector, rhs.ApproxOnFullHistory, rhs.BoostingType, rhs.BoostFromAverage,
                rhs.PermutationCount, rhs.MinFoldSize, rhs.DataPartitionType, rhs.ModelShrinkRate, rhs.ModelShrinkMode,
                rhs.Langevin, rhs.DiffusionTemperature, rhs.PosteriorSampling);
}

bool NCatboostOptions::TBoostingOptions::operator!=(const TBoostingOptions& rhs) const {
    return !(rhs == *this);
}

void NCatboostOptions::TBoostingOptions::Validate() const {
    OverfittingDetector->Validate();
    CB_ENSURE(FoldLenMultiplier.Get() > 1.0f, "fold len multiplier should be greater than 1");
    CB_ENSURE(IterationCount.Get() > 0, "Iterations count should be positive");

    CB_ENSURE(PermutationCount.Get() > 0, "Permutation count should be positive");

    CB_ENSURE(MinFoldSize.GetUnchecked() > 0, "Min fold size should be positive");

    if (BoostingType.IsSet()) {
        if (DataPartitionType.GetUnchecked() == EDataPartitionType::DocParallel) {
            CB_ENSURE(BoostingType == EBoostingType::Plain, "Can't use ordered boosting in doc-parallel mode");
        }
    }

    CB_ENSURE(!(ApproxOnFullHistory.GetUnchecked() && BoostingType.Get() == EBoostingType::Plain), "Can't use approx-on-full-history with Plain boosting-type");
    if (LearningRate.IsSet()) {
        CB_ENSURE(Abs(LearningRate.Get()) > std::numeric_limits<float>::epsilon(), "Learning rate should be non-zero");
        if (LearningRate.Get() > 1) {
            CATBOOST_WARNING_LOG
            << "learning rate is greater than 1. You probably need to decrease learning rate." << Endl;
        }
    }

    switch(ModelShrinkMode.GetUnchecked()) {
        case EModelShrinkMode::Constant: {
            const float actualShrinkCoef = ModelShrinkRate.GetUnchecked() * LearningRate.Get();
            CB_ENSURE(actualShrinkCoef >= 0.0f && actualShrinkCoef < 1.0f,
                "For Constant shrink mode: (model_shrink_rate * learning_rate) should be in [0, 1).");
            break;
        }
        case EModelShrinkMode::Decreasing: {
            CB_ENSURE(
                ModelShrinkRate.GetUnchecked() >= 0.0 && ModelShrinkRate.GetUnchecked() < 1.0,
                "For Decreasing shrink mode: model shrink rate should be in [0, 1)."
            );
            break;
        }
    }

    CB_ENSURE(
        DiffusionTemperature >= 0.0,
        "Diffusion temperature should be non-negative"
    );
}
