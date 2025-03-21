#include "catboost_options.h"

#include "json_helper.h"
#include "restrictions.h"

#include <catboost/libs/helpers/json_helpers.h>

#include <library/cpp/json/json_reader.h>

#include <util/generic/algorithm.h>
#include <util/generic/set.h>
#include <util/string/cast.h>
#include <util/system/info.h>
#include <util/string/builder.h>
#include <util/generic/hash_set.h>

template <>
void Out<NCatboostOptions::TCatBoostOptions>(IOutputStream& out, const NCatboostOptions::TCatBoostOptions& options) {
    NJson::TJsonValue json;
    options.Save(&json);
    out << WriteTJsonValue(json);
}

template <>
inline TCatboostOptions FromString<NCatboostOptions::TCatBoostOptions>(const TString& str) {
    NJson::TJsonValue json;
    NJson::ReadJsonTree(str, &json, true);
    return NCatboostOptions::LoadOptions(json);
}

static std::tuple<ui32, ui32, ELeavesEstimation, double> GetEstimationMethodDefaults(
    ETaskType taskType,
    const NCatboostOptions::TLossDescription& lossFunctionConfig
) {
    ui32 defaultNewtonIterations = 1;
    ui32 defaultGradientIterations = 1;
    ELeavesEstimation defaultEstimationMethod = ELeavesEstimation::Newton;
    double defaultL2Reg = 3.0;

    switch (lossFunctionConfig.GetLossFunction()) {
        case ELossFunction::MultiRMSE: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::MultiRMSEWithMissingValues: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::SurvivalAft: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::RMSE: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::LogCosh: {
            defaultEstimationMethod = ELeavesEstimation::Exact;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::Cox: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::RMSEWithUncertainty: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::Lq: {
            CB_ENSURE(lossFunctionConfig.GetLossParamsMap().contains("q"), "Param q is mandatory for Lq loss");
            defaultEstimationMethod = ELeavesEstimation::Newton;
            const auto q = GetLqParam(lossFunctionConfig);
            if (q < 2) {
                defaultEstimationMethod = ELeavesEstimation::Gradient;
            }
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::QueryRMSE: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::QuerySoftMax: {
            defaultEstimationMethod = ELeavesEstimation::Gradient;
            defaultNewtonIterations = 10;
            defaultGradientIterations = 100;
            break;
        }
        case ELossFunction::MultiClass:
        case ELossFunction::MultiClassOneVsAll: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 10;
            break;
        }
        case ELossFunction::MAE:
        case ELossFunction::MAPE:
        case ELossFunction::Quantile:
        case ELossFunction::GroupQuantile:
        case ELossFunction::MultiQuantile:
        case ELossFunction::LogLinQuantile: {
            defaultEstimationMethod = ELeavesEstimation::Gradient;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::Expectile: {
            CB_ENSURE(lossFunctionConfig.GetLossParamsMap().contains("alpha"), "Param alpha is mandatory for expectile loss");
            defaultNewtonIterations = 5;
            defaultGradientIterations = 10;
            defaultEstimationMethod = ELeavesEstimation::Newton;
            break;
        }
        case ELossFunction::PairLogit: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 10;
            defaultGradientIterations = 40;
            break;
        }
        case ELossFunction::PairLogitPairwise: {
            defaultL2Reg = 5.0;
            if (taskType == ETaskType::CPU) {
                defaultEstimationMethod = ELeavesEstimation::Gradient;
                //CPU doesn't have Newton yet
                defaultGradientIterations = 50;
            } else {
                //newton is a way faster, so default for GPU
                defaultEstimationMethod = ELeavesEstimation::Newton;
                defaultGradientIterations = 5;
            }
            break;
        }
        case ELossFunction::Poisson: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 10;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::Logloss:
        case ELossFunction::CrossEntropy:
        case ELossFunction::MultiLogloss:
        case ELossFunction::MultiCrossEntropy: {
            defaultNewtonIterations = 10;
            defaultGradientIterations = 40;
            defaultEstimationMethod = ELeavesEstimation::Newton;
            break;
        }
        case ELossFunction::YetiRank: {
            defaultL2Reg = 0;
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultGradientIterations = 1;
            defaultNewtonIterations = 1;
            break;
        }
        case ELossFunction::YetiRankPairwise: {
            defaultL2Reg = 0;
            defaultEstimationMethod = (taskType == ETaskType::GPU) ? ELeavesEstimation::Simple : ELeavesEstimation::Gradient;
            defaultGradientIterations = 1;
            defaultNewtonIterations = 1;
            break;
        }
        case ELossFunction::QueryCrossEntropy: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultGradientIterations = 1;
            defaultNewtonIterations = 10;
            defaultL2Reg = 1;
            break;
        }
        case ELossFunction::Huber: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::StochasticFilter: {
            defaultEstimationMethod = ELeavesEstimation::Gradient;
            defaultGradientIterations = 100;
            // doesn't have Newton
            break;
        }
        case ELossFunction::LambdaMart: {
            defaultL2Reg = 0;
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultGradientIterations = 1;
            defaultNewtonIterations = 1;
            break;
        }
        case ELossFunction::StochasticRank: {
            defaultEstimationMethod = ELeavesEstimation::Gradient;
            defaultGradientIterations = 1;
            // doesn't have Newton
            break;
        }
        case ELossFunction::UserPerObjMetric:
        case ELossFunction::UserQuerywiseMetric:
        case ELossFunction::PythonUserDefinedMultiTarget:
        case ELossFunction::PythonUserDefinedPerObject: {
            //skip
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::Tweedie: {
            CB_ENSURE(lossFunctionConfig.GetLossParamsMap().contains("variance_power"), "Param variance_power is mandatory for Tweedie loss");
            defaultEstimationMethod = ELeavesEstimation::Newton;
            if (taskType == ETaskType::CPU) {
                defaultNewtonIterations = 1;
                defaultGradientIterations = 1;
            } else {
                defaultNewtonIterations = 20;
                defaultGradientIterations = 20;
            }
            break;
        }
        case ELossFunction::Focal: {
            CB_ENSURE(lossFunctionConfig.GetLossParamsMap().contains("focal_alpha"), "Param focal_alpha is mandatory for Focal loss");
            CB_ENSURE(lossFunctionConfig.GetLossParamsMap().contains("focal_gamma"), "Param focal_gamma is mandatory for Focal loss");
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::Combination: {
            bool haveDefaults = false;
            IterateOverCombination(
                    lossFunctionConfig.GetLossParamsMap(),
                [&] (const auto& loss, float weight) {
                    if (!haveDefaults) {
                        std::tie(defaultNewtonIterations, defaultGradientIterations, defaultEstimationMethod, defaultL2Reg) =
                            GetEstimationMethodDefaults(taskType, loss);
                        defaultL2Reg *= weight;
                        return;
                    }
                    ui32 newtonIterations;
                    ui32 gradientIterations;
                    ELeavesEstimation method;
                    double l2Reg;
                    std::tie(newtonIterations, gradientIterations, method, l2Reg) = GetEstimationMethodDefaults(taskType, loss);
                    defaultNewtonIterations = Max(newtonIterations, defaultNewtonIterations);
                    defaultGradientIterations = Max(gradientIterations, defaultGradientIterations);
                    if (method != defaultEstimationMethod) {
                        defaultEstimationMethod = ELeavesEstimation::Gradient;
                    }
                    defaultL2Reg += l2Reg * weight;
            });
            break;
        }
        default: {
            CB_ENSURE(false, "Unknown loss function " << lossFunctionConfig.GetLossFunction());
        }
    }
    return std::tie(defaultNewtonIterations, defaultGradientIterations, defaultEstimationMethod, defaultL2Reg);
}

void NCatboostOptions::TCatBoostOptions::SetLeavesEstimationDefault() {
    ui32 defaultNewtonIterations = 1;
    ui32 defaultGradientIterations = 1;
    ELeavesEstimation defaultEstimationMethod = ELeavesEstimation::Newton;
    double defaultL2Reg = 3.0;

    const auto& lossFunctionConfig = LossFunctionDescription.Get();

    std::tie(defaultNewtonIterations, defaultGradientIterations, defaultEstimationMethod, defaultL2Reg)
        = GetEstimationMethodDefaults(GetTaskType(), lossFunctionConfig);

    auto& treeConfig = ObliviousTreeOptions.Get();

    if (lossFunctionConfig.GetLossFunction() == ELossFunction::UserQuerywiseMetric) {
        treeConfig.PairwiseNonDiagReg.SetDefault(0);
    }
    const bool useExact = EqualToOneOf(lossFunctionConfig.GetLossFunction(), ELossFunction::MAE, ELossFunction::MAPE, ELossFunction::Quantile, ELossFunction::GroupQuantile, ELossFunction::MultiQuantile)
            && SystemOptions->IsSingleHost()
            && (
                (TaskType == ETaskType::GPU && BoostingOptions->BoostingType == EBoostingType::Plain)
                || (TaskType == ETaskType::CPU && !BoostingOptions->ApproxOnFullHistory && treeConfig.MonotoneConstraints.Get().empty())
            );

    if (useExact) {
        defaultEstimationMethod = ELeavesEstimation::Exact;
        defaultNewtonIterations = 1;
        defaultGradientIterations = 1;
    }

    ObliviousTreeOptions->L2Reg.SetDefault(defaultL2Reg);

    if (treeConfig.LeavesEstimationMethod.NotSet()) {
        treeConfig.LeavesEstimationMethod.SetDefault(defaultEstimationMethod);
    } else if (treeConfig.LeavesEstimationMethod != defaultEstimationMethod) {
        CB_ENSURE((lossFunctionConfig.GetLossFunction() != ELossFunction::YetiRank),
                  "At the moment, in the YetiRank mode, changing the leaf_estimation_method parameter is prohibited.");
        if (GetTaskType() == ETaskType::CPU) {
            CB_ENSURE(lossFunctionConfig.GetLossFunction() != ELossFunction::PairLogitPairwise,
                "At the moment, in the PairLogitPairwise mode on CPU, changing the leaf_estimation_method parameter is prohibited.");
        }
    }

    if (treeConfig.LeavesEstimationIterations.NotSet()) {
        const ELeavesEstimation method = treeConfig.LeavesEstimationMethod;
        switch (method) {
            case ELeavesEstimation::Newton: {
                treeConfig.LeavesEstimationIterations.SetDefault(defaultNewtonIterations);
                break;
            }
            case ELeavesEstimation::Gradient: {
                treeConfig.LeavesEstimationIterations.SetDefault(defaultGradientIterations);
                break;
            }
            case ELeavesEstimation::Exact:
            case ELeavesEstimation::Simple: {
                treeConfig.LeavesEstimationIterations.SetDefault(1);
                break;
            }
            default: {
                ythrow TCatBoostException() << "Unknown estimation type "
                                            << method;
            }
        }
    }

    if (treeConfig.LeavesEstimationMethod == ELeavesEstimation::Simple) {
        CB_ENSURE(treeConfig.LeavesEstimationIterations == 1u,
                  "Leaves estimation iterations can't be greater, than 1 for Simple leaf-estimation mode");
    }

    if (treeConfig.LeavesEstimationMethod == ELeavesEstimation::Exact) {
        auto loss = lossFunctionConfig.GetLossFunction();
        CB_ENSURE(EqualToOneOf(loss, ELossFunction::Quantile, ELossFunction::GroupQuantile, ELossFunction::MAE, ELossFunction::MAPE, ELossFunction::LogCosh, ELossFunction::MultiQuantile),
            "Exact method is only available for Quantile, GroupQuantile, MultiQuantile, MAE, MAPE and LogCosh loss functions.");
        CB_ENSURE(
            BoostingOptions->BoostingType == EBoostingType::Plain || TaskType == ETaskType::CPU,
            "Exact leaf estimation method don't work with ordered boosting on GPU"
        );
        CB_ENSURE(
            TaskType == ETaskType::GPU || !BoostingOptions->ApproxOnFullHistory,
            "ApproxOnFullHistory option is not available within Exact method on CPU."
        );
    }

    if (treeConfig.L2Reg == 0.0f) {
        treeConfig.L2Reg = 1e-20f;
    }

    if (lossFunctionConfig.GetLossFunction() == ELossFunction::QueryCrossEntropy) {
        CB_ENSURE(treeConfig.LeavesEstimationMethod != ELeavesEstimation::Gradient, "Gradient leaf estimation is not supported for QueryCrossEntropy");
    }

    if (lossFunctionConfig.GetLossFunction() == ELossFunction::StochasticFilter) {
        CB_ENSURE(treeConfig.LeavesEstimationMethod != ELeavesEstimation::Newton, "Newton leaf estimation is not supported for StochasticFilter");
    }
}

void NCatboostOptions::TCatBoostOptions::Load(const NJson::TJsonValue& options) {
    ETaskType currentTaskType = TaskType;
    CheckedLoad(options,
                &TaskType,
                &SystemOptions, &BoostingOptions, &ModelBasedEvalOptions,
                &ObliviousTreeOptions,
                &DataProcessingOptions, &LossFunctionDescription,
                &RandomSeed, &CatFeatureParams,
                &FlatParams, &Metadata, &PoolMetaInfoOptions,
                &LoggingLevel, &IsProfile, &MetricOptions);
    SetNotSpecifiedOptionsToDefaults();
    CB_ENSURE(currentTaskType == GetTaskType(), "Task type in json-config is not equal to one specified for options");
    Validate();
}

void NCatboostOptions::TCatBoostOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, TaskType, SystemOptions, BoostingOptions, ModelBasedEvalOptions, ObliviousTreeOptions,
               DataProcessingOptions, LossFunctionDescription,
               RandomSeed, CatFeatureParams, FlatParams,
               Metadata, PoolMetaInfoOptions, LoggingLevel, IsProfile, MetricOptions);
}

NCatboostOptions::TCtrDescription
NCatboostOptions::TCatBoostOptions::CreateDefaultCounter(EProjectionType projectionType) const {
    if (GetTaskType() == ETaskType::CPU) {
        return TCtrDescription(ECtrType::Counter, GetDefaultPriors(ECtrType::Counter));
    } else {
        CB_ENSURE(GetTaskType() == ETaskType::GPU);
        EBorderSelectionType borderSelectionType;
        switch (projectionType) {
            case EProjectionType::TreeCtr: {
                borderSelectionType = EBorderSelectionType::Median;
                break;
            }
            case EProjectionType::SimpleCtr: {
                borderSelectionType = EBorderSelectionType::MinEntropy;
                break;
            }
            default: {
                ythrow TCatBoostException() << "Unknown projection type " << projectionType;
            }
        }
        return TCtrDescription(ECtrType::FeatureFreq,
                               GetDefaultPriors(ECtrType::FeatureFreq),
                               TBinarizationOptions(borderSelectionType, 15));
    }
}

static Y_NO_INLINE void SetDefaultBinarizationsIfNeeded(
    EProjectionType projectionType,
    TVector<NCatboostOptions::TCtrDescription>* descriptions)
{
    for (auto& description : (*descriptions)) {
        if (description.CtrBinarization.NotSet() && description.Type.Get() == ECtrType::FeatureFreq) {
            description.CtrBinarization->BorderSelectionType = projectionType == EProjectionType::SimpleCtr ? EBorderSelectionType::MinEntropy : EBorderSelectionType::Median;
        }
    }
}

void NCatboostOptions::TCatBoostOptions::SetCtrDefaults() {
    TCatFeatureParams& catFeatureParams = CatFeatureParams.Get();
    ELossFunction lossFunction = LossFunctionDescription->GetLossFunction();

     if (IsGroupwiseMetric(lossFunction)) {
        if (TaskType == ETaskType::GPU) {
            catFeatureParams.CtrHistoryUnit.SetDefault(ECtrHistoryUnit::Group);
        }
    }

    TVector<TCtrDescription> defaultSimpleCtrs;
    TVector<TCtrDescription> defaultTreeCtrs;

    switch (lossFunction) {
        case ELossFunction::PairLogit:
        case ELossFunction::PairLogitPairwise: {
            defaultSimpleCtrs = {CreateDefaultCounter(EProjectionType::SimpleCtr)};
            defaultTreeCtrs = {CreateDefaultCounter(EProjectionType::TreeCtr)};
            break;
        }
        default: {
            defaultSimpleCtrs = {TCtrDescription(ECtrType::Borders, GetDefaultPriors(ECtrType::Borders)), CreateDefaultCounter(EProjectionType::SimpleCtr)};
            defaultTreeCtrs = {TCtrDescription(ECtrType::Borders, GetDefaultPriors(ECtrType::Borders)), CreateDefaultCounter(EProjectionType::TreeCtr)};
        }
    }

    if (catFeatureParams.SimpleCtrs.IsSet() && catFeatureParams.CombinationCtrs.NotSet()) {
        CATBOOST_WARNING_LOG << "Change of simpleCtr will not affect combinations ctrs." << Endl;
    }
    if (catFeatureParams.CombinationCtrs.IsSet() && catFeatureParams.SimpleCtrs.NotSet()) {
        CATBOOST_WARNING_LOG << "Change of combinations ctrs will not affect simple ctrs" << Endl;
    }
    if (catFeatureParams.SimpleCtrs.NotSet()) {
        CatFeatureParams->SimpleCtrs = defaultSimpleCtrs;
    } else {
        SetDefaultPriorsIfNeeded(CatFeatureParams->SimpleCtrs);
        SetDefaultBinarizationsIfNeeded(EProjectionType::SimpleCtr, &CatFeatureParams->SimpleCtrs.Get());
    }
    if (catFeatureParams.CombinationCtrs.NotSet()) {
        CatFeatureParams->CombinationCtrs = defaultTreeCtrs;
    } else {
        SetDefaultPriorsIfNeeded(CatFeatureParams->CombinationCtrs);
        SetDefaultBinarizationsIfNeeded(EProjectionType::TreeCtr, &CatFeatureParams->CombinationCtrs.Get());
    }

    for (auto& perFeatureCtr : CatFeatureParams->PerFeatureCtrs.Get()) {
        SetDefaultBinarizationsIfNeeded(EProjectionType::SimpleCtr, &perFeatureCtr.second);
    }
}


static void ValidateCtrTargetBinarization(
    const NCatboostOptions::TOption<NCatboostOptions::TBinarizationOptions>& ctrTargetBinarization,
    ELossFunction lossFunction)
{
    if (ctrTargetBinarization->BorderCount > 1) {
        CB_ENSURE(EqualToOneOf(lossFunction,
                      ELossFunction::RMSE, ELossFunction::LogCosh, ELossFunction::Quantile, ELossFunction::MultiQuantile,
                      ELossFunction::LogLinQuantile, ELossFunction::Poisson,
                      ELossFunction::MAPE, ELossFunction::MAE, ELossFunction::MultiClass,
                      ELossFunction::MultiRMSE, ELossFunction::MultiRMSEWithMissingValues, ELossFunction::SurvivalAft),
                  "Setting TargetBorderCount is not supported for loss function " << lossFunction);
    }
}


void NCatboostOptions::TCatBoostOptions::ValidateCtr(const TCtrDescription& ctr, ELossFunction lossFunction, bool isTreeCtrs) const {
    ValidateCtrTargetBinarization(ctr.TargetBinarization, lossFunction);
    CB_ENSURE(ctr.GetPriors().size(), "Provide at least one prior for CTR" << ToString(*this));

    const ETaskType taskType = GetTaskType();
    const ECtrType ctrType = ctr.Type;

    if (taskType == ETaskType::GPU) {
        CB_ENSURE(IsSupportedCtrType(ETaskType::GPU, ctrType),
                  "Ctr type " << ctrType << " is not implemented on GPU yet");
        CB_ENSURE(ctr.TargetBinarization.IsDefault(), "Error: GPU doesn't not support target binarization per CTR description currently. Please use ctr_target_border_count option instead");
    } else {
        CB_ENSURE(taskType == ETaskType::CPU);
        CB_ENSURE(IsSupportedCtrType(ETaskType::CPU, ctrType),
                  "Ctr type " << ctrType << " is not implemented on CPU yet");
        CB_ENSURE(ctr.PriorEstimation == EPriorEstimation::No, "Error: CPU doesn't not support prior estimation currently");
    }

    const EBorderSelectionType borderSelectionType = ctr.CtrBinarization->BorderSelectionType;
    if (taskType == ETaskType::CPU) {
        CB_ENSURE(borderSelectionType == EBorderSelectionType::Uniform,
                  "Error: custom ctr binarization is not supported on CPU yet");
    } else {
        CB_ENSURE(taskType == ETaskType::GPU);
        if (isTreeCtrs) {
            EBorderSelectionType borderType = borderSelectionType;
            CB_ENSURE(borderType == EBorderSelectionType::Uniform || borderType == EBorderSelectionType::Median,
                      "Error: GPU supports Median and Uniform combinations-ctr binarization only");

            CB_ENSURE(ctr.PriorEstimation == EPriorEstimation::No, "Error: prior estimation is not available for combinations-ctr");
        } else {
            switch (ctrType) {
                case ECtrType::Borders: {
                    break;
                }
                default: {
                    CB_ENSURE(ctr.PriorEstimation == EPriorEstimation::No, "Error: prior estimation is not available for ctr type " << ctrType);
                }
            }
        }
    }

    if ((ctrType == ECtrType::FeatureFreq) && borderSelectionType == EBorderSelectionType::Uniform) {
        CATBOOST_WARNING_LOG << "Uniform ctr binarization for featureFreq ctr is not good choice. Use MinEntropy for simpleCtrs and Median for combinations-ctrs instead" << Endl;
    }
}

inline double CalculateExpectedSizeModel(const ui32 leafCount, const ui32 iterations) {
    return static_cast<double>(leafCount) * sizeof(double) * iterations * 2;
}

inline TString GetMessageDecreaseDepth(const ui32 leafCount, const ui32 border) {
    return "Each tree in the model is requested to have " + ToString(leafCount) +
           " leaves. Model will weight more than " + ToString(border) + " Gb. Try decreasing depth.";
}

inline TString GetMessageDecreaseNumberIter(const ui32 treeCount, const ui32 border) {
    return "Model with " + ToString(treeCount) + " trees will weight more then " + ToString(border) +
           " Gb. Try decreasing number of iterations";
}

static void ValidateModelSize(const NCatboostOptions::TObliviousTreeLearnerOptions& treeConfig,
                              const NCatboostOptions::TOverfittingDetectorOptions& overfittingDetectorConfig) {
    ui32 leafCount;
    const bool isSymmetricTreeOrDepthwise = (treeConfig.GrowPolicy.Get() == EGrowPolicy::SymmetricTree ||
                                        treeConfig.GrowPolicy.Get() == EGrowPolicy::Depthwise);
    if (isSymmetricTreeOrDepthwise) {
        leafCount = 1 << treeConfig.MaxDepth.Get();
    } else {
        leafCount = treeConfig.MaxLeaves.Get();
    }

    constexpr ui32 OneGb = (1 << 30);
    constexpr ui32 TwoGb = (1 << 31);
    const ui32 treeCount = treeConfig.LeavesEstimationIterations.Get();
    const double totalSizeModel = CalculateExpectedSizeModel(leafCount, treeCount);
    const bool hasOverfittingDetector = (overfittingDetectorConfig.OverfittingDetectorType.Get() != EOverfittingDetectorType::None);
    const bool isModelSizeExceedOneGb = (totalSizeModel > OneGb);
    const bool isModelSizeLowerTwoGb = (hasOverfittingDetector || totalSizeModel < TwoGb);

    if (leafCount > (1 << 12)) {
        CB_ENSURE(isModelSizeLowerTwoGb, GetMessageDecreaseDepth(leafCount, 2));
        if (isModelSizeExceedOneGb) {
            CATBOOST_WARNING_LOG << GetMessageDecreaseDepth(leafCount, 1);
        }
    } else {
        CB_ENSURE(isModelSizeLowerTwoGb, GetMessageDecreaseNumberIter(treeCount, 2));
        if (isModelSizeExceedOneGb) {
            CATBOOST_WARNING_LOG << GetMessageDecreaseNumberIter(treeCount, 1);
        }
    }
}

static void EnsureNewtonIsAvailable(ETaskType taskType, const NCatboostOptions::TLossDescription& lossDescription) {
    const auto lossFunction = lossDescription.GetLossFunction();
    CB_ENSURE(!EqualToOneOf(lossFunction,
        ELossFunction::StochasticFilter,
        ELossFunction::StochasticRank,
        ELossFunction::Quantile,
        ELossFunction::MultiQuantile,
        ELossFunction::MAE,
        ELossFunction::LogLinQuantile,
        ELossFunction::MAPE) &&
        !(taskType == ETaskType::CPU && IsPairwiseScoring(lossFunction)),
        "Newton leaves estimation method is not supported for " << lossFunction << " loss function");
    CB_ENSURE(
        lossFunction != ELossFunction::Lq || NCatboostOptions::GetLqParam(lossDescription) >= 2,
        "Newton leaves estimation method is not supported for Lq loss function with q < 2");
}

void NCatboostOptions::TCatBoostOptions::Validate() const {
    ELossFunction lossFunction = LossFunctionDescription->GetLossFunction();
    {
        const ui32 classesCount = DataProcessingOptions->ClassesCount;
        if (classesCount != 0) {
            CB_ENSURE(IsMultiClassOnlyMetric(lossFunction) || IsUserDefined(lossFunction),
                "classes_count parameter takes effect only with MultiClass/MultiClassOneVsAll and user-defined loss functions");
            CB_ENSURE(classesCount > 1, "classes-count should be at least 2");
        }
        const auto& classWeights = DataProcessingOptions->ClassWeights.Get();
        const EAutoClassWeightsType autoClassWeights = DataProcessingOptions->AutoClassWeights.Get();
        if (!classWeights.empty() || autoClassWeights != EAutoClassWeightsType::None) {
            CB_ENSURE(lossFunction == ELossFunction::Logloss || IsMultiClassOnlyMetric(lossFunction) || IsUserDefined(lossFunction),
                      "class weights takes effect only with Logloss, MultiClass, MultiClassOneVsAll and user-defined loss functions");
            if (!classWeights.empty()) {
                CB_ENSURE(lossFunction != ELossFunction::Logloss || (classWeights.size() == 2),
                          "if loss-function is Logloss, then class weights should be given for 0 and 1 classes");
                CB_ENSURE(classesCount == 0 || classesCount == classWeights.size(), "class weights should be specified for each class in range 0, ... , classes_count - 1");
            }
        }
        const auto& classLabels = DataProcessingOptions->ClassLabels.Get();
        if (!classLabels.empty()) {
            CB_ENSURE(lossFunction != ELossFunction::Logloss || (classLabels.size() == 2),
                          "if loss-function is Logloss, then class labels should be given for 0 and 1 classes");
            CB_ENSURE(classesCount == 0 || classesCount == classLabels.size(), "class labels should be specified for each class in range 0, ... , classes_count - 1");
        }
    }

    DataProcessingOptions->TextProcessingOptions->Validate(IsClassificationObjective(lossFunction));

    ESamplingUnit samplingUnit = ObliviousTreeOptions->BootstrapConfig->GetSamplingUnit();
    if (GetTaskType() == ETaskType::GPU) {
        if (!IsPairwiseScoring(lossFunction)) {
            CB_ENSURE(ObliviousTreeOptions->Rsm.IsDefault(), "Error: rsm on GPU is supported for pairwise modes only");
        } else {
            if (!ObliviousTreeOptions->Rsm.IsDefault()) {
                CATBOOST_WARNING_LOG << "RSM on GPU will work only for non-binary features. Plus current implementation will sample by groups, so this could slightly affect quality in positive or negative way" << Endl;
            }
            CB_ENSURE(ObliviousTreeOptions->MaxDepth.Get() <= 8, "Error: GPU pairwise learning works with tree depth <= 8 only");
        }
        ModelBasedEvalOptions.Get().Validate();

        if (samplingUnit == ESamplingUnit::Group) {
            CB_ENSURE(lossFunction == ELossFunction::YetiRankPairwise,
                      "sampling_unit option on GPU is supported only for loss function YetiRankPairwise");
        }
    }

    CB_ENSURE(!(IsPlainOnlyModeLoss(lossFunction) && (BoostingOptions->BoostingType == EBoostingType::Ordered)),
        "Boosting type should be Plain for loss functions " << lossFunction);

    CB_ENSURE(!(!SystemOptions->IsSingleHost() && (BoostingOptions->BoostingType == EBoostingType::Ordered)),
        "Boosting type should be Plain in distributed mode");

    if (GetTaskType() == ETaskType::CPU) {
        CB_ENSURE(lossFunction != ELossFunction::QueryCrossEntropy,
                  ELossFunction::QueryCrossEntropy << " loss function is not supported for CPU learning");
        CB_ENSURE(
            ObliviousTreeOptions->LeavesEstimationBacktrackingType != ELeavesEstimationStepBacktracking::Armijo,
            "Backtracking type Armijo is supported only on GPU");
    }

    if (ObliviousTreeOptions->LeavesEstimationBacktrackingType != ELeavesEstimationStepBacktracking::No) {
        CB_ENSURE(
            lossFunction != ELossFunction::YetiRank && lossFunction != ELossFunction::YetiRankPairwise,
            "Backtracking is not supported for yetiRank and YetiRankPairwise loss functions");
    }

    ValidateCtrs(CatFeatureParams->SimpleCtrs, lossFunction, false);
    for (const auto& perFeatureCtr : CatFeatureParams->PerFeatureCtrs.Get()) {
        ValidateCtrs(perFeatureCtr.second, lossFunction, false);
    }
    ValidateCtrs(CatFeatureParams->CombinationCtrs, lossFunction, true);
    ValidateCtrTargetBinarization(CatFeatureParams->TargetBinarization, lossFunction);

    CB_ENSURE(Metadata.Get().IsMap(), "metadata should be map");
    for (const auto& keyValue : Metadata.Get().GetMapSafe()) {
        CB_ENSURE(keyValue.second.IsString(), "only string to string metadata dictionary supported");
    }
    CB_ENSURE(!Metadata.Get().Has("params"), "\"params\" key in metadata prohibited");

    // Delete it when MLTOOLS-3572 is implemented.
    if (ShouldBinarizeLabel(LossFunctionDescription->LossFunction.Get())) {
        const TString message = "Metric parameter 'border' isn't supported when target is binarized.";
        CB_ENSURE(!LossFunctionDescription->GetLossParamsMap().contains("border"), message);
        CB_ENSURE(!MetricOptions->EvalMetric->GetLossParamsMap().contains("border"), message);
        CB_ENSURE(!MetricOptions->ObjectiveMetric->GetLossParamsMap().contains("border"), message);
        for (const auto& metric : MetricOptions->CustomMetrics.Get()) {
            CB_ENSURE(!metric.GetLossParamsMap().contains("border"), message);
        }
    }

    // Delete it when MLTOOLS-3612 is implemented.
    CB_ENSURE(!LossFunctionDescription->GetLossParamsMap().contains("use_weights"),
        "Metric parameter 'use_weights' isn't supported for objective function. " <<
        "If weights are present they will necessarily be used in optimization. " <<
        "It cannot be disabled.");

    if (BoostingOptions->BoostFromAverage.Get()) {
        // we may adjust non-set BoostFromAverage in data dependant tuning
        CB_ENSURE(EqualToOneOf(lossFunction, ELossFunction::RMSE, ELossFunction::Logloss,
            ELossFunction::CrossEntropy, ELossFunction::Quantile, ELossFunction::MultiQuantile, ELossFunction::MAE, ELossFunction::MAPE,
            ELossFunction::MultiRMSE, ELossFunction::MultiRMSEWithMissingValues),
            "You can use boost_from_average only for these loss functions now: " <<
            "RMSE, Logloss, CrossEntropy, Quantile, MultiQuantile, MAE, MAPE, MultiRMSE or MultiRMSEWithMissingValues.");
        CB_ENSURE(SystemOptions->IsSingleHost(), "You can use boost_from_average only on single host now.");
    }

    if (GetTaskType() == ETaskType::CPU && !ObliviousTreeOptions->MonotoneConstraints.Get().empty()) {
        // validate monotone constraints
        const auto& monotoneConstraints = ObliviousTreeOptions->MonotoneConstraints.Get();
        CB_ENSURE(!IsPairwiseScoring(lossFunction),
            "Monotone constraints is unsupported for pairwise loss functions."
        );
        CB_ENSURE(!IsMultiClassOnlyMetric(lossFunction),
            "Monotone constraints is unsupported for multiclass."
        );
        CB_ENSURE(!BoostingOptions->ApproxOnFullHistory.Get(),
            "Can't combine approx_on_full_history with monotone constraints."
        );
        CB_ENSURE(
            SystemOptions->IsSingleHost(), "Monotone constraints is unsupported for distributed learning."
        );
        CB_ENSURE(ObliviousTreeOptions->LeavesEstimationMethod != ELeavesEstimation::Exact,
            "Monotone constraints are unsupported for Exact leaves estimation method."
        );
        const THashSet<int> validMonotoneConstraintValues = {-1, 0, 1};
        for (auto [featureIdx, constraint] : monotoneConstraints) {
            CB_ENSURE(validMonotoneConstraintValues.contains(constraint),
                "Monotone constraints should be values in {-1, 0, 1}. Got: " << featureIdx << ":" << constraint);
        }
    }
    ValidateModelSize(ObliviousTreeOptions.Get(), BoostingOptions->OverfittingDetector.Get());

    const ELeavesEstimation leavesEstimation = ObliviousTreeOptions->LeavesEstimationMethod;
    if (leavesEstimation == ELeavesEstimation::Newton) {
        EnsureNewtonIsAvailable(GetTaskType(), LossFunctionDescription);
    }

    if (BoostingOptions->PosteriorSampling.GetUnchecked()) {
        CB_ENSURE(BoostingOptions->Langevin.NotSet() || BoostingOptions->Langevin.Get(),
              "Posterior Sampling requires Langevin boosting.");
        CB_ENSURE(BoostingOptions->DiffusionTemperature.NotSet(),
             "Diffusion Temperature in Posterior Sampling is specified");
        CB_ENSURE(BoostingOptions->ModelShrinkMode.GetUnchecked() == EModelShrinkMode::Constant,
             "Posterior Sampling requires Сonstant Model Shrink Mode");
    }

    if (GetTaskType() == ETaskType::CPU && ObliviousTreeOptions->FeaturePenalties.IsSet()) {
        ValidateFeaturePenaltiesOptions(ObliviousTreeOptions->FeaturePenalties.Get());
    }

    if (ObliviousTreeOptions->GrowPolicy != EGrowPolicy::SymmetricTree) {
        CB_ENSURE(BoostingOptions->BoostingType == EBoostingType::Plain,
            "Ordered boosting is not supported for nonsymmetric trees.");
        CB_ENSURE(SystemOptions->IsSingleHost(),
            "MultiHost training is not supported for nonsymmetric trees.");
        if (TaskType == ETaskType::CPU) {
            CB_ENSURE(!IsPairwiseScoring(lossFunction),
                "Pairwise mode is not supported for nonsymmetric trees on CPU.");
        }
    }

    if (ObliviousTreeOptions->GrowPolicy == EGrowPolicy::Lossguide) {
        CB_ENSURE(ObliviousTreeOptions->SamplingFrequency == ESamplingFrequency::PerTree,
            "PerTreeLevel sampling is not supported for Lossguide grow policy.");
    }
}

void NCatboostOptions::TCatBoostOptions::SetNotSpecifiedOptionsToDefaults() {
    const auto lossFunction = LossFunctionDescription->GetLossFunction();

    // TODO(nikitxskv): Support MVS for GPU.
    auto& boostingType = BoostingOptions->BoostingType;
    TOption<EBootstrapType>& bootstrapType = ObliviousTreeOptions->BootstrapConfig->GetBootstrapType();
    TOption<float>& subsample = ObliviousTreeOptions->BootstrapConfig->GetTakenFraction();
    if (bootstrapType.NotSet()) {
        if (!IsMultiClassOnlyMetric(lossFunction)
            && !IsMultiRegressionObjective(lossFunction)
            && TaskType == ETaskType::CPU
            && ObliviousTreeOptions->BootstrapConfig->GetSamplingUnit() == ESamplingUnit::Object)
        {
            bootstrapType.SetDefault(EBootstrapType::MVS);
        }
    } else {
        if (TaskType == ETaskType::GPU && IsMultiClassOnlyMetric(lossFunction)) {
            CB_ENSURE(bootstrapType != EBootstrapType::MVS, "MVS is not supported for multiclass models on GPU");
        }
    }
    if (subsample.IsSet()) {
        CB_ENSURE(bootstrapType != EBootstrapType::Bayesian, "Error: default bootstrap type (bayesian) doesn't support 'subsample' option");
    } else {
        if (bootstrapType == EBootstrapType::MVS) {
            subsample.SetDefault(0.8);
        }
    }

    if (!IsMultiClassOnlyMetric(lossFunction)
        && !EqualToOneOf(lossFunction, ELossFunction::RMSEWithUncertainty, ELossFunction::MultiLogloss, ELossFunction::MultiCrossEntropy)
        && TaskType == ETaskType::GPU && !boostingType.IsSet()
    ) {
        boostingType.SetDefault(EBoostingType::Ordered);
    }

    if (IsPlainOnlyModeLoss(lossFunction)) {
        boostingType.SetDefault(EBoostingType::Plain);
        CB_ENSURE(boostingType.IsDefault(), "Boosting type should be plain for " << lossFunction);
    }

    if (boostingType.NotSet() && !SystemOptions->IsSingleHost()) {
        boostingType.SetDefault(EBoostingType::Plain);
    }

    switch (lossFunction) {
        case ELossFunction::QueryCrossEntropy:
        case ELossFunction::YetiRankPairwise:
        case ELossFunction::PairLogitPairwise: {
            ObliviousTreeOptions->RandomStrength.SetDefault(0.0);
            DataProcessingOptions->FloatFeaturesBinarization->BorderCount.SetDefault(32);

            if (ObliviousTreeOptions->BootstrapConfig->GetBaggingTemperature().IsSet()) {
                CB_ENSURE(ObliviousTreeOptions->BootstrapConfig->GetTakenFraction().NotSet(), "Error: can't use bagging temperature and subsample at the same time");
                //fallback to bayesian bootstrap
                if (bootstrapType.NotSet()) {
                    CATBOOST_WARNING_LOG << "Implicitly assume bayesian bootstrap, learning could be slower" << Endl;
                }
            } else {
                bootstrapType.SetDefault(EBootstrapType::Bernoulli);
                ObliviousTreeOptions->BootstrapConfig->GetTakenFraction().SetDefault(0.5);
            }
            break;
        }
        case ELossFunction::PythonUserDefinedMultiTarget:
        case ELossFunction::PythonUserDefinedPerObject: {
            ObliviousTreeOptions->LeavesEstimationBacktrackingType.SetDefault(ELeavesEstimationStepBacktracking::No);
            break;
        }
        default: {
            //skip
            break;
        }
    }

    switch (lossFunction) {
        case ELossFunction::YetiRank:
        case ELossFunction::YetiRankPairwise: {
            NCatboostOptions::TLossDescription lossDescription;
            lossDescription.Load(LossDescriptionToJson("PFound"));
            MetricOptions->ObjectiveMetric.Set(lossDescription);
            ObliviousTreeOptions->LeavesEstimationBacktrackingType.SetDefault(ELeavesEstimationStepBacktracking::No);
            break;
        }
        case ELossFunction::PairLogit:
        case ELossFunction::PairLogitPairwise: {
            NCatboostOptions::TLossDescription lossDescription = LossFunctionDescription->CloneWithLossFunction(ELossFunction::PairLogit);
            MetricOptions->ObjectiveMetric.Set(lossDescription);
            break;
        }
        case ELossFunction::StochasticFilter: {
            NCatboostOptions::TLossDescription lossDescription;
            lossDescription.LossFunction.Set(ELossFunction::FilteredDCG);
            MetricOptions->ObjectiveMetric.Set(lossDescription);
            break;
        }
        case ELossFunction::LambdaMart: {
            NCatboostOptions::TLossDescription lossDescription;
            const auto& lossParams = LossFunctionDescription->GetLossParamsMap();
            ELossFunction targetMetric = lossParams.contains("metric") ? FromString<ELossFunction>(lossParams.at("metric")) : ELossFunction::NDCG;
            TVector<std::pair<TString, TString>> metricParams;
            TSet<TString> validParams;
            switch (targetMetric) {
                case ELossFunction::DCG:
                case ELossFunction::NDCG:
                    validParams = {"top", "type", "denominator", "hints"};
                    break;
                case ELossFunction::MRR:
                    validParams = {"hints"};
                    break;
                case ELossFunction::ERR:
                    validParams = {"hints"};
                    break;
                case ELossFunction::MAP:
                    validParams = {"hints"};
                    break;
                default:
                    CB_ENSURE(false, "LambdaMart does not support target_metric " << targetMetric);
            }
            for (const auto& key : LossFunctionDescription->GetLossParamKeysOrdered()) {
                if (!validParams.contains(key)) {
                    continue;
                }
                metricParams.emplace_back(key, lossParams.at(key));
            }
            lossDescription.LossParams.Set(TLossParams::FromVector(metricParams));
            lossDescription.LossFunction.Set(targetMetric);
            MetricOptions->ObjectiveMetric.Set(lossDescription);
            ObliviousTreeOptions->LeavesEstimationBacktrackingType.SetDefault(ELeavesEstimationStepBacktracking::No);
            break;
        }
        case ELossFunction::StochasticRank: {
            NCatboostOptions::TLossDescription lossDescription;
            const auto& lossParams = LossFunctionDescription->GetLossParamsMap();
            CB_ENSURE(lossParams.contains("metric"), "StochasticRank requires metric param");
            ELossFunction targetMetric = FromString<ELossFunction>(lossParams.at("metric"));
            TVector<std::pair<TString, TString>> metricParams;
            TSet<TString> validParams;
            switch (targetMetric) {
                case ELossFunction::DCG:
                case ELossFunction::NDCG:
                    validParams = {"top", "type", "denominator", "hints"};
                    break;
                case ELossFunction::PFound:
                    validParams = {"top", "decay", "hints"};
                    break;
                case ELossFunction::FilteredDCG:
                    validParams = {"type", "denominator", "hints"};
                    break;
                case ELossFunction::MRR:
                    validParams = {"hints"};
                    break;
                case ELossFunction::ERR:
                    validParams = {"hints"};
                    break;
                default:
                    CB_ENSURE(false, "StochasticRank does not support target_metric " << targetMetric);
            }
            for (const auto& key : LossFunctionDescription->GetLossParamKeysOrdered()) {
                if (!validParams.contains(key)) {
                    continue;
                }
                metricParams.emplace_back(key, lossParams.at(key));
            }
            lossDescription.LossParams.Set(TLossParams::FromVector(metricParams));
            lossDescription.LossFunction.Set(targetMetric);
            MetricOptions->ObjectiveMetric.Set(lossDescription);
            ObliviousTreeOptions->LeavesEstimationBacktrackingType.SetDefault(ELeavesEstimationStepBacktracking::No);
            break;
        }
        default: {
            MetricOptions->ObjectiveMetric.Set(LossFunctionDescription.Get());
            break;
        }
    }

    if (TaskType == ETaskType::GPU) {
        if (IsGpuPlainDocParallelOnlyMode(lossFunction) ||
            ObliviousTreeOptions->GrowPolicy != EGrowPolicy::SymmetricTree) {
            //lets check correctness first
            BoostingOptions->DataPartitionType.SetDefault(EDataPartitionType::DocParallel);
            boostingType.SetDefault(EBoostingType::Plain);

            TString option;
            if (IsGpuPlainDocParallelOnlyMode(lossFunction)) {
                option = "loss " + ToString(lossFunction);
            } else {
                option = "grow policy " + ToString(ObliviousTreeOptions->GrowPolicy.Get());
            }

            CB_ENSURE(BoostingOptions->DataPartitionType == EDataPartitionType::DocParallel,
                    "On GPU " << option << " is implemented in doc-parallel mode only");
            CB_ENSURE(boostingType == EBoostingType::Plain,
                    "On GPU " << option << " can't be used with ordered boosting");

            //now ensure automatic estimations won't override this
            boostingType = EBoostingType::Plain;
            BoostingOptions->DataPartitionType = EDataPartitionType::DocParallel;
        }

        if (IsPlainOnlyModeScoreFunction(ObliviousTreeOptions->ScoreFunction)) {
            boostingType.SetDefault(EBoostingType::Plain);
            CB_ENSURE(boostingType == EBoostingType::Plain,
                    "Score function " << ObliviousTreeOptions->ScoreFunction.Get() << " can't be used with ordered boosting");
            boostingType = EBoostingType::Plain;
        }

        if (ObliviousTreeOptions->GrowPolicy == EGrowPolicy::Lossguide) {
            const bool useL2ScoreFunction = EqualToOneOf(
                lossFunction,
                ELossFunction::MultiClass,
                ELossFunction::MultiClassOneVsAll,
                ELossFunction::RMSEWithUncertainty);
            if (useL2ScoreFunction) {
                ObliviousTreeOptions->ScoreFunction.SetDefault(EScoreFunction::L2);
            } else {
                ObliviousTreeOptions->ScoreFunction.SetDefault(EScoreFunction::NewtonL2);
            }
        }
    }
    if (ObliviousTreeOptions->GrowPolicy != EGrowPolicy::Lossguide) {
        const ui32 maxLeaves = 1u << ObliviousTreeOptions->MaxDepth.Get();
        if (ObliviousTreeOptions->MaxLeaves.IsDefault()) {
            ObliviousTreeOptions->MaxLeaves.SetDefault(maxLeaves);
        } else {
            CB_ENSURE(ObliviousTreeOptions->MaxLeaves == maxLeaves,
                        "max_leaves option works only with lossguide tree growing");
        }
    }
    if (TaskType == ETaskType::CPU) {
        if (BoostingOptions->PosteriorSampling.GetUnchecked()) {
            BoostingOptions->Langevin.SetDefault(true);
        }

        if (BoostingOptions->DiffusionTemperature > 0.0f && BoostingOptions->Langevin.NotSet()) {
            BoostingOptions->Langevin.SetDefault(true);
        }

        auto& shrinkRate = BoostingOptions->ModelShrinkRate;
        if (BoostingOptions->Langevin) {
            if (BoostingOptions->DiffusionTemperature.NotSet()) {
                BoostingOptions->DiffusionTemperature.SetDefault(1e4);
            }
            if (shrinkRate.NotSet()) {
                if (BoostingOptions->ModelShrinkMode == EModelShrinkMode::Constant) {
                    shrinkRate = 0.001;
                } else {
                    shrinkRate = 0.01;
                }
            }
            if (ObliviousTreeOptions->LeavesEstimationBacktrackingType.NotSet()) {
                ObliviousTreeOptions->LeavesEstimationBacktrackingType.SetDefault(ELeavesEstimationStepBacktracking::No);
            }
        }

        if (!ObliviousTreeOptions->MonotoneConstraints->empty() &&
            shrinkRate.NotSet())
        {
            if (BoostingOptions->ModelShrinkMode == EModelShrinkMode::Constant) {
                shrinkRate = 0.01;
            } else {
                shrinkRate = 0.2;
            }
        }
    }

    SetLeavesEstimationDefault();
    SetCtrDefaults();

    if (DataProcessingOptions->HasTimeFlag) {
        BoostingOptions->PermutationCount = 1;
    }

    if (CatFeatureParams->MaxTensorComplexity.NotSet() && IsSmallIterationCount(BoostingOptions->IterationCount)) {
        CatFeatureParams->MaxTensorComplexity = 1;
    }

    DataProcessingOptions->TextProcessingOptions->SetDefault(IsClassificationObjective(lossFunction));
}

static TVector<ui32> GetIndices(const NJson::TJsonValue& catBoostJsonOptions, const TString& key, const TString& subKey) {
    CB_ENSURE(catBoostJsonOptions.Has(key), "Invalid option section '" << key << "'");
    auto& group = catBoostJsonOptions[key];
    if (group.IsMap() && group.Has(subKey)) {
        auto& value = group[subKey];
        if (value.IsArray()) {
            try {
                TVector<ui32> result;
                TJsonFieldHelper<TVector<ui32>>::Read(value, &result);
                return result;
            } catch (NJson::TJsonException) {
                TVector<TVector<ui32>> indexSets;
                TJsonFieldHelper<TVector<TVector<ui32>>>::Read(value, &indexSets);
                TVector<ui32> result;
                for (const auto& indexSet : indexSets) {
                    result.insert(result.end(), indexSet.begin(), indexSet.end());
                }
                Sort(result.begin(), result.end());
                return result;
            }
        }
    }
    return {};
}

TVector<ui32> GetOptionIgnoredFeatures(const NJson::TJsonValue& catBoostJsonOptions) {
    return GetIndices(catBoostJsonOptions, "data_processing_options", "ignored_features");
}

TVector<ui32> GetOptionFeaturesToEvaluate(const NJson::TJsonValue& catBoostJsonOptions) {
    if (NCatboostOptions::GetTaskType(catBoostJsonOptions) == ETaskType::CPU) {
        return {};
    } else {
        return GetIndices(catBoostJsonOptions, "model_based_eval_options", "features_to_evaluate");
    }
}

ETaskType NCatboostOptions::GetTaskType(const NJson::TJsonValue& source) {
    TOption<ETaskType> taskType("task_type", ETaskType::CPU);
    TJsonFieldHelper<decltype(taskType)>::Read(source, &taskType);
    return taskType.Get();
}

ui32 NCatboostOptions::GetThreadCount(const NJson::TJsonValue& source) {
    TOption<ui32> threadCount("thread_count", NSystemInfo::CachedNumberOfCpus());
    TJsonFieldHelper<decltype(threadCount)>::Read(source["system_options"], &threadCount);
    return threadCount.Get();
}

NCatboostOptions::TCatBoostOptions NCatboostOptions::LoadOptions(const NJson::TJsonValue& source) {
    //little hack. JSON parsing needs to known device_type
    TCatBoostOptions options(GetTaskType(source));
    options.Load(source);
    return options;
}

static bool IsFullBaseline(const NJson::TJsonValue& source) {
    NCatboostOptions::TOption<bool> isFullBaseline(
        "use_evaluated_features_in_baseline_model",
        false
    );
    TJsonFieldHelper<decltype(isFullBaseline)>::Read(
        source["model_based_eval_options"],
        &isFullBaseline
    );
    return isFullBaseline.Get();
}

static TSet<ui32> GetMaybeIgnoredFeatures(const NJson::TJsonValue& params) {
    const auto ignoredFeatures = GetOptionIgnoredFeatures(params);
    const auto featuresToEvaluate = GetOptionFeaturesToEvaluate(params);
    TSet<ui32> result;
    result.insert(ignoredFeatures.begin(), ignoredFeatures.end());
    const bool isFullBaseline = IsFullBaseline(params);
    if (!isFullBaseline) {
        result.insert(featuresToEvaluate.begin(), featuresToEvaluate.end());
    }
    return result;
}

bool NCatboostOptions::IsParamsCompatible(
    const TStringBuf firstSerializedParams,
    const TStringBuf secondSerializedParams)
{
    //TODO:(noxoomo, nikitxskv): i don't think this way of checking compatible is good. We should parse params and comprare fields that are essential, not all
    const TStringBuf paramsToIgnore[] = {
        "system_options",
        "flat_params",
        "metadata",
        "model_based_eval_options"
    };
    const TStringBuf dataProcessingParamsToIgnore[] = {
        "ignored_features"
    };
    const TStringBuf boostingParamsToIgnore[] = {
        "iterations",
        "learning_rate",
    };
    NJson::TJsonValue firstParams, secondParams;
    ReadJsonTree(firstSerializedParams, &firstParams);
    ReadJsonTree(secondSerializedParams, &secondParams);

    // Check ignored and MBE features
    const bool isSameMaybeIgnoredFeatures = GetMaybeIgnoredFeatures(firstParams) == GetMaybeIgnoredFeatures(secondParams);

    for (const auto& paramName : paramsToIgnore) {
        firstParams.EraseValue(paramName);
        secondParams.EraseValue(paramName);
    }
    for (const auto& paramName : dataProcessingParamsToIgnore) {
        firstParams["data_processing_options"].EraseValue(paramName);
        secondParams["data_processing_options"].EraseValue(paramName);
    }
    for (const auto& paramName : boostingParamsToIgnore) {
        firstParams["boosting_options"].EraseValue(paramName);
        secondParams["boosting_options"].EraseValue(paramName);
    }
    return isSameMaybeIgnoredFeatures && firstParams == secondParams;
}

NCatboostOptions::TCatBoostOptions::TCatBoostOptions(ETaskType taskType)
    : SystemOptions("system_options", TSystemOptions(taskType))
    , BoostingOptions("boosting_options", TBoostingOptions(taskType))
    , ObliviousTreeOptions("tree_learner_options", TObliviousTreeLearnerOptions(taskType))
    , DataProcessingOptions("data_processing_options", TDataProcessingOptions(taskType))
    , LossFunctionDescription("loss_function", TLossDescription())
    , CatFeatureParams("cat_feature_params", TCatFeatureParams(taskType))
    , FlatParams("flat_params", NJson::TJsonValue(NJson::JSON_MAP))
    , Metadata("metadata", NJson::TJsonValue(NJson::JSON_MAP))
    , PoolMetaInfoOptions("pool_metainfo_options", TPoolMetaInfoOptions())
    , RandomSeed("random_seed", 0)
    , LoggingLevel("logging_level", ELoggingLevel::Verbose)
    , IsProfile("detailed_profile", false)
    , MetricOptions("metrics", TMetricOptions())
    , ModelBasedEvalOptions("model_based_eval_options", TModelBasedEvalOptions(taskType), taskType)
    , TaskType("task_type", taskType) {
}

bool NCatboostOptions::TCatBoostOptions::operator==(const TCatBoostOptions& rhs) const {
    return std::tie(SystemOptions, BoostingOptions, ModelBasedEvalOptions, ObliviousTreeOptions,  DataProcessingOptions,
            LossFunctionDescription, CatFeatureParams, RandomSeed, LoggingLevel,
            IsProfile, MetricOptions, FlatParams, Metadata, PoolMetaInfoOptions) ==
        std::tie(rhs.SystemOptions, rhs.BoostingOptions, rhs.ModelBasedEvalOptions, rhs.ObliviousTreeOptions,
                rhs.DataProcessingOptions, rhs.LossFunctionDescription, rhs.CatFeatureParams,
                rhs.RandomSeed, rhs.LoggingLevel,
                rhs.IsProfile, rhs.MetricOptions, rhs.FlatParams, rhs.Metadata, rhs.PoolMetaInfoOptions);
}

bool NCatboostOptions::TCatBoostOptions::operator!=(const TCatBoostOptions& rhs) const {
    return !(rhs == *this);
}

ETaskType NCatboostOptions::TCatBoostOptions::GetTaskType() const {
    return TaskType.Get();
}

void NCatboostOptions::TCatBoostOptions::SetDefaultPriorsIfNeeded(TVector<TCtrDescription>& ctrs) const {
    for (auto& ctr : ctrs) {
        if (!ctr.ArePriorsSet()) {
            ctr.SetPriors(GetDefaultPriors(ctr.Type));
        }
    }
}

void NCatboostOptions::TCatBoostOptions::ValidateCtrs(
    const TVector<TCtrDescription>& ctrDescription,
    ELossFunction lossFunction,
    bool isTreeCtrs) const
{
    for (const auto& ctr : ctrDescription) {
        ValidateCtr(ctr, lossFunction, isTreeCtrs);
    }
}
