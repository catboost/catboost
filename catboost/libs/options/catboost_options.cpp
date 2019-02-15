#include "catboost_options.h"
#include "restrictions.h"

#include <library/json/json_reader.h>

#include <util/string/cast.h>

template <>
void Out<NCatboostOptions::TCatBoostOptions>(IOutputStream& out, const NCatboostOptions::TCatBoostOptions& options) {
    NJson::TJsonValue json;
    options.Save(&json);
    out << ToString(json);
}

template <>
inline TCatboostOptions FromString<NCatboostOptions::TCatBoostOptions>(const TString& str) {
    NJson::TJsonValue json;
    NJson::ReadJsonTree(str, &json, true);
    return NCatboostOptions::LoadOptions(json);
}

void NCatboostOptions::TCatBoostOptions::SetLeavesEstimationDefault() {
    const auto& lossFunctionConfig = LossFunctionDescription.Get();

    auto& treeConfig = ObliviousTreeOptions.Get();
    ui32 defaultNewtonIterations = 1;
    ui32 defaultGradientIterations = 1;
    ELeavesEstimation defaultEstimationMethod = ELeavesEstimation::Newton;

    double defaultL2Reg = 3.0;

    switch (lossFunctionConfig.GetLossFunction()) {
        case ELossFunction::RMSE: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::Lq: {
            CB_ENSURE(lossFunctionConfig.GetLossParams().contains("q"), "Param q is mandatory for Lq loss");
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
        case ELossFunction::Quantile:
        case ELossFunction::MAE:
        case ELossFunction::LogLinQuantile:
        case ELossFunction::MAPE: {
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            defaultEstimationMethod = ELeavesEstimation::Gradient;
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
            if (TaskType == ETaskType::CPU) {
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
            defaultEstimationMethod = ELeavesEstimation::Gradient;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::Logloss:
        case ELossFunction::CrossEntropy: {
            defaultNewtonIterations = 10;
            defaultGradientIterations = 40;
            defaultEstimationMethod = ELeavesEstimation::Newton;
            break;
        }
        case ELossFunction::YetiRank: {
            defaultL2Reg = 0;
            defaultEstimationMethod = (GetTaskType() == ETaskType::GPU) ? ELeavesEstimation::Newton : ELeavesEstimation::Gradient;
            defaultGradientIterations = 1;
            defaultNewtonIterations = 1;
            break;
        }
        case ELossFunction::YetiRankPairwise: {
            defaultL2Reg = 0;
            defaultEstimationMethod = (GetTaskType() == ETaskType::GPU) ? ELeavesEstimation::Simple : ELeavesEstimation::Gradient;
            defaultGradientIterations = 1;
            defaultNewtonIterations = 1;
            break;
        }
        case ELossFunction::QueryCrossEntropy: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultGradientIterations = 1;
            defaultNewtonIterations = 10;
            treeConfig.PairwiseNonDiagReg.SetDefault(0);
            defaultL2Reg = 1;
            break;
        }
        case ELossFunction::UserPerObjMetric:
        case ELossFunction::UserQuerywiseMetric:
        case ELossFunction::Custom: {
            //skip
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        default: {
            CB_ENSURE(false, "Unknown loss function " << lossFunctionConfig.GetLossFunction());
        }
    }
    ObliviousTreeOptions->L2Reg.SetDefault(defaultL2Reg);

    if (treeConfig.LeavesEstimationMethod.NotSet()) {
        treeConfig.LeavesEstimationMethod.SetDefault(defaultEstimationMethod);
    } else if (treeConfig.LeavesEstimationMethod != defaultEstimationMethod) {
        CB_ENSURE((lossFunctionConfig.GetLossFunction() != ELossFunction::YetiRank ||
                   lossFunctionConfig.GetLossFunction() != ELossFunction::YetiRankPairwise),
                  "At the moment, in the YetiRank and YetiRankPairwise mode, changing the leaf_estimation_method parameter is prohibited.");
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

    if (treeConfig.L2Reg == 0.0f) {
        treeConfig.L2Reg = 1e-20f;
    }

    if (lossFunctionConfig.GetLossFunction() == ELossFunction::QueryCrossEntropy) {
        CB_ENSURE(treeConfig.LeavesEstimationMethod != ELeavesEstimation::Gradient, "Gradient leaf estimation is not supported for QueryCrossEntropy");
    }
}

void NCatboostOptions::TCatBoostOptions::Load(const NJson::TJsonValue& options) {
    ETaskType currentTaskType = TaskType;
    CheckedLoad(options,
                &TaskType,
                &SystemOptions, &BoostingOptions,
                &ObliviousTreeOptions,
                &DataProcessingOptions, &LossFunctionDescription,
                &RandomSeed, &CatFeatureParams,
                &FlatParams, &Metadata, &LoggingLevel,
                &IsProfile, &MetricOptions);
    SetNotSpecifiedOptionsToDefaults();
    CB_ENSURE(currentTaskType == GetTaskType(), "Task type in json-config is not equal to one specified for options");
    Validate();
}

void NCatboostOptions::TCatBoostOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, TaskType, SystemOptions, BoostingOptions, ObliviousTreeOptions,
               DataProcessingOptions, LossFunctionDescription,
               RandomSeed, CatFeatureParams, FlatParams, Metadata, LoggingLevel, IsProfile, MetricOptions);
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
        CB_ENSURE(lossFunction == ELossFunction::RMSE || lossFunction == ELossFunction::Quantile ||
                      lossFunction == ELossFunction::LogLinQuantile || lossFunction == ELossFunction::Poisson ||
                      lossFunction == ELossFunction::MAPE || lossFunction == ELossFunction::MAE || lossFunction == ELossFunction::MultiClass,
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

void NCatboostOptions::TCatBoostOptions::Validate() const {
    SystemOptions.Get().Validate();
    BoostingOptions.Get().Validate();
    ObliviousTreeOptions.Get().Validate();

    ELossFunction lossFunction = LossFunctionDescription->GetLossFunction();
    {
        const ui32 classesCount = DataProcessingOptions->ClassesCount;
        if (classesCount != 0 ) {
            CB_ENSURE(IsMultiClassMetric(lossFunction), "classes_count parameter takes effect only with MultiClass/MultiClassOneVsAll loss functions");
            CB_ENSURE(classesCount > 1, "classes-count should be at least 2");
        }
        const auto& classWeights = DataProcessingOptions->ClassWeights.Get();
        if (!classWeights.empty()) {
            CB_ENSURE(lossFunction == ELossFunction::Logloss || IsMultiClassMetric(lossFunction),
                      "class weights takes effect only with Logloss, MultiClass and MultiClassOneVsAll loss functions");
            CB_ENSURE(IsMultiClassMetric(lossFunction) || (classWeights.size() == 2),
                      "if loss-function is Logloss, then class weights should be given for 0 and 1 classes");
            CB_ENSURE(classesCount == 0 || classesCount == classWeights.size(), "class weights should be specified for each class in range 0, ... , classes_count - 1");
        }
    }

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

        if (samplingUnit == ESamplingUnit::Group) {
            CB_ENSURE(lossFunction == ELossFunction::YetiRankPairwise,
                      "sampling_unit option on GPU is supported only for loss function YetiRankPairwise");
        }
    }

    ELeavesEstimation leavesEstimation = ObliviousTreeOptions->LeavesEstimationMethod;
    if (lossFunction == ELossFunction::Quantile ||
        lossFunction == ELossFunction::MAE ||
        lossFunction == ELossFunction::LogLinQuantile ||
        lossFunction == ELossFunction::MAPE)
    {
        CB_ENSURE(leavesEstimation != ELeavesEstimation::Newton,
                  "Newton leave estimation method is not supported for " << lossFunction << " loss function");
        CB_ENSURE(ObliviousTreeOptions->LeavesEstimationIterations == 1U,
                  "gradient_iterations should equals 1 for this mode");
    }

    CB_ENSURE(!(IsPlainOnlyModeLoss(lossFunction) && (BoostingOptions->BoostingType == EBoostingType::Ordered)),
        "Boosting type should be Plain for loss functions " << lossFunction);

    if (GetTaskType() == ETaskType::CPU) {
        CB_ENSURE(!(IsPairwiseScoring(lossFunction) && leavesEstimation == ELeavesEstimation::Newton),
                  "This leaf estimation method is not supported for querywise error for CPU learning");
        CB_ENSURE(
            ObliviousTreeOptions->LeavesEstimationBacktrackingType != ELeavesEstimationStepBacktracking::Armijo,
            "Backtracking type Armijo is supported only on GPU");
        CB_ENSURE(
            LossFunctionDescription->GetLossFunction() != ELossFunction::Custom
            || ObliviousTreeOptions->LeavesEstimationBacktrackingType == ELeavesEstimationStepBacktracking::None,
            "Backtracking is not supported for custom loss functions on CPU");
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
}

void NCatboostOptions::TCatBoostOptions::SetNotSpecifiedOptionsToDefaults() {
    if (IsPlainOnlyModeLoss(LossFunctionDescription->GetLossFunction())) {
        BoostingOptions->BoostingType.SetDefault(EBoostingType::Plain);
        CB_ENSURE(BoostingOptions->BoostingType.IsDefault(), "Boosting type should be plain for " << LossFunctionDescription->GetLossFunction());
    }

    switch (LossFunctionDescription->GetLossFunction()) {
        case ELossFunction::QueryCrossEntropy:
        case ELossFunction::YetiRankPairwise:
        case ELossFunction::PairLogitPairwise: {
            ObliviousTreeOptions->RandomStrength.SetDefault(0.0);
            DataProcessingOptions->FloatFeaturesBinarization->BorderCount.SetDefault(32);

            if (ObliviousTreeOptions->BootstrapConfig->GetBaggingTemperature().IsSet()) {
                CB_ENSURE(ObliviousTreeOptions->BootstrapConfig->GetTakenFraction().NotSet(), "Error: can't use bagging temperature and subsample at the same time");
                //fallback to bayesian bootstrap
                if (ObliviousTreeOptions->BootstrapConfig->GetBootstrapType().NotSet()) {
                    CATBOOST_WARNING_LOG << "Implicitly assume bayesian bootstrap, learning could be slower" << Endl;
                }
            } else {
                ObliviousTreeOptions->BootstrapConfig->GetBootstrapType().SetDefault(EBootstrapType::Bernoulli);
                ObliviousTreeOptions->BootstrapConfig->GetTakenFraction().SetDefault(0.5);
            }
            break;
        }
        case ELossFunction::Custom: {
            ObliviousTreeOptions->LeavesEstimationBacktrackingType.SetDefault(ELeavesEstimationStepBacktracking::None);
            break;
        }
        default: {
            //skip
            break;
        }
    }
    if (TaskType == ETaskType::GPU) {
        if (IsGpuPlainDocParallelOnlyMode(LossFunctionDescription->GetLossFunction())) {
            //lets check correctness first
            BoostingOptions->DataPartitionType.SetDefault(EDataPartitionType::DocParallel);
            BoostingOptions->BoostingType.SetDefault(EBoostingType::Plain);
            CB_ENSURE(BoostingOptions->DataPartitionType == EDataPartitionType::DocParallel, "Loss " << LossFunctionDescription->GetLossFunction() << " on GPU is implemented in doc-parallel mode only");
            CB_ENSURE(BoostingOptions->BoostingType == EBoostingType::Plain, "Loss " << LossFunctionDescription->GetLossFunction() << " on GPU can't be used for ordered boosting");
            //now ensure automatic estimations won't override this
            BoostingOptions->BoostingType = EBoostingType::Plain;
            BoostingOptions->DataPartitionType = EDataPartitionType::DocParallel;
        }

        if (ObliviousTreeOptions->GrowingPolicy == EGrowingPolicy::Lossguide) {
            ObliviousTreeOptions->MaxDepth.SetDefault(16);
        }
        if (ObliviousTreeOptions->MaxLeavesCount.IsDefault() && ObliviousTreeOptions->GrowingPolicy != EGrowingPolicy::Lossguide) {
            const ui32 maxLeaves = 1u << ObliviousTreeOptions->MaxDepth.Get();
            ObliviousTreeOptions->MaxLeavesCount.SetDefault(maxLeaves);

            if (ObliviousTreeOptions->GrowingPolicy != EGrowingPolicy::Lossguide) {
                CB_ENSURE(ObliviousTreeOptions->MaxLeavesCount == maxLeaves,
                          "max_leaves_count options works only with lossguide tree growing");
            }
        }
    }

    SetLeavesEstimationDefault();
    SetCtrDefaults();

    if (DataProcessingOptions->HasTimeFlag) {
        BoostingOptions->PermutationCount = 1;
    }
}

TVector<ui32> GetOptionIgnoredFeatures(const NJson::TJsonValue& catBoostJsonOptions) {
    TVector<ui32> result;
    auto& dataProcessingOptions = catBoostJsonOptions["data_processing_options"];
    if (dataProcessingOptions.IsMap()) {
        auto& ignoredFeatures = dataProcessingOptions["ignored_features"];
        if (ignoredFeatures.IsArray()) {
            NCatboostOptions::TJsonFieldHelper<TVector<ui32>>::Read(ignoredFeatures, &result);
        }
    }
    return result;
}

ETaskType NCatboostOptions::GetTaskType(const NJson::TJsonValue& source) {
    TOption<ETaskType> taskType("task_type", ETaskType::CPU);
    TJsonFieldHelper<decltype(taskType)>::Read(source, &taskType);
    return taskType.Get();
}

NCatboostOptions::TCatBoostOptions NCatboostOptions::LoadOptions(const NJson::TJsonValue& source) {
    //little hack. JSON parsing needs to known device_type
    TOption<ETaskType> taskType("task_type", ETaskType::CPU);
    TJsonFieldHelper<decltype(taskType)>::Read(source, &taskType);
    TCatBoostOptions options(taskType.Get());
    options.Load(source);
    return options;
}

bool NCatboostOptions::IsParamsCompatible(
    const TStringBuf firstSerializedParams,
    const TStringBuf secondSerializedParams)
{
    //TODO:(noxoomo, nikitxskv): i don't think this way of checking compatible is good. We should parse params and comprare fields that are essential, not all
    const TStringBuf paramsToIgnore[] = {
        "system_options",
        "flat_params",
        "metadata"
    };
    const TStringBuf boostingParamsToIgnore[] = {
        "iterations",
        "learning_rate",
    };
    NJson::TJsonValue firstParams, secondParams;
    ReadJsonTree(firstSerializedParams, &firstParams);
    ReadJsonTree(secondSerializedParams, &secondParams);

    for (const auto& paramName : paramsToIgnore) {
        firstParams.EraseValue(paramName);
        secondParams.EraseValue(paramName);
    }
    for (const auto& paramName : boostingParamsToIgnore) {
        firstParams["boosting_options"].EraseValue(paramName);
        secondParams["boosting_options"].EraseValue(paramName);
    }
    return firstParams == secondParams;
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
    , RandomSeed("random_seed", 0)
    , LoggingLevel("logging_level", ELoggingLevel::Verbose)
    , IsProfile("detailed_profile", false)
    , MetricOptions("metrics", TMetricOptions())
    , TaskType("task_type", taskType) {
}

bool NCatboostOptions::TCatBoostOptions::operator==(const TCatBoostOptions& rhs) const {
    return std::tie(SystemOptions, BoostingOptions, ObliviousTreeOptions,  DataProcessingOptions,
            LossFunctionDescription, CatFeatureParams, RandomSeed, LoggingLevel,
            IsProfile, MetricOptions, FlatParams, Metadata) ==
        std::tie(rhs.SystemOptions, rhs.BoostingOptions, rhs.ObliviousTreeOptions,
                rhs.DataProcessingOptions, rhs.LossFunctionDescription, rhs.CatFeatureParams,
                rhs.RandomSeed, rhs.LoggingLevel,
                rhs.IsProfile, rhs.MetricOptions, rhs.FlatParams, rhs.Metadata);
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
