#include "catboost_options.h"

using namespace NCatboostOptions;

void TCatboostOptions::SetLeavesEstimationDefault() {
    const auto& lossFunctionConfig = LossFunctionDescription.Get();

    auto& treeConfig = ObliviousTreeOptions.Get();
    ui32 defaultNewtonIterations = 1;
    ui32 defaultGradientIterations = 1;
    ELeavesEstimation defaultEstimationMethod = ELeavesEstimation::Newton;

    switch (lossFunctionConfig.GetLossFunction()) {
        case ELossFunction::RMSE: {
            defaultEstimationMethod = ELeavesEstimation::Newton;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::QueryRMSE: {
            defaultEstimationMethod = ELeavesEstimation::Gradient;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 1;
            break;
        }
        case ELossFunction::QuerySoftMax: {
            //TODO(annaveronika;noxoomo): Change it to Newton when it is supported by CPU querywise loss
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
            defaultEstimationMethod = ELeavesEstimation::Gradient;
            defaultNewtonIterations = 1;
            defaultGradientIterations = 10;
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
            defaultEstimationMethod = (GetTaskType() == ETaskType::GPU) ? ELeavesEstimation::Newton : ELeavesEstimation::Gradient;
            defaultGradientIterations = 1;
            defaultNewtonIterations = 1;
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

    if (treeConfig.LeavesEstimationMethod.NotSet()) {
        treeConfig.LeavesEstimationMethod.SetDefault(defaultEstimationMethod);
    } else if (treeConfig.LeavesEstimationMethod != defaultEstimationMethod) {
        CB_ENSURE(lossFunctionConfig.GetLossFunction() != ELossFunction::YetiRank,
            "At the moment, in the YetiRank mode, changing the leaf_estimation_method parameter is prohibited.");
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
                ythrow TCatboostException() << "Unknown estimation type "
                                            << method;
            }
        }
    }

    if (treeConfig.L2Reg == 0.0f) {
        treeConfig.L2Reg = 1e-20f;
    }
}

void TCatboostOptions::Load(const NJson::TJsonValue& options) {
    ETaskType currentTaskType = TaskType;
    CheckedLoad(options,
                &TaskType,
                &SystemOptions, &BoostingOptions,
                &ObliviousTreeOptions,
                &DataProcessingOptions, &LossFunctionDescription,
                &RandomSeed, &CatFeatureParams,
                &FlatParams, &LoggingLevel,
                &IsProfile, &MetricOptions);
    SetNotSpecifiedOptionsToDefaults();
    CB_ENSURE(currentTaskType == GetTaskType(), "Task type in json-config is not equal to one specified for options");
    Validate();
}

void TCatboostOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, TaskType, SystemOptions, BoostingOptions, ObliviousTreeOptions,
               DataProcessingOptions, LossFunctionDescription,
               RandomSeed, CatFeatureParams, FlatParams, LoggingLevel, IsProfile, MetricOptions);
}

TCtrDescription TCatboostOptions::CreateDefaultCounter(EProjectionType projectionType) const {
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
                ythrow TCatboostException() << "Unknown projection type " << projectionType;
            }
        }
        return TCtrDescription(ECtrType::FeatureFreq,
                               GetDefaultPriors(ECtrType::FeatureFreq),
                               TBinarizationOptions(borderSelectionType, 15));
    }
}

static inline void SetDefaultBinarizationsIfNeeded(EProjectionType projectionType, TVector<TCtrDescription>* descriptions) {
    for (auto& description : (*descriptions)) {
        if (description.CtrBinarization.NotSet() && description.Type.Get() == ECtrType::FeatureFreq) {
            description.CtrBinarization->BorderSelectionType =  projectionType == EProjectionType::SimpleCtr ? EBorderSelectionType::MinEntropy : EBorderSelectionType::Median;
        }
    }
}

void TCatboostOptions::SetCtrDefaults() {
    TCatFeatureParams& catFeatureParams = CatFeatureParams.Get();
    ELossFunction lossFunction = LossFunctionDescription->GetLossFunction();

    TVector<TCtrDescription> defaultSimpleCtrs;
    TVector<TCtrDescription> defaultTreeCtrs;

    switch (lossFunction) {
        case ELossFunction::PairLogit: {
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
        MATRIXNET_WARNING_LOG << "Change of simpleCtr will not affect combinations ctrs." << Endl;
    }
    if (catFeatureParams.CombinationCtrs.IsSet() && catFeatureParams.SimpleCtrs.NotSet()) {
        MATRIXNET_WARNING_LOG << "Change of combinations ctrs will not affect simple ctrs" << Endl;
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

void TCatboostOptions::ValidateCtr(const TCtrDescription& ctr, ELossFunction lossFunction, bool isTreeCtrs) const {
    if (ctr.TargetBinarization->BorderCount > 1) {
        CB_ENSURE(lossFunction == ELossFunction::RMSE || lossFunction == ELossFunction::Quantile ||
                      lossFunction == ELossFunction::LogLinQuantile || lossFunction == ELossFunction::Poisson ||
                      lossFunction == ELossFunction::MAPE || lossFunction == ELossFunction::MAE,
                  "Setting TargetBorderCount is not supported for loss function " << lossFunction);
    }
    CB_ENSURE(ctr.GetPriors().size(), "Provide at least one prior for CTR" << ToString(*this));

    const ETaskType taskType = GetTaskType();
    const ECtrType ctrType = ctr.Type;

    if (taskType == ETaskType::GPU) {
        CB_ENSURE(IsSupportedOnGpu(ctrType),
                  "Ctr type " << ctrType << " is not implemented on GPU yet");
        CB_ENSURE(ctr.TargetBinarization.IsDefault(), "Error: GPU doesn't not support target binarization per CTR description currently. Please use target_borders option instead");
    } else {
        CB_ENSURE(taskType == ETaskType::CPU);
        CB_ENSURE(IsSupportedOnCpu(ctrType),
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

    if (taskType == ETaskType::GPU) {
        CB_ENSURE(!LossFunctionDescription->GetLossParams().has("decay"), "GPU implementation doesn't support decay parameter yet.");
    }

    if ((ctrType == ECtrType::FeatureFreq) && borderSelectionType == EBorderSelectionType::Uniform) {
        MATRIXNET_WARNING_LOG << "Uniform ctr binarization for featureFreq ctr is not good choice. Use MinEntropy for simpleCtrs and Median for combinations-ctrs instead" << Endl;
    }
}

void TCatboostOptions::Validate() const {
    SystemOptions.Get().Validate();
    BoostingOptions.Get().Validate();
    ObliviousTreeOptions.Get().Validate();

    ELossFunction lossFunction = LossFunctionDescription->GetLossFunction();
    {
        const ui32 classesCount = DataProcessingOptions->ClassesCount;
        if (classesCount != 0 ) {
            CB_ENSURE(IsMultiClassError(lossFunction), "classes_count parameter takes effect only with MultiClass/MultiClassOneVsAll loss functions");
            CB_ENSURE(classesCount > 1, "classes-count should be at least 2");
        }
        const auto& classWeights = DataProcessingOptions->ClassWeights.Get();
        if (!classWeights.empty()) {
            CB_ENSURE(lossFunction == ELossFunction::Logloss || IsMultiClassError(lossFunction),
                      "class weights takes effect only with Logloss, MultiClass and MultiClassOneVsAll loss functions");
            CB_ENSURE(IsMultiClassError(lossFunction) || (classWeights.size() == 2),
                      "if loss-function is Logloss, then class weights should be given for 0 and 1 classes");
            CB_ENSURE(classesCount == 0 || classesCount == classWeights.size(), "class weights should be specified for each class in range 0, ... , classes_count - 1");
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

    if (GetTaskType() == ETaskType::CPU) {
        CB_ENSURE(!(IsQuerywiseError(lossFunction) && leavesEstimation == ELeavesEstimation::Newton),
                  "This leaf estimation method is not supported for querywise error for CPU learning");
    }


    ValidateCtrs(CatFeatureParams->SimpleCtrs, lossFunction, false);
    for (const auto& perFeatureCtr : CatFeatureParams->PerFeatureCtrs.Get()) {
        ValidateCtrs(perFeatureCtr.second, lossFunction, false);
    }
    ValidateCtrs(CatFeatureParams->CombinationCtrs, lossFunction, true);
}

