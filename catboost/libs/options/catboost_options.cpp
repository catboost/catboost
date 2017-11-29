#include "catboost_options.h"

namespace NCatboostOptions {
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
                //TODO(noxoomo): update to 10 after options merge
                defaultNewtonIterations = 1;
                defaultGradientIterations = 1;
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
                defaultGradientIterations = 100;
                defaultEstimationMethod = ELeavesEstimation::Newton;
                break;
            }
            case ELossFunction::UserPerObjErr:
            case ELossFunction::UserQuerywiseErr:
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
            treeConfig.LeavesEstimationMethod = defaultEstimationMethod;
        }

        if (treeConfig.LeavesEstimationIterations.NotSet()) {
            const ELeavesEstimation method = treeConfig.LeavesEstimationMethod;
            switch (method) {
                case ELeavesEstimation::Newton: {
                    treeConfig.LeavesEstimationIterations = defaultNewtonIterations;
                    break;
                }
                case ELeavesEstimation::Gradient: {
                    treeConfig.LeavesEstimationIterations = defaultGradientIterations;
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

    TCtrDescription TCatboostOptions::CreateDefaultCounter(EProjectionType projectionType) const {
        if (GetTaskType() == ETaskType::CPU) {
            return TCtrDescription(ETaskType::CPU, ECtrType::Counter, GetDefaultPriors(ECtrType::Counter));
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
            return TCtrDescription(ETaskType::GPU,
                                   ECtrType::FeatureFreq,
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
                defaultSimpleCtrs = {TCtrDescription(GetTaskType(), ECtrType::Borders, GetDefaultPriors(ECtrType::Borders)), CreateDefaultCounter(EProjectionType::SimpleCtr)};
                defaultTreeCtrs = {TCtrDescription(GetTaskType(), ECtrType::Borders, GetDefaultPriors(ECtrType::Borders)), CreateDefaultCounter(EProjectionType::TreeCtr)};
            }
        }

        if (catFeatureParams.SimpleCtrs.IsSet() && catFeatureParams.CombinationCtrs.NotSet()) {
            MATRIXNET_WARNING_LOG << "Change of simpleCtr will not affect tree ctrs." << Endl;
        }
        if (catFeatureParams.CombinationCtrs.IsSet() && catFeatureParams.SimpleCtrs.NotSet()) {
            MATRIXNET_WARNING_LOG << "Change of treeCtr will not affect simple ctrs" << Endl;
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
        if (!ctr.TargetBinarization.IsUnimplementedForCurrentTask() && (ctr.TargetBinarization->BorderCount > 1)) {
            CB_ENSURE(lossFunction == ELossFunction::RMSE || lossFunction == ELossFunction::Quantile ||
                          lossFunction == ELossFunction::LogLinQuantile || lossFunction == ELossFunction::Poisson ||
                          lossFunction == ELossFunction::MAPE || lossFunction == ELossFunction::MAE,
                      "target-border-cnt is not supported for loss function " << lossFunction);
        }
        CB_ENSURE(ctr.GetPriors().size(), "Provide at least one prior for CTR" << ToString(*this));

        if (GetTaskType() == ETaskType::GPU) {
            CB_ENSURE(IsSupportedOnGpu(ctr.Type),
                      "Ctr type " << ctr.Type << " is not implemented on GPU yet");
        } else {
            CB_ENSURE(GetTaskType() == ETaskType::CPU);
            CB_ENSURE(IsSupportedOnCpu(ctr.Type),
                      "Ctr type " << ctr.Type << " is not implemented on CPU yet");
        }

        const EBorderSelectionType borderSelectionType = ctr.CtrBinarization->BorderSelectionType;
        if (GetTaskType() == ETaskType::CPU) {
            CB_ENSURE(borderSelectionType == EBorderSelectionType::Uniform,
                      "Error: custom ctr binarization is not supported on CPU yet");
        } else {
            CB_ENSURE(GetTaskType() == ETaskType::GPU);
            if (isTreeCtrs) {
                EBorderSelectionType borderType = borderSelectionType;
                CB_ENSURE(borderType == EBorderSelectionType::Uniform || borderType == EBorderSelectionType::Median,
                          "Error: GPU supports Median and Uniform tree-ctr binarization only currently");

                CB_ENSURE(ctr.CtrBinarization->BorderCount <= GetMaxTreeCtrBinarizationForGpu(), "Error: max tree-ctr binarization for GPU is " << GetMaxTreeCtrBinarizationForGpu());
            }
        }

        if ((ctr.Type.Get() == ECtrType::FeatureFreq) && borderSelectionType == EBorderSelectionType::Uniform) {
            MATRIXNET_WARNING_LOG << "Uniform ctr binarization for featureFreq ctr is not good choice. Use MinEntropy for simpleCtrs and Median for tree-ctrs instead";
        }
    }

    void TCatboostOptions::Validate() const {
        SystemOptions.Get().Validate();
        BoostingOptions.Get().Validate();
        ObliviousTreeOptions.Get().Validate();

        ELossFunction lossFunction = LossFunctionDescription->GetLossFunction();
        {
            const ui32 classesCount = DataProcessingOptions->ClassesCount;
            if (classesCount != 0) {
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

        CB_ENSURE(!(IsQuerywiseError(lossFunction) && leavesEstimation == ELeavesEstimation::Newton),
                  "This leaf estimation method is not supported for querywise error");
        CB_ENSURE(!(IsPairwiseError(lossFunction) && leavesEstimation == ELeavesEstimation::Newton),
                  "This leaf estimation method is not supported for pairwise error");

        ValidateCtrs(CatFeatureParams->SimpleCtrs, lossFunction, false);
        for (const auto& perFeatureCtr : CatFeatureParams->PerFeatureCtrs.Get()) {
            ValidateCtrs(perFeatureCtr.second, lossFunction, false);
        }
        ValidateCtrs(CatFeatureParams->CombinationCtrs, lossFunction, true);
    }

    void TCatboostOptions::SetMetricDefaults(const TLossDescription& lossFunction) {
        if (GetTaskType() == ETaskType::CPU) {
            auto& evalMetric = MetricOptions->EvalMetric;
            if (evalMetric.IsSet()) {
                return;
            } else {
                evalMetric = lossFunction;
            }
        }
    }
}
