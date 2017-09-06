#include "params.h"

#include <catboost/libs/helpers/exception.h>
#include "full_features.h"
#include "metric.h"

#include <library/json/json_reader.h>

#include <util/string/vector.h>
#include <util/system/info.h>
#include <util/string/split.h>
#include <util/string/iterator.h>

NJson::TJsonValue ReadTJsonValue(const TString& paramsJson) {
    TStringInput is(paramsJson);
    NJson::TJsonValue tree;
    NJson::ReadJsonTree(&is, &tree);
    return tree;
}

void CheckFitParams(const NJson::TJsonValue& tree,
                     const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                     const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor) {
    NJson::TJsonValue* resultingParams = nullptr;
    TFitParams params(tree, objectiveDescriptor, evalMetricDescriptor, resultingParams);
}

ELossFunction GetLossType(const TString& lossDescription) {
    yvector<TString> tokens = StringSplitter(lossDescription).SplitLimited(':', 2).ToList<TString>();
    CB_ENSURE(!tokens.empty(), "custom loss is missing in desctiption: " << lossDescription);
    ELossFunction customLoss;
    CB_ENSURE(TryFromString<ELossFunction>(tokens[0], customLoss), tokens[0] + " loss is not supported");
    return customLoss;
}

yhash<TString, float> GetLossParams(const TString& lossDescription) {
    const char* errorMessage = "Invalid metric description, it should be in the form "
                               "\"metric_name:param1=value1;...;paramN=valueN\"";

    yvector<TString> tokens = StringSplitter(lossDescription).SplitLimited(':', 2).ToList<TString>();
    CB_ENSURE(!tokens.empty(), "Metric description should not be empty");
    CB_ENSURE(tokens.size() <= 2, errorMessage);

    yhash<TString, float> params;
    if (tokens.size() == 2) {
        yvector<TString> paramsTokens = StringSplitter(tokens[1]).Split(';').ToList<TString>();

        for (const auto& token : paramsTokens) {
            yvector<TString> keyValue = StringSplitter(token).SplitLimited('=', 2).ToList<TString>();
            CB_ENSURE(keyValue.size() == 2, errorMessage);
            params[keyValue[0]] = FromString<float>(keyValue[1]);
        }
    }
    return params;
}

void CheckValues(const TFitParams& params) {
    if (!params.AllowWritingFiles) {
        CB_ENSURE(!params.SaveSnapshot, "allow_writing_files is set to False, and save_snapshot is set to True.");
    } else {
        CB_ENSURE(!params.TimeLeftLog.empty(), "empty time_left filename");
        CB_ENSURE(!params.LearnErrorLog.empty(), "empty learn_error filename");
    }

    CB_ENSURE(0 < params.BorderCount && params.BorderCount <= Max<ui8>(), "Invalid border count");
    CB_ENSURE(0 < params.CtrParams.CtrBorderCount && params.CtrParams.CtrBorderCount <= Max<ui8>(), "Invalid border count");
    CB_ENSURE(params.L2LeafRegularizer >= 0, "L2LeafRegularizer should be >= 0, current value: " << params.L2LeafRegularizer);
    const int maxModelDepth = Min(sizeof(TIndexType) * 8, size_t(16));
    CB_ENSURE(params.Depth > 0);
    CB_ENSURE(params.Depth <= maxModelDepth, "Maximum depth is " << maxModelDepth);
    CB_ENSURE(params.GradientIterations > 0);
    CB_ENSURE(params.Iterations >= 0);
    CB_ENSURE(params.Rsm > 0 && params.Rsm <= 1);
    CB_ENSURE(params.OverfittingDetectorIterationsWait >= 0);
    CB_ENSURE(params.BaggingTemperature >= 0);
    CB_ENSURE(params.FoldPermutationBlockSize > 0 || params.FoldPermutationBlockSize == ParameterNotSet);
    CB_ENSURE(!params.SaveSnapshot || !params.SnapshotFileName.empty(), "snapshot saving enabled but snapshot file name is empty");
    CB_ENSURE(params.ThreadCount > 0, "thread count should be positive");
    CB_ENSURE(params.ThreadCount <= CB_THREAD_LIMIT, "at most " << CB_THREAD_LIMIT << " thread(s) are supported; adjust CB_THREAD_LIMIT in params.h and rebuild");
    CB_ENSURE(AllOf(params.IgnoredFeatures, [](const int x) { return x >= 0; }), "ignored feature should not be negative");

    if (!params.ClassWeights.empty()) {
        CB_ENSURE(params.LossFunction == ELossFunction::Logloss || IsMultiClassError(params.LossFunction),
                  "class weights takes effect only with Logloss, MultiClass and MultiClassOneVsAll loss functions");
        CB_ENSURE(IsMultiClassError(params.LossFunction) || (params.ClassWeights.ysize() == 2),
                  "if loss-function is Logloss, then class weights should be given for 0 and 1 classes");
        CB_ENSURE(params.ClassesCount == 0 || params.ClassesCount == params.ClassWeights.ysize(), "class weights should be specified for each class in range 0, ... , classes_count - 1");
    }
    if (!params.ClassNames.empty()) {
        CB_ENSURE(params.LossFunction == ELossFunction::Logloss || IsMultiClassError(params.LossFunction),
                  "class names takes effect only with Logloss, MultiClass and MultiClassOneVsAll loss functions");
        CB_ENSURE(IsMultiClassError(params.LossFunction) || (params.ClassNames.ysize() == 2),
                  "if loss-function is Logloss, then class names should be given for 0 and 1 classes");
        CB_ENSURE(params.ClassesCount == 0 || params.ClassesCount == params.ClassNames.ysize(), "class names should be specified for each class in range 0, ... , classes_count - 1");
        if (!params.ClassWeights.empty()) {
            CB_ENSURE(params.ClassWeights.ysize() == params.ClassNames.ysize(), "classNames and classWeights should be the same size");
        }
    }

    for (auto& ctr : params.CtrParams.Ctrs) {
        CB_ENSURE(ctr.TargetBorderCount > 0, "at least one border for target");
        if (ctr.TargetBorderCount > 1) {
            CB_ENSURE(params.LossFunction == ELossFunction::RMSE || params.LossFunction == ELossFunction::Quantile || params.LossFunction == ELossFunction::LogLinQuantile || params.LossFunction == ELossFunction::Poisson || params.LossFunction == ELossFunction::MAPE || params.LossFunction == ELossFunction::MAE,
                      "target-border-cnt is not supported in this mode");
        }
    }

    if (params.LossFunction == ELossFunction::Quantile ||
        params.LossFunction == ELossFunction::MAE ||
        params.LossFunction == ELossFunction::LogLinQuantile ||
        params.LossFunction == ELossFunction::MAPE) {
        CB_ENSURE(params.LeafEstimationMethod != ELeafEstimation::Newton, "newton is not supported in this mode");
        CB_ENSURE(params.GradientIterations == 1, "gradient_iterations should equals 1 for this mode");
    }

    CB_ENSURE(params.FoldLenMultiplier > 1, "fold len multiplier should be greater than 1");
    CB_ENSURE(IsClassificationLoss(params.LossFunction) || params.PredictionType == EPredictionType::RawFormulaVal,
              "This prediction type is supported only for classification: " + ToString<EPredictionType>(params.PredictionType));

    for (const TString& lossDescription : params.CustomLoss) {
        auto customLoss = GetLossType(lossDescription);
        CB_ENSURE(IsClassificationLoss(params.LossFunction) == IsClassificationLoss(customLoss));
        CB_ENSURE(customLoss != ELossFunction::Custom, "User-defined custom metrics are not yet supported");
    }
}

template <class TInputIterator>
static yvector<float> ParsePriors(TInputIterator begin, TInputIterator end) {
    yvector<float> priors;
    for (auto middle = begin; middle != end; ++middle) {
        priors.push_back(middle->GetDoubleSafe());
    }
    return priors;
}

yvector<std::pair<int, yvector<float>>> GetPriors(const yvector<NJson::TJsonValue::TArray>& priors) {
    yvector<std::pair<int, yvector<float>>> result;
    for (const auto& jsonArray : priors) {
        Y_ASSERT(jsonArray.ysize() > 1);
        int index = jsonArray[0].GetIntegerSafe();
        result.emplace_back(index, ParsePriors(begin(jsonArray) + 1, end(jsonArray)));
    }
    return result;
}

yvector<std::pair<std::pair<int, int>, yvector<float>>> GetFeatureCtrPriors(const yvector<NJson::TJsonValue::TArray>& priors) {
    yvector<std::pair<std::pair<int, int>, yvector<float>>> result;
    for (const auto& jsonArray : priors) {
        Y_ASSERT(jsonArray.ysize() > 2);
        int feature = jsonArray[0].GetIntegerSafe();
        int ctr = jsonArray[1].GetIntegerSafe();
        result.emplace_back(std::make_pair(feature, ctr), ParsePriors(begin(jsonArray) + 2, end(jsonArray)));
    }
    return result;
}

void TFitParams::ParseCtrDescription(const NJson::TJsonValue& tree, yset<TString>* validKeys) {
    TString ctrDescriptionKey = "ctr_description";
    validKeys->insert(ctrDescriptionKey);
    if (tree.Has(ctrDescriptionKey)) {
        CtrParams.Ctrs.clear();
        auto ctrDescriptionParserFunc = [&](TStringBuf ctrStringDescription) {
            TCtrDescription ctr;
            GetNext<ECtrType>(ctrStringDescription, ':', ctr.CtrType);

            TMaybe<int> targetBorderCount;
            GetNext<int>(ctrStringDescription, ':', targetBorderCount);
            if (targetBorderCount.Defined()) {
                ctr.TargetBorderCount = *targetBorderCount.Get();
            }

            TMaybe<EBorderSelectionType> targetBorderType;
            GetNext<EBorderSelectionType>(ctrStringDescription, ':', targetBorderType);
            if (targetBorderType.Defined()) {
                ctr.TargetBorderType = *targetBorderType.Get();
            }

            CtrParams.Ctrs.emplace_back(ctr);
        };
        if (tree[ctrDescriptionKey].IsArray()) {
            for (const auto& treeElem : tree[ctrDescriptionKey].GetArraySafe()) {
                ctrDescriptionParserFunc(treeElem.GetStringSafe());
            }
        }
        else {
            ctrDescriptionParserFunc(tree[ctrDescriptionKey].GetStringSafe());
        }
    }
}

void TFitParams::InitFromJson(const NJson::TJsonValue& tree, NJson::TJsonValue* resultingParams) {
    if (resultingParams) {
        *resultingParams = tree;
    }
    yset<TString> validKeys;
#define GET_FIELD(json_name, target_name, type)           \
    validKeys.insert(#json_name);                         \
    if (tree.Has(#json_name)) {                           \
        target_name = tree[#json_name].Get##type##Safe(); \
    }
    GET_FIELD(iterations, Iterations, Integer)
    GET_FIELD(thread_count, ThreadCount, Integer)
    GET_FIELD(border, Border, Double)
    GET_FIELD(learning_rate, LearningRate, Double)
    GET_FIELD(depth, Depth, Integer)
    GET_FIELD(random_seed, RandomSeed, Integer)
    GET_FIELD(gradient_iterations, GradientIterations, Integer)
    GET_FIELD(rsm, Rsm, Double)
    GET_FIELD(bagging_temperature, BaggingTemperature, Double)
    GET_FIELD(fold_permutation_block_size, FoldPermutationBlockSize, Integer)
    GET_FIELD(ctr_border_count, CtrParams.CtrBorderCount, Integer)
    GET_FIELD(max_ctr_complexity, CtrParams.MaxCtrComplexity, Integer)
    GET_FIELD(border_count, BorderCount, Integer)
    GET_FIELD(od_pval, AutoStopPval, Double)
    GET_FIELD(od_wait, OverfittingDetectorIterationsWait, Integer)
    GET_FIELD(use_best_model, UseBestModel, Boolean)
    GET_FIELD(detailed_profile, DetailedProfile, Boolean)
    GET_FIELD(learn_error_log, LearnErrorLog, String)
    GET_FIELD(test_error_log, TestErrorLog, String)
    GET_FIELD(l2_leaf_reg, L2LeafRegularizer, Double)
    GET_FIELD(verbose, Verbose, Boolean)
    GET_FIELD(allow_writing_files, AllowWritingFiles, Boolean)
    GET_FIELD(has_time, HasTime, Boolean)
    GET_FIELD(name, Name, String)
    GET_FIELD(meta, MetaFileName, String)
    GET_FIELD(save_snapshot, SaveSnapshot, Boolean)
    GET_FIELD(snapshot_file, SnapshotFileName, String)
    GET_FIELD(train_dir, TrainDir, String)
    GET_FIELD(time_left_log, TimeLeftLog, String)
    GET_FIELD(fold_len_multiplier, FoldLenMultiplier, Double)
    GET_FIELD(ctr_leaf_count_limit, CtrLeafCountLimit, UInteger)
    GET_FIELD(store_all_simple_ctr, StoreAllSimpleCtr, Boolean)
    GET_FIELD(loss_function, Objective, String)
    GET_FIELD(eval_metric, EvalMetric, String)
    GET_FIELD(classes_count, ClassesCount, Integer)
    GET_FIELD(one_hot_max_size, OneHotMaxSize, Integer)
    GET_FIELD(random_strength, RandomStrength, Double)
    GET_FIELD(print_trees, PrintTrees, Boolean)
    GET_FIELD(developer_mode, DeveloperMode, Boolean)
    GET_FIELD(used_ram_limit, UsedRAMLimit, UInteger)
    GET_FIELD(approx_on_full_history, ApproxOnPartialHistory, Boolean)
#undef GET_FIELD

#define GET_ENUM_FIELD(json_name, target_name, type)                      \
    validKeys.insert(#json_name);                                         \
    if (tree.Has(#json_name)) {                                           \
        target_name = FromString<type>(tree[#json_name].GetStringSafe()); \
    }
    GET_ENUM_FIELD(od_type, OverfittingDetectorType, EOverfittingDetectorType)
    GET_ENUM_FIELD(leaf_estimation_method, LeafEstimationMethod, ELeafEstimation)
    GET_ENUM_FIELD(counter_calc_method, CounterCalcMethod, ECounterCalc)
    GET_ENUM_FIELD(feature_border_type, FeatureBorderType, EBorderSelectionType)
    GET_ENUM_FIELD(prediction_type, PredictionType, EPredictionType)
    GET_ENUM_FIELD(nan_mode, NanMode, ENanMode)
#undef GET_ENUM_FIELD

#define GET_VECTOR_FIELD(json_name, target_name, type)                  \
    validKeys.insert(#json_name);                                       \
    if (tree.Has(#json_name)) {                                         \
        target_name.clear();                                            \
        if (tree[#json_name].IsArray()) {                               \
            for (const auto& value : tree[#json_name].GetArraySafe()) { \
                target_name.push_back(value.Get##type##Safe());         \
            }                                                           \
        } else {                                                        \
            target_name.push_back(                                      \
                tree[#json_name].Get##type##Safe());                    \
        }                                                               \
    }
    yvector<NJson::TJsonValue::TArray> ctrPriors, featurePriors, featureCtrPriors;
    GET_VECTOR_FIELD(ignored_features, IgnoredFeatures, Integer)
    GET_VECTOR_FIELD(priors, CtrParams.DefaultPriors, Double)
    GET_VECTOR_FIELD(custom_loss, CustomLoss, String)
    GET_VECTOR_FIELD(class_weights, ClassWeights, Double)
    GET_VECTOR_FIELD(class_names, ClassNames, String)
    GET_VECTOR_FIELD(ctr_priors, ctrPriors, Array)
    GET_VECTOR_FIELD(feature_priors, featurePriors, Array)
    GET_VECTOR_FIELD(feature_ctr_priors, featureCtrPriors, Array)

#undef GET_VECTOR_FIELD

    LossFunction = GetLossType(Objective);
    CtrParams.PerCtrPriors = GetPriors(ctrPriors);
    CtrParams.PerFeaturePriors = GetPriors(featurePriors);
    CtrParams.PerFeatureCtrPriors = GetFeatureCtrPriors(featureCtrPriors);
    ParseCtrDescription(tree, &validKeys);


    TString threadCountKey = "thread_count";
    if (!tree.Has(threadCountKey)) {
        ThreadCount = Min(8, (int)NSystemInfo::CachedNumberOfCpus());
        if (resultingParams) {
            resultingParams->InsertValue(threadCountKey, ThreadCount);
        }
    }

    TString leafEstimationMethodKey = "leaf_estimation_method";
    TString gradientItersKey = "gradient_iterations";
    if (LossFunction == ELossFunction::Logloss) {
        bool leafEstimationMethodSet = tree.Has(leafEstimationMethodKey);
        if (!leafEstimationMethodSet) {
            LeafEstimationMethod = ELeafEstimation::Newton;
        }
        bool gradientItersSet = tree.Has(gradientItersKey);
        if (!gradientItersSet) {
            GradientIterations =
                LeafEstimationMethod == ELeafEstimation::Newton ? 10 : 100;
        }
    }
    if (IsMultiClassError(LossFunction)) {
        bool leafEstimationMethodSet = tree.Has(leafEstimationMethodKey);
        if (!leafEstimationMethodSet) {
            LeafEstimationMethod = ELeafEstimation::Newton;
        }
        bool gradientItersSet = tree.Has(gradientItersKey);
        if (!gradientItersSet) {
            GradientIterations =
                LeafEstimationMethod == ELeafEstimation::Newton ? 1 : 10;
        }
    }

    for (const auto& keyVal : tree.GetMap()) {
        CB_ENSURE(validKeys.has(keyVal.first), "invalid parameter: " << keyVal.first);
    }
    if (!EvalMetric) {
        CB_ENSURE(LossFunction != ELossFunction::Custom,
                  "Eval metric should be specified for custom objective");
        EvalMetric = Objective;
    }

    if (resultingParams) {
        (*resultingParams)[leafEstimationMethodKey] =
            ToString<ELeafEstimation>(LeafEstimationMethod);
        (*resultingParams)[gradientItersKey] = GradientIterations;
        (*resultingParams)["random_seed"] = RandomSeed;
        (*resultingParams)["eval_metric"] = *EvalMetric;
        (*resultingParams)["loss_function"] = Objective;
    }
    if (tree.Has("border")) {
        CB_ENSURE(LossFunction == ELossFunction::Logloss, "Border parameter should be set only for Logloss mode");
    }

    if (tree.Has("classes_count")) {
        CB_ENSURE(IsMultiClassError(LossFunction), "classes_count parameter takes effect only with MultiClass/MultiClassOneVsAll loss functions");
        CB_ENSURE(ClassesCount > 1, "classes-count should be at least 2");
    }
    CB_ENSURE(OverfittingDetectorType != EOverfittingDetectorType::Wilcoxon, "Wilcoxon detector is not supported");
    if (tree.Has("od_pval")) {
        CB_ENSURE(OverfittingDetectorType != EOverfittingDetectorType::Iter, "od_pval is not supported with iterational overfitting detector");
    }

    CheckValues(*this);
}
