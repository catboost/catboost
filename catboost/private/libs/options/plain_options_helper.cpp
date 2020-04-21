#include "catboost_options.h"
#include "loss_description.h"
#include "cat_feature_options.h"
#include "binarization_options.h"
#include "plain_options_helper.h"
#include "text_processing_options.h"

#include <catboost/libs/logging/logging.h>

#include <library/json/json_value.h>

#include <util/generic/strbuf.h>
#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/string/escape.h>
#include <util/string/strip.h>
#include <util/string/subst.h>
#include <util/system/compiler.h>

using NCatboostOptions::ParseCtrDescription;
using NCatboostOptions::ParsePerFeatureBinarization;
using NCatboostOptions::ParsePerFeatureCtrDescription;
using NCatboostOptions::BuildCtrOptionsDescription;


static Y_NO_INLINE void CopyCtrDescription(
    const NJson::TJsonValue& options,
    const TStringBuf srcKey,
    const TStringBuf dstKey,
    NJson::TJsonValue* const dst,
    TSet<TString>* const seenKeys
) {
    if (!options.Has(srcKey)) {
        return;
    }

    auto& arr = ((*dst)[dstKey] = NJson::TJsonValue(NJson::JSON_ARRAY));

    const NJson::TJsonValue& ctrDescriptions = options[srcKey];
    if (ctrDescriptions.IsArray()) {
        for (const auto& ctr : ctrDescriptions.GetArraySafe()) {
            arr.AppendValue(ParseCtrDescription(ctr.GetStringSafe()));
        }
    } else {
        arr.AppendValue(ParseCtrDescription(ctrDescriptions.GetStringSafe()));
    }

    seenKeys->insert(TString(srcKey));
}

static Y_NO_INLINE void ConcatenateCtrDescription(
    const NJson::TJsonValue& options,
    const TStringBuf sourceKey,
    const TStringBuf destinationKey,
    NJson::TJsonValue* const destination
) {
    if (!options.Has(sourceKey)) {
        return;
    }

    auto& ctrOptionsArray = (*destination)[destinationKey] = NJson::TJsonValue(NJson::JSON_ARRAY);
    const NJson::TJsonValue& ctrDescriptions = options[sourceKey];
    for (const auto& element: ctrDescriptions.GetArraySafe()) {
        const auto& ctrOptionsConcatenated = BuildCtrOptionsDescription(element);
        ctrOptionsArray.AppendValue(ctrOptionsConcatenated);
    }
}

static Y_NO_INLINE void CopyPerFeatureCtrDescription(
    const NJson::TJsonValue& options,
    const TStringBuf srcKey,
    const TStringBuf dstKey,
    NJson::TJsonValue* dst,
    TSet<TString>* seenKeys
) {
    if (!options.Has(srcKey)) {
        return;
    }

    NJson::TJsonValue& perFeatureCtrsMap = (*dst)[dstKey];
    perFeatureCtrsMap.SetType(NJson::JSON_MAP);
    const NJson::TJsonValue& ctrDescriptions = options[srcKey];
    CB_ENSURE(ctrDescriptions.IsArray());

    for (const auto& onePerFeatureCtrConfig : ctrDescriptions.GetArraySafe()) {
        auto perFeatureCtr = ParsePerFeatureCtrDescription(onePerFeatureCtrConfig.GetStringSafe());
        perFeatureCtrsMap[ToString<ui32>(perFeatureCtr.first)] = perFeatureCtr.second;
    }

    seenKeys->insert(TString(srcKey));
}

static Y_NO_INLINE void CopyPerFloatFeatureQuantization(
    const NJson::TJsonValue& options,
    const TStringBuf key,
    NJson::TJsonValue* dst,
    TSet<TString>* seenKeys
) {
    if (!options.Has(key)) {
        return;
    }

    NJson::TJsonValue& perFeatureBinarizationMap = (*dst)[key];
    perFeatureBinarizationMap.SetType(NJson::JSON_MAP);
    const NJson::TJsonValue& binarizationDescription = options[key];
    CB_ENSURE(binarizationDescription.IsArray());

    for (const auto& onePerFeatureCtrConfig : binarizationDescription.GetArraySafe()) {
        auto perFeatureBinarization = ParsePerFeatureBinarization(onePerFeatureCtrConfig.GetStringSafe());
        perFeatureBinarizationMap[perFeatureBinarization.first] = perFeatureBinarization.second;
    }

    seenKeys->insert(TString(key));
}

static Y_NO_INLINE void CopyOption(
    const NJson::TJsonValue& options,
    const TStringBuf key,
    NJson::TJsonValue* dst,
    TSet<TString>* seenKeys
) {
    if (options.Has(key)) {
        (*dst)[key] = options[key];
        seenKeys->insert(TString(key));
    }
}

static Y_NO_INLINE void CopyOptionWithNewKey(
    const NJson::TJsonValue& options,
    const TStringBuf srcKey,
    const TStringBuf dstKey,
    NJson::TJsonValue* dst,
    TSet<TString>* seenKeys
) {

    if (options.Has(srcKey)) {
        (*dst)[dstKey] = options[srcKey];
        seenKeys->insert(TString(srcKey));
    }
}

static bool HasLossFunctionSomeWhereInPlainOptions(
    const NJson::TJsonValue& plainOptions,
    const ELossFunction lossFunction
) {
    bool hasLossFunction = false;

    auto checkLossFunction = [&](const NJson::TJsonValue& metricOrLoss) {
        const auto& value = metricOrLoss.GetStringSafe();
        if (FromString<ELossFunction>(TStringBuf(value).Before(':')) == lossFunction) {
            hasLossFunction = true;
        }
    };

    for (const TStringBuf optionName : {"loss_function", "eval_metric"}) {
        if (!plainOptions.Has(optionName)) {
            continue;
        }
        checkLossFunction(plainOptions[optionName]);
    }

    if (plainOptions.Has("custom_metric") || plainOptions.Has("custom_loss")) {
        const NJson::TJsonValue& metrics = plainOptions.Has("custom_metric") ? plainOptions["custom_metric"] : plainOptions["custom_loss"];
        if (metrics.IsArray()) {
            for (const auto& metric : metrics.GetArraySafe()) {
                checkLossFunction(metric);
            }
        } else {
            checkLossFunction(metrics);
        }
    }

    return hasLossFunction;
}

// TODO(yazevnul): split catboost/app into catboost/app/lib and catboost/app so we can write
// unittests for cmdline invocation.
static void ValidatePlainOptionsConsistency(const NJson::TJsonValue& plainOptions) {
    CB_ENSURE(
        !(plainOptions.Has("custom_metric") && plainOptions.Has("custom_loss")),
        "custom_metric and custom_loss are incompatible");

    const auto hasMultiClass = HasLossFunctionSomeWhereInPlainOptions(
        plainOptions,
        ELossFunction::MultiClass);
    const auto hasMultiClassOneVsAll = HasLossFunctionSomeWhereInPlainOptions(
        plainOptions,
        ELossFunction::MultiClassOneVsAll);
    if (hasMultiClass && hasMultiClassOneVsAll) {
        // MultiClass trains multiclassifier (when implemented efficiently it has N-1, where N is
        // number of classes, target values and N-1 approxes), while MultiClassOneVsAll trains N
        // binary classifiers (thus we have N target valeus and N approxes), thus it's not clear how
        // we can compute both metrics at the same time.
        ythrow TCatBoostException()
            << ELossFunction::MultiClass << " and "
            << ELossFunction::MultiClassOneVsAll << " are incompatible";
    }
}


static Y_NO_INLINE void RemapPerFeatureCtrDescription(
    const NJson::TJsonValue& options,
    const TStringBuf sourceKey,
    const TStringBuf destinationKey,
    NJson::TJsonValue* const destination
) {
    auto& result = (*destination)[destinationKey] = NJson::TJsonValue(NJson::JSON_ARRAY);
    for (const auto& elem : options[sourceKey].GetMap()) {
        TString catFeatureIndex = elem.first;
        auto& ctrDict = elem.second[0];
        const auto& ctrOptionsConcatenated = BuildCtrOptionsDescription(ctrDict);
        result.AppendValue(catFeatureIndex + ":" + ctrOptionsConcatenated);
    }
}


static Y_NO_INLINE void ConcatenatePerFloatFeatureQuantizationOptions(
    const NJson::TJsonValue& options,
    const TStringBuf destinationKey,
    NJson::TJsonValue* const destination
) {
    auto& plainConcatenatedParams = (*destination)[destinationKey] = NJson::TJsonValue(NJson::JSON_ARRAY);
    for (auto& oneFeatureConfig : options["per_float_feature_quantization"].GetMap()) {
        TString concatenatedParams = ToString(oneFeatureConfig.first) + ":";
        for (auto& paramKeyValuePair : oneFeatureConfig.second.GetMapSafe()) {
            if (paramKeyValuePair.first == "border_count") {
                concatenatedParams = concatenatedParams + paramKeyValuePair.first + "=" + ToString(paramKeyValuePair.second) + ",";
            } else {
                concatenatedParams =
                        concatenatedParams + paramKeyValuePair.first + "=" + paramKeyValuePair.second.GetString() + ",";
            }
        }
        concatenatedParams.pop_back();
        plainConcatenatedParams.AppendValue(concatenatedParams);
    }
}

static Y_NO_INLINE void DeleteSeenOption(NJson::TJsonValue* options, const TStringBuf key) {
    if (options->Has(key)) {
        options->EraseValue(key);
    }
}

void NCatboostOptions::PlainJsonToOptions(
    const NJson::TJsonValue& plainOptions,
    NJson::TJsonValue* options,
    NJson::TJsonValue* outputOptions
) {
    ValidatePlainOptionsConsistency(plainOptions);
    TSet<TString> seenKeys;
    auto& trainOptions = *options;

    auto& lossFunctionRef = trainOptions["loss_function"];
    lossFunctionRef.SetType(NJson::JSON_MAP);
    if (plainOptions.Has("loss_function")) {
        lossFunctionRef = LossDescriptionToJson(plainOptions["loss_function"].GetStringSafe());
        seenKeys.insert("loss_function");
    }

    trainOptions["metrics"].SetType(NJson::JSON_MAP);

    if (plainOptions.Has("eval_metric")) {
        trainOptions["metrics"]["eval_metric"] = LossDescriptionToJson(plainOptions["eval_metric"].GetStringSafe());
        seenKeys.insert("eval_metric");
    }

    if (plainOptions.Has("custom_metric") || plainOptions.Has("custom_loss")) {
        const NJson::TJsonValue& metrics = plainOptions.Has("custom_metric") ? plainOptions["custom_metric"] : plainOptions["custom_loss"];
        if (metrics.IsArray()) {
            for (const auto& metric : metrics.GetArraySafe()) {
                trainOptions["metrics"]["custom_metrics"].AppendValue(LossDescriptionToJson(metric.GetStringSafe()));
            }
        } else {
            trainOptions["metrics"]["custom_metrics"].AppendValue(LossDescriptionToJson(metrics.GetStringSafe()));
        }
        seenKeys.insert(plainOptions.Has("custom_metric") ? "custom_metric" : "custom_loss");
    }

    NJson::TJsonValue& outputFilesJson = *outputOptions;
    outputFilesJson.SetType(NJson::JSON_MAP);

    CopyOption(plainOptions, "train_dir", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "name", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "meta", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "json_log", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "profile_log", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "learn_error_log", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "test_error_log", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "time_left_log", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "result_model_file", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "snapshot_file", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "save_snapshot", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "snapshot_interval", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "verbose", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "metric_period", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "prediction_type", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "output_columns", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "allow_writing_files", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "final_ctr_computation_mode", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "final_feature_calcer_computation_mode", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "use_best_model", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "best_model_min_trees", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "eval_file_name", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "fstr_regular_file", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "fstr_internal_file", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "fstr_type", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "training_options_file", &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "model_format",  &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "output_borders",  &outputFilesJson, &seenKeys);
    CopyOption(plainOptions, "roc_file",  &outputFilesJson, &seenKeys);


    //boosting options
    const char* const boostingOptionsKey = "boosting_options";
    NJson::TJsonValue& boostingOptionsRef = trainOptions[boostingOptionsKey];
    boostingOptionsRef.SetType(NJson::JSON_MAP);

    CopyOption(plainOptions, "iterations", &boostingOptionsRef, &seenKeys);
    CopyOption(plainOptions, "learning_rate", &boostingOptionsRef, &seenKeys);
    CopyOption(plainOptions, "fold_len_multiplier", &boostingOptionsRef, &seenKeys);
    CopyOption(plainOptions, "approx_on_full_history", &boostingOptionsRef, &seenKeys);
    CopyOption(plainOptions, "fold_permutation_block", &boostingOptionsRef, &seenKeys);
    CopyOption(plainOptions, "min_fold_size", &boostingOptionsRef, &seenKeys);
    CopyOption(plainOptions, "permutation_count", &boostingOptionsRef, &seenKeys);
    CopyOption(plainOptions, "boosting_type", &boostingOptionsRef, &seenKeys);
    CopyOption(plainOptions, "boost_from_average", &boostingOptionsRef, &seenKeys);
    CopyOption(plainOptions, "data_partition", &boostingOptionsRef, &seenKeys);
    CopyOption(plainOptions, "model_shrink_rate", &boostingOptionsRef, &seenKeys);
    CopyOption(plainOptions, "model_shrink_mode", &boostingOptionsRef, &seenKeys);
    CopyOption(plainOptions, "langevin", &boostingOptionsRef, &seenKeys);
    CopyOption(plainOptions, "diffusion_temperature", &boostingOptionsRef, &seenKeys);

    auto& odConfig = boostingOptionsRef["od_config"];
    odConfig.SetType(NJson::JSON_MAP);

    CopyOptionWithNewKey(plainOptions, "od_pval", "stop_pvalue", &odConfig, &seenKeys);
    CopyOptionWithNewKey(plainOptions, "od_wait", "wait_iterations", &odConfig, &seenKeys);
    CopyOptionWithNewKey(plainOptions, "od_type", "type", &odConfig, &seenKeys);

    auto& treeOptions = trainOptions["tree_learner_options"];
    treeOptions.SetType(NJson::JSON_MAP);
    CopyOption(plainOptions, "rsm", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "leaf_estimation_iterations", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "leaf_estimation_backtracking", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "depth", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "l2_leaf_reg", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "bayesian_matrix_reg", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "model_size_reg", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "dev_score_calc_obj_block_size", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "dev_efb_max_buckets", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "sparse_features_conflict_fraction", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "random_strength", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "leaf_estimation_method", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "grow_policy", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "max_leaves", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "min_data_in_leaf", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "score_function", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "fold_size_loss_normalization", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "add_ridge_penalty_to_loss_function", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "sampling_frequency", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "dev_max_ctr_complexity_for_borders_cache", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "observations_to_bootstrap", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "monotone_constraints", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "dev_leafwise_approxes", &treeOptions, &seenKeys);

    auto& bootstrapOptions = treeOptions["bootstrap"];
    bootstrapOptions.SetType(NJson::JSON_MAP);

    CopyOptionWithNewKey(plainOptions, "bootstrap_type", "type", &bootstrapOptions, &seenKeys);
    CopyOption(plainOptions, "bagging_temperature", &bootstrapOptions, &seenKeys);
    CopyOption(plainOptions, "subsample", &bootstrapOptions, &seenKeys);
    CopyOption(plainOptions, "mvs_reg", &bootstrapOptions, &seenKeys);
    CopyOption(plainOptions, "sampling_unit", &bootstrapOptions, &seenKeys);

    auto& featurePenaltiesOptions = treeOptions["penalties"];
    featurePenaltiesOptions.SetType(NJson::JSON_MAP);
    CopyOption(plainOptions, "feature_weights", &featurePenaltiesOptions, &seenKeys);
    CopyOption(plainOptions, "penalties_coefficient", &featurePenaltiesOptions, &seenKeys);
    CopyOption(plainOptions, "first_feature_use_penalties", &featurePenaltiesOptions, &seenKeys);

    //feature evaluation options
    if (GetTaskType(plainOptions) == ETaskType::GPU) {
        auto& modelBasedEvalOptions = trainOptions["model_based_eval_options"];
        modelBasedEvalOptions.SetType(NJson::JSON_MAP);

        CopyOption(plainOptions, "features_to_evaluate", &modelBasedEvalOptions, &seenKeys);
        CopyOption(plainOptions, "offset", &modelBasedEvalOptions, &seenKeys);
        CopyOption(plainOptions, "experiment_count", &modelBasedEvalOptions, &seenKeys);
        CopyOption(plainOptions, "experiment_size", &modelBasedEvalOptions, &seenKeys);
        CopyOption(plainOptions, "baseline_model_snapshot", &modelBasedEvalOptions, &seenKeys);
        CopyOption(plainOptions, "use_evaluated_features_in_baseline_model", &modelBasedEvalOptions, &seenKeys);
    }

    //cat-features
    auto& ctrOptions = trainOptions["cat_feature_params"];
    ctrOptions.SetType(NJson::JSON_MAP);

    if (plainOptions.Has("ctr_description")) {
        CATBOOST_WARNING_LOG << "ctr_description option is deprecated and will be removed soon. Tree/Simple ctr option will override this" << Endl;
        CopyCtrDescription(plainOptions, "ctr_description", "simple_ctrs", &ctrOptions, &seenKeys);
        CopyCtrDescription(plainOptions, "ctr_description", "combinations_ctrs", &ctrOptions, &seenKeys);
    }
    CopyCtrDescription(plainOptions, "simple_ctr", "simple_ctrs", &ctrOptions, &seenKeys);
    CopyCtrDescription(plainOptions, "combinations_ctr", "combinations_ctrs", &ctrOptions, &seenKeys);
    CopyPerFeatureCtrDescription(plainOptions, "per_feature_ctr", "per_feature_ctrs", &ctrOptions, &seenKeys);

    auto& ctrTargetBinarization = ctrOptions["target_binarization"];
    ctrTargetBinarization.SetType(NJson::JSON_MAP);
    CopyOptionWithNewKey(plainOptions, "ctr_target_border_count", "border_count", &ctrTargetBinarization, &seenKeys);

    CopyOption(plainOptions, "max_ctr_complexity", &ctrOptions, &seenKeys);
    CopyOption(plainOptions, "simple_ctr_description", &ctrOptions, &seenKeys);
    CopyOption(plainOptions, "tree_ctr_description", &ctrOptions, &seenKeys);
    CopyOption(plainOptions, "per_feature_ctr_description", &ctrOptions, &seenKeys);
    CopyOption(plainOptions, "counter_calc_method", &ctrOptions, &seenKeys);
    CopyOption(plainOptions, "store_all_simple_ctr", &ctrOptions, &seenKeys);
    CopyOption(plainOptions, "one_hot_max_size", &ctrOptions, &seenKeys);
    CopyOption(plainOptions, "ctr_leaf_count_limit", &ctrOptions, &seenKeys);
    CopyOption(plainOptions, "ctr_history_unit", &ctrOptions, &seenKeys);

    //data processing
    auto& dataProcessingOptions = trainOptions["data_processing_options"];
    dataProcessingOptions.SetType(NJson::JSON_MAP);

    CopyOption(plainOptions, "ignored_features", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "has_time", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "allow_const_label", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "target_border", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "classes_count", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "class_names", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "class_weights", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "dev_default_value_fraction_for_sparse", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "dev_sparse_array_indexing", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "gpu_cat_features_storage", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "dev_leafwise_scoring", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "dev_group_features", &dataProcessingOptions, &seenKeys);

    auto& floatFeaturesBinarization = dataProcessingOptions["float_features_binarization"];
    floatFeaturesBinarization.SetType(NJson::JSON_MAP);

    CopyOption(plainOptions, "border_count", &floatFeaturesBinarization, &seenKeys);
    CopyOptionWithNewKey(plainOptions, "feature_border_type", "border_type", &floatFeaturesBinarization, &seenKeys);
    CopyOption(plainOptions, "nan_mode", &floatFeaturesBinarization, &seenKeys);
    CopyOption(plainOptions, "dev_max_subset_size_for_build_borders", &floatFeaturesBinarization, &seenKeys);
    CopyPerFloatFeatureQuantization(plainOptions, "per_float_feature_quantization", &dataProcessingOptions, &seenKeys);

    auto& textProcessingOptions = dataProcessingOptions["text_processing_options"];
    ParseTextProcessingOptionsFromPlainJson(plainOptions, &textProcessingOptions, &seenKeys);

    //system
    auto& systemOptions = trainOptions["system_options"];
    systemOptions.SetType(NJson::JSON_MAP);

    CopyOption(plainOptions, "thread_count", &systemOptions, &seenKeys);
    CopyOptionWithNewKey(plainOptions, "device_config", "devices", &systemOptions, &seenKeys);
    CopyOption(plainOptions, "devices", &systemOptions, &seenKeys);
    CopyOption(plainOptions, "used_ram_limit", &systemOptions, &seenKeys);
    CopyOption(plainOptions, "gpu_ram_part", &systemOptions, &seenKeys);
    CopyOptionWithNewKey(plainOptions, "pinned_memory_size",
                            "pinned_memory_bytes", &systemOptions, &seenKeys);
    CopyOption(plainOptions, "node_type", &systemOptions, &seenKeys);
    CopyOption(plainOptions, "node_port", &systemOptions, &seenKeys);
    CopyOption(plainOptions, "file_with_hosts", &systemOptions, &seenKeys);


    //rest
    CopyOption(plainOptions, "random_seed", &trainOptions, &seenKeys);
    CopyOption(plainOptions, "logging_level", &trainOptions, &seenKeys);
    CopyOption(plainOptions, "detailed_profile", &trainOptions, &seenKeys);
    CopyOption(plainOptions, "task_type", &trainOptions, &seenKeys);
    CopyOption(plainOptions, "metadata", &trainOptions, &seenKeys);

    for (const auto& [optionName, optionValue] : plainOptions.GetMapSafe()) {
        if (!seenKeys.contains(optionName)) {
            const TString message = TStringBuilder()
                    //TODO(kirillovs): this cast fixes structured binding problem in msvc 14.12 compilator
                << "Unknown option {" << static_cast<const TString&>(optionName) << '}'
                << " with value \"" << EscapeC(optionValue.GetStringRobust()) << '"';
            ythrow TCatBoostException() << message;
        }
    }

    trainOptions["flat_params"] = plainOptions;
}

void NCatboostOptions::ConvertOptionsToPlainJson(
    const NJson::TJsonValue& options,
    const NJson::TJsonValue& outputOptions,
    NJson::TJsonValue* plainOptions
) {
    TSet<TString> seenKeys;

    NJson::TJsonValue& plainOptionsJson = *plainOptions;
    plainOptionsJson.SetType(NJson::JSON_MAP);

    NJson::TJsonValue optionsCopy(options);
    NJson::TJsonValue outputoptionsCopy(outputOptions);

    if (options.Has("loss_function")) {
        plainOptionsJson["loss_function"] = BuildMetricOptionDescription(options["loss_function"]);
        DeleteSeenOption(&optionsCopy, "loss_function");
    }

    if (options.Has("metrics")) {
        if (options["metrics"].Has("eval_metric")) {
            plainOptionsJson["eval_metric"] = BuildMetricOptionDescription(options["metrics"]["eval_metric"]);
            DeleteSeenOption(&optionsCopy["metrics"], "eval_metric");
        }
        if (options["metrics"].Has("custom_metrics")) {
            auto& result = plainOptionsJson["custom_metric"] = NJson::TJsonValue(NJson::JSON_ARRAY);
            for (auto& metric : options["metrics"]["custom_metrics"].GetArraySafe()) {
                result.AppendValue(BuildMetricOptionDescription(metric));
            }
            DeleteSeenOption(&optionsCopy["metrics"], "custom_metrics");
        }
        if (options["metrics"].Has("objective_metric")) {
            plainOptionsJson["objective_metric"] = BuildMetricOptionDescription(options["metrics"]["objective_metric"]);
            DeleteSeenOption(&optionsCopy["metrics"], "objective_metric");
        }
        CB_ENSURE(optionsCopy["metrics"].GetMapSafe().empty(), "metrics: key " + optionsCopy["metrics"].GetMapSafe().begin()->first + " wasn't added to plain options.");
        DeleteSeenOption(&optionsCopy, "metrics");
    }

    // outputOptions
    DeleteSeenOption(&outputoptionsCopy, "train_dir");
    DeleteSeenOption(&outputoptionsCopy, "name");
    DeleteSeenOption(&outputoptionsCopy, "meta");
    DeleteSeenOption(&outputoptionsCopy, "json_log");
    DeleteSeenOption(&outputoptionsCopy, "profile_log");
    DeleteSeenOption(&outputoptionsCopy, "learn_error_log");
    DeleteSeenOption(&outputoptionsCopy, "test_error_log");
    DeleteSeenOption(&outputoptionsCopy, "time_left_log");
    DeleteSeenOption(&outputoptionsCopy, "result_model_file");
    DeleteSeenOption(&outputoptionsCopy, "snapshot_file");
    DeleteSeenOption(&outputoptionsCopy, "save_snapshot");
    DeleteSeenOption(&outputoptionsCopy, "snapshot_interval");
    DeleteSeenOption(&outputoptionsCopy, "verbose");
    DeleteSeenOption(&outputoptionsCopy, "metric_period");
    DeleteSeenOption(&outputoptionsCopy, "prediction_type");
    DeleteSeenOption(&outputoptionsCopy, "output_columns");
    DeleteSeenOption(&outputoptionsCopy, "allow_writing_files");
    DeleteSeenOption(&outputoptionsCopy, "final_ctr_computation_mode");
    DeleteSeenOption(&outputoptionsCopy, "final_feature_calcer_computation_mode");

    CopyOption(outputOptions, "use_best_model", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&outputoptionsCopy, "use_best_model");

    CopyOption(outputOptions, "best_model_min_trees", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&outputoptionsCopy, "best_model_min_trees");

    DeleteSeenOption(&outputoptionsCopy, "eval_file_name");
    DeleteSeenOption(&outputoptionsCopy, "fstr_regular_file");
    DeleteSeenOption(&outputoptionsCopy, "fstr_internal_file");
    DeleteSeenOption(&outputoptionsCopy, "fstr_type");
    DeleteSeenOption(&outputoptionsCopy, "training_options_file");
    DeleteSeenOption(&outputoptionsCopy, "model_format");
    DeleteSeenOption(&outputoptionsCopy, "output_borders");
    DeleteSeenOption(&outputoptionsCopy, "roc_file");
    CB_ENSURE(outputoptionsCopy.GetMapSafe().empty(), "output_options: key " + outputoptionsCopy.GetMapSafe().begin()->first + " wasn't added to plain options.");

    // boosting options
    if (options.Has("boosting_options")) {
        const auto& boostingOptionsRef = options["boosting_options"];
        auto& optionsCopyBoosting = optionsCopy["boosting_options"];

        CopyOption(boostingOptionsRef, "iterations", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "iterations");

        CopyOption(boostingOptionsRef, "learning_rate", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "learning_rate");

        CopyOption(boostingOptionsRef, "fold_len_multiplier", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "fold_len_multiplier");

        CopyOption(boostingOptionsRef, "approx_on_full_history", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "approx_on_full_history");

        CopyOption(boostingOptionsRef, "fold_permutation_block", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "fold_permutation_block");

        CopyOption(boostingOptionsRef, "min_fold_size", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "min_fold_size");

        CopyOption(boostingOptionsRef, "permutation_count", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "permutation_count");

        CopyOption(boostingOptionsRef, "boosting_type", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "boosting_type");

        CopyOption(boostingOptionsRef, "boost_from_average", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "boost_from_average");

        CopyOption(boostingOptionsRef, "data_partition", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "data_partition");

        CopyOption(boostingOptionsRef, "model_shrink_rate", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "model_shrink_rate");

        CopyOption(boostingOptionsRef, "model_shrink_mode", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "model_shrink_mode");

        CopyOption(boostingOptionsRef, "langevin", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "langevin");

        CopyOption(boostingOptionsRef, "diffusion_temperature", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyBoosting, "diffusion_temperature");

        if (boostingOptionsRef.Has("od_config")) {
            const auto& odConfig = boostingOptionsRef["od_config"];
            auto& optionsCopyOdConfig = optionsCopyBoosting["od_config"];

            CopyOptionWithNewKey(odConfig, "type", "od_type", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyOdConfig, "type");

            CopyOptionWithNewKey(odConfig, "stop_pvalue", "od_pval", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyOdConfig, "stop_pvalue");

            CopyOptionWithNewKey(odConfig, "wait_iterations", "od_wait", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyOdConfig, "wait_iterations");

            CB_ENSURE(optionsCopyOdConfig.GetMapSafe().empty(), "od_config: key " + optionsCopyOdConfig.GetMapSafe().begin()->first + " wasn't added to plain options.");
            DeleteSeenOption(&optionsCopyBoosting, "od_config");
        }
        CB_ENSURE(optionsCopyBoosting.GetMapSafe().empty(), "boosting_options: key " + optionsCopyBoosting.GetMapSafe().begin()->first + " wasn't added to plain options.");
        DeleteSeenOption(&optionsCopy, "boosting_options");
    }

    if (options.Has("tree_learner_options")) {
        const auto& treeOptions = options["tree_learner_options"];
        auto& optionsCopyTree = optionsCopy["tree_learner_options"];

        CopyOption(treeOptions, "rsm", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "rsm");

        CopyOption(treeOptions, "leaf_estimation_iterations", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "leaf_estimation_iterations");

        CopyOption(treeOptions, "leaf_estimation_backtracking", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "leaf_estimation_backtracking");

        CopyOption(treeOptions, "depth", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "depth");

        CopyOption(treeOptions, "l2_leaf_reg", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "l2_leaf_reg");

        CopyOption(treeOptions, "bayesian_matrix_reg", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "bayesian_matrix_reg");

        CopyOption(treeOptions, "model_size_reg", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "model_size_reg");

        DeleteSeenOption(&optionsCopyTree, "dev_score_calc_obj_block_size");

        DeleteSeenOption(&optionsCopyTree, "dev_efb_max_buckets");

        CopyOption(treeOptions, "sparse_features_conflict_fraction", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "sparse_features_conflict_fraction");

        CopyOption(treeOptions, "random_strength", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "random_strength");

        CopyOption(treeOptions, "leaf_estimation_method", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "leaf_estimation_method");

        CopyOption(treeOptions, "grow_policy", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "grow_policy");

        CopyOption(treeOptions, "max_leaves", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "max_leaves");

        CopyOption(treeOptions, "min_data_in_leaf", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "min_data_in_leaf");

        CopyOption(treeOptions, "score_function", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "score_function");

        CopyOption(treeOptions, "fold_size_loss_normalization", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "fold_size_loss_normalization");

        CopyOption(treeOptions, "add_ridge_penalty_to_loss_function", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "add_ridge_penalty_to_loss_function");

        CopyOption(treeOptions, "sampling_frequency", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "sampling_frequency");

        DeleteSeenOption(&optionsCopyTree, "dev_max_ctr_complexity_for_borders_cache");

        CopyOption(treeOptions, "observations_to_bootstrap", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "observations_to_bootstrap");

        CopyOption(treeOptions, "monotone_constraints", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "monotone_constraints");

        CopyOption(treeOptions, "dev_leafwise_approxes", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyTree, "dev_leafwise_approxes");

        // bootstrap
        if (treeOptions.Has("bootstrap")) {
            const auto& bootstrapOptions = treeOptions["bootstrap"];
            auto& optionsCopyTreeBootstrap = optionsCopyTree["bootstrap"];

            CopyOptionWithNewKey(bootstrapOptions, "type", "bootstrap_type", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyTreeBootstrap, "type");

            CopyOption(bootstrapOptions, "bagging_temperature", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyTreeBootstrap, "bagging_temperature");

            CopyOption(bootstrapOptions, "subsample", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyTreeBootstrap, "subsample");

            CopyOption(bootstrapOptions, "mvs_reg", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyTreeBootstrap, "mvs_reg");

            CopyOption(bootstrapOptions, "sampling_unit", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyTreeBootstrap, "sampling_unit");

            CB_ENSURE(optionsCopyTreeBootstrap.GetMapSafe().empty(), "bootstrap: key " + optionsCopyTreeBootstrap.GetMapSafe().begin()->first + " wasn't added to plain options.");
            DeleteSeenOption(&optionsCopyTree, "bootstrap");
        }

        if (treeOptions.Has("penalties")) {
            const auto& penaltiesOptions = treeOptions["penalties"];
            auto& optionsCopyTreePenalties = optionsCopyTree["penalties"];

            CopyOption(penaltiesOptions, "feature_weights", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyTreePenalties, "feature_weights");

            CopyOption(penaltiesOptions, "penalties_coefficient", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyTreePenalties, "penalties_coefficient");

            CopyOption(penaltiesOptions, "first_feature_use_penalties", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyTreePenalties, "first_feature_use_penalties");

            CB_ENSURE(optionsCopyTreePenalties.GetMapSafe().empty(), "penalties: key " + optionsCopyTreePenalties.GetMapSafe().begin()->first + " wasn't added to plain options.");
            DeleteSeenOption(&optionsCopyTree, "penalties");
        }

        CB_ENSURE(optionsCopyTree.GetMapSafe().empty(), "tree_learner_options: key " + optionsCopyTree.GetMapSafe().begin()->first + " wasn't added to plain options.");
        DeleteSeenOption(&optionsCopy, "tree_learner_options");
    }

    // feature evaluation options
    if (optionsCopy.Has("model_based_eval_options")) {
        auto& optionsCopyBasedEval = optionsCopy["model_based_eval_options"];

        DeleteSeenOption(&optionsCopyBasedEval, "features_to_evaluate");
        DeleteSeenOption(&optionsCopyBasedEval, "offset");
        DeleteSeenOption(&optionsCopyBasedEval, "experiment_count");
        DeleteSeenOption(&optionsCopyBasedEval, "experiment_size");
        DeleteSeenOption(&optionsCopyBasedEval, "baseline_model_snapshot");
        DeleteSeenOption(&optionsCopyBasedEval, "use_evaluated_features_in_baseline_model");

        CB_ENSURE(optionsCopyBasedEval.GetMapSafe().empty(), "model_based_eval_options: key " + optionsCopyBasedEval.GetMapSafe().begin()->first + " wasn't added to plain options.");
        DeleteSeenOption(&optionsCopy, "model_based_eval_options");
    }

    // cat-features
    if (options.Has("cat_feature_params")) {
        const auto& ctrOptions = options["cat_feature_params"];
        auto& optionsCopyCtr = optionsCopy["cat_feature_params"];

        ConcatenateCtrDescription(ctrOptions, "simple_ctrs", "simple_ctr", &plainOptionsJson);
        DeleteSeenOption(&optionsCopyCtr, "simple_ctrs");

        ConcatenateCtrDescription(ctrOptions, "combinations_ctrs", "combinations_ctr", &plainOptionsJson);
        DeleteSeenOption(&optionsCopyCtr, "combinations_ctrs");

        RemapPerFeatureCtrDescription(ctrOptions, "per_feature_ctrs", "per_feature_ctr", &plainOptionsJson);
        DeleteSeenOption(&optionsCopyCtr, "per_feature_ctrs");

        if (ctrOptions.Has("target_binarization")) {
            const auto& ctrTargetBinarization = ctrOptions["target_binarization"];
            auto& optionsCopyCtrTargetBinarization = optionsCopyCtr["target_binarization"];
            CopyOptionWithNewKey(ctrTargetBinarization, "border_count", "ctr_target_border_count", &plainOptionsJson,
                                 &seenKeys);
            DeleteSeenOption(&optionsCopyCtrTargetBinarization, "border_count");
            DeleteSeenOption(&optionsCopyCtr, "target_binarization");
        }

        CopyOption(ctrOptions, "max_ctr_complexity", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyCtr, "max_ctr_complexity");

        CopyOption(ctrOptions, "simple_ctr_description", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyCtr, "simple_ctr_description");

        CopyOption(ctrOptions, "tree_ctr_description", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyCtr, "tree_ctr_description");

        CopyOption(ctrOptions, "per_feature_ctr_description", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyCtr, "per_feature_ctr_description");

        CopyOption(ctrOptions, "counter_calc_method", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyCtr, "counter_calc_method");

        CopyOption(ctrOptions, "store_all_simple_ctr", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyCtr, "store_all_simple_ctr");

        CopyOption(ctrOptions, "one_hot_max_size", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyCtr, "one_hot_max_size");

        CopyOption(ctrOptions, "ctr_leaf_count_limit", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyCtr, "ctr_leaf_count_limit");

        CopyOption(ctrOptions, "ctr_history_unit", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyCtr, "ctr_history_unit");

        CB_ENSURE(optionsCopyCtr.GetMapSafe().empty(), "cat_feature_params: key " + optionsCopyCtr.GetMapSafe().begin()->first + " wasn't added to plain options.");
        DeleteSeenOption(&optionsCopy, "cat_feature_params");
    }

    // data processing
    if (options.Has("data_processing_options")) {
        const auto& dataProcessingOptions = options["data_processing_options"];
        auto& optionsCopyDataProcessing = optionsCopy["data_processing_options"];

        CopyOption(dataProcessingOptions, "ignored_features", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyDataProcessing, "ignored_features");

        CopyOption(dataProcessingOptions, "has_time", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyDataProcessing, "has_time");

        CopyOption(dataProcessingOptions, "allow_const_label", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyDataProcessing, "allow_const_label");

        CopyOption(dataProcessingOptions, "classes_count", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyDataProcessing, "classes_count");

        CopyOption(dataProcessingOptions, "class_names", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyDataProcessing, "class_names");

        CopyOption(dataProcessingOptions, "class_weights", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyDataProcessing, "class_weights");

        CopyOption(dataProcessingOptions, "dev_default_value_fraction_for_sparse", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyDataProcessing, "dev_default_value_fraction_for_sparse");

        CopyOption(dataProcessingOptions, "dev_sparse_array_indexing", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyDataProcessing, "dev_sparse_array_indexing");

        CopyOption(dataProcessingOptions, "gpu_cat_features_storage", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyDataProcessing, "gpu_cat_features_storage");

        SaveTextProcessingOptionsToPlainJson(dataProcessingOptions["text_processing_options"], &plainOptionsJson);
        seenKeys.insert("text_processing_options");
        DeleteSeenOption(&optionsCopyDataProcessing, "text_processing_options");

        CopyOption(dataProcessingOptions, "dev_leafwise_scoring", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyDataProcessing, "dev_leafwise_scoring");

        CopyOption(dataProcessingOptions, "dev_group_features", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyDataProcessing, "dev_group_features");

        ConcatenatePerFloatFeatureQuantizationOptions(
            dataProcessingOptions,
            "per_float_feature_quantization",
            &plainOptionsJson);
        DeleteSeenOption(&optionsCopyDataProcessing, "per_float_feature_quantization");

        CopyOption(dataProcessingOptions, "target_border", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopyDataProcessing, "target_border");

        if (dataProcessingOptions.Has("float_features_binarization")) {
            const auto& floatFeaturesBinarization = dataProcessingOptions["float_features_binarization"];
            auto& optionsCopyDataProcessingFloatFeaturesBinarization = optionsCopyDataProcessing["float_features_binarization"];

            CopyOption(floatFeaturesBinarization, "border_count", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyDataProcessingFloatFeaturesBinarization, "border_count");

            CopyOptionWithNewKey(floatFeaturesBinarization, "border_type", "feature_border_type", &plainOptionsJson,
                                 &seenKeys);
            DeleteSeenOption(&optionsCopyDataProcessingFloatFeaturesBinarization, "border_type");

            CopyOption(floatFeaturesBinarization, "nan_mode", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyDataProcessingFloatFeaturesBinarization, "nan_mode");

            CopyOption(floatFeaturesBinarization, "dev_max_subset_size_for_build_borders", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&optionsCopyDataProcessingFloatFeaturesBinarization, "dev_max_subset_size_for_build_borders");

            CB_ENSURE(optionsCopyDataProcessingFloatFeaturesBinarization.GetMapSafe().empty(),
                      "float_features_binarization: key " + optionsCopyDataProcessingFloatFeaturesBinarization.GetMapSafe().begin()->first + " wasn't added to plain options.");
            DeleteSeenOption(&optionsCopyDataProcessing, "float_features_binarization");
        }
        CB_ENSURE(optionsCopyDataProcessing.GetMapSafe().empty(), "data_processing_options: key " + optionsCopyDataProcessing.GetMapSafe().begin()->first + " wasn't added to plain options.");
        DeleteSeenOption(&optionsCopy, "data_processing_options");
    }

    // system
    if (options.Has("system_options")) {
        const auto& systemOptions = options["system_options"];
        auto& optionsCopySystemOptions = optionsCopy["system_options"];

        CopyOption(systemOptions, "thread_count", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopySystemOptions, "thread_count");

        CopyOption(systemOptions, "devices", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopySystemOptions, "devices");

        CopyOption(systemOptions, "used_ram_limit", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopySystemOptions, "used_ram_limit");

        CopyOption(systemOptions, "gpu_ram_part", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopySystemOptions, "gpu_ram_part");

        CopyOption(systemOptions, "pinned_memory_bytes", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopySystemOptions, "pinned_memory_bytes");

        CopyOption(systemOptions, "node_type", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopySystemOptions, "node_type");

        CopyOption(systemOptions, "node_port", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopySystemOptions, "node_port");

        CopyOption(systemOptions, "file_with_hosts", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&optionsCopySystemOptions, "file_with_hosts");

        CB_ENSURE(optionsCopySystemOptions.GetMapSafe().empty(), "system_options: key " + optionsCopySystemOptions.GetMapSafe().begin()->first + " wasn't added to plain options.");
        DeleteSeenOption(&optionsCopy, "system_options");
    }

    // rest
    CopyOption(options, "random_seed", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&optionsCopy, "random_seed");

    CopyOption(options, "logging_level", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&optionsCopy, "logging_level");

    CopyOption(options, "detailed_profile", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&optionsCopy, "detailed_profile");

    CopyOption(options, "task_type", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&optionsCopy, "task_type");

    CopyOption(options, "metadata", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&optionsCopy, "metadata");

    DeleteSeenOption(&optionsCopy, "flat_params");
    CB_ENSURE(optionsCopy.GetMapSafe().empty(), "key " + optionsCopy.GetMapSafe().begin()->first + " wasn't added to plain options.");
}

void NCatboostOptions::CleanPlainJson(
    bool hasCatFeatures,
    NJson::TJsonValue* plainOptionsJsonEfficient,
    bool hasTextFeatures
) {

    CB_ENSURE(!plainOptionsJsonEfficient->GetMapSafe().empty(), "plainOptionsJsonEfficient should not be empty");

    if ((*plainOptionsJsonEfficient)["od_type"].GetStringSafe() == ToString(EOverfittingDetectorType::None)) {
        DeleteSeenOption(plainOptionsJsonEfficient, "od_type");
        DeleteSeenOption(plainOptionsJsonEfficient, "od_wait");
        DeleteSeenOption(plainOptionsJsonEfficient, "od_pval");
    }
    // options for distributed training
    DeleteSeenOption(plainOptionsJsonEfficient, "node_port");
    DeleteSeenOption(plainOptionsJsonEfficient, "file_with_hosts");
    DeleteSeenOption(plainOptionsJsonEfficient, "node_type");

    // options with no influence on the final model
    DeleteSeenOption(plainOptionsJsonEfficient, "objective_metric");
    DeleteSeenOption(plainOptionsJsonEfficient, "thread_count");
    DeleteSeenOption(plainOptionsJsonEfficient, "allow_const_label");
    DeleteSeenOption(plainOptionsJsonEfficient, "detailed_profile");
    DeleteSeenOption(plainOptionsJsonEfficient, "logging_level");

    if (!hasCatFeatures) {
        DeleteSeenOption(plainOptionsJsonEfficient, "simple_ctrs");
        DeleteSeenOption(plainOptionsJsonEfficient, "combinations_ctrs");
        DeleteSeenOption(plainOptionsJsonEfficient, "per_feature_ctrs");
        DeleteSeenOption(plainOptionsJsonEfficient, "target_binarization");
        DeleteSeenOption(plainOptionsJsonEfficient, "max_ctr_complexity");
        DeleteSeenOption(plainOptionsJsonEfficient, "simple_ctr_description");
        DeleteSeenOption(plainOptionsJsonEfficient, "tree_ctr_description");
        DeleteSeenOption(plainOptionsJsonEfficient, "per_feature_ctr_description");
        DeleteSeenOption(plainOptionsJsonEfficient, "counter_calc_method");
        DeleteSeenOption(plainOptionsJsonEfficient, "store_all_simple_ctr");
        DeleteSeenOption(plainOptionsJsonEfficient, "one_hot_max_size");
        DeleteSeenOption(plainOptionsJsonEfficient, "ctr_leaf_count_limit");
        DeleteSeenOption(plainOptionsJsonEfficient, "ctr_history_unit");
        DeleteSeenOption(plainOptionsJsonEfficient, "per_feature_ctr");
        DeleteSeenOption(plainOptionsJsonEfficient, "ctr_target_border_count");
        DeleteSeenOption(plainOptionsJsonEfficient, "combinations_ctr");
        DeleteSeenOption(plainOptionsJsonEfficient, "simple_ctr");
    }

    if ((*plainOptionsJsonEfficient)["boosting_type"].GetStringSafe() == ToString(EBoostingType::Plain)) {
        DeleteSeenOption(plainOptionsJsonEfficient, "approx_on_full_history");
        DeleteSeenOption(plainOptionsJsonEfficient, "fold_len_multiplier");
        if (!hasCatFeatures) {
            DeleteSeenOption(plainOptionsJsonEfficient, "permutation_count");
            DeleteSeenOption(plainOptionsJsonEfficient, "fold_permutation_block");
            DeleteSeenOption(plainOptionsJsonEfficient, "has_time");
        }
    }

    if (!hasTextFeatures) {
        DeleteSeenOption(plainOptionsJsonEfficient, "tokenizers");
        DeleteSeenOption(plainOptionsJsonEfficient, "dictionaries");
        DeleteSeenOption(plainOptionsJsonEfficient, "feature_calcers");
        DeleteSeenOption(plainOptionsJsonEfficient, "text_processing");
    }
    TVector<TStringBuf> keysToDelete;
    auto& map = plainOptionsJsonEfficient->GetMapSafe();
    for (const auto& [key, value] : map) {
        if (value.IsNull() ||
            value.IsArray() && value.GetArray().empty() ||
            value.IsMap() && value.GetMap().empty() ||
            value.IsString() && value.GetString().empty() ||
            key.substr(0, 4) == "dev_") {
            keysToDelete.push_back(key);
        }
    }
    for (const TStringBuf& key : keysToDelete) {
        DeleteSeenOption(plainOptionsJsonEfficient, key);
    }
}

