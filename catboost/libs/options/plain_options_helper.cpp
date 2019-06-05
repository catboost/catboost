#include "cat_feature_options.h"
#include "loss_description.h"
#include "binarization_options.h"
#include "plain_options_helper.h"

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
    TSet<TString>* const seenKeys)
{
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
        const TStringBuf srcKey,
        const TStringBuf dstKey,
        NJson::TJsonValue* const dst,
        TSet<TString>* const seenKeys
) {
    if (!options.Has(srcKey)) {
        return;
    }

    auto& arr = (*dst)[dstKey] = NJson::TJsonValue(NJson::JSON_ARRAY);
    const NJson::TJsonValue& ctrDescriptions = options[srcKey];
    for (const auto& elem: ctrDescriptions.GetArraySafe()) {
        const auto& ctr_options_concatenated = BuildCtrOptionsDescription(elem);
        arr.AppendValue(ctr_options_concatenated);
    }
    seenKeys->insert(TString(srcKey));
}


static Y_NO_INLINE void CopyPerFeatureCtrDescription(
    const NJson::TJsonValue& options,
    const TStringBuf srcKey,
    const TStringBuf dstKey,
    NJson::TJsonValue* dst,
    TSet<TString>* seenKeys)
{
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

static Y_NO_INLINE void CopyPerFloatFeatureBinarization(
    const NJson::TJsonValue& options,
    const TStringBuf key,
    NJson::TJsonValue* dst,
    TSet<TString>* seenKeys)
{
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
    TSet<TString>* seenKeys)
{
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
    TSet<TString>* seenKeys) {

    if (options.Has(srcKey)) {
        (*dst)[dstKey] = options[srcKey];
        seenKeys->insert(TString(srcKey));
    }
}

static bool HasLossFunctionSomeWhereInPlainOptions(
    const NJson::TJsonValue& plainOptions,
    const ELossFunction lossFunction)
{
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
        const TStringBuf srcKey,
        const TStringBuf dstKey,
        NJson::TJsonValue* const dst)
{
    auto& result = (*dst)[dstKey] = NJson::TJsonValue(NJson::JSON_ARRAY);
    for (const auto& elem : options[srcKey].GetMap()) {
        TString CatFeatureIndex = elem.first;
        auto& CtrDict = elem.second[0];
        const auto& ctr_options_concatenated = BuildCtrOptionsDescription(CtrDict);
        result.AppendValue(CatFeatureIndex + ":" + ctr_options_concatenated);
    }
}

static Y_NO_INLINE void DeleteSeenOption(
        NJson::TJsonValue* options,
        const TStringBuf key)
{
    if (options->Has(key)) {
        options->EraseValue(key);
    }
}

void NCatboostOptions::PlainJsonToOptions(
    const NJson::TJsonValue& plainOptions,
    NJson::TJsonValue* options,
    NJson::TJsonValue* outputOptions)
{
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
    CopyOption(plainOptions, "data_partition", &boostingOptionsRef, &seenKeys);

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
    CopyOption(plainOptions, "efb_max_conflict_fraction", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "random_strength", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "leaf_estimation_method", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "grow_policy", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "max_leaves", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "min_data_in_leaf", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "score_function", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "fold_size_loss_normalization", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "add_ridge_penalty_to_loss_function", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "sampling_frequency", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "dev_max_ctr_complexity_for_border_cache", &treeOptions, &seenKeys);
    CopyOption(plainOptions, "observations_to_bootstrap", &treeOptions, &seenKeys);

    auto& bootstrapOptions = treeOptions["bootstrap"];
    bootstrapOptions.SetType(NJson::JSON_MAP);

    CopyOptionWithNewKey(plainOptions, "bootstrap_type", "type", &bootstrapOptions, &seenKeys);
    CopyOption(plainOptions, "bagging_temperature", &bootstrapOptions, &seenKeys);
    CopyOption(plainOptions, "subsample", &bootstrapOptions, &seenKeys);
    CopyOption(plainOptions, "mvs_head_fraction", &bootstrapOptions, &seenKeys);

    //feature evaluation options
    auto& modelBasedEvalOptions = trainOptions["model_based_eval_options"];
    modelBasedEvalOptions.SetType(NJson::JSON_MAP);

    CopyOption(plainOptions, "features_to_evaluate", &modelBasedEvalOptions, &seenKeys);
    CopyOption(plainOptions, "offset", &modelBasedEvalOptions, &seenKeys);
    CopyOption(plainOptions, "experiment_count", &modelBasedEvalOptions, &seenKeys);
    CopyOption(plainOptions, "experiment_size", &modelBasedEvalOptions, &seenKeys);
    CopyOption(plainOptions, "baseline_model_snapshot", &modelBasedEvalOptions, &seenKeys);

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
    CopyOption(plainOptions, "classes_count", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "class_names", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "class_weights", &dataProcessingOptions, &seenKeys);
    CopyOption(plainOptions, "gpu_cat_features_storage", &dataProcessingOptions, &seenKeys);

    auto& floatFeaturesBinarization = dataProcessingOptions["float_features_binarization"];
    floatFeaturesBinarization.SetType(NJson::JSON_MAP);

    CopyOption(plainOptions, "border_count", &floatFeaturesBinarization, &seenKeys);
    CopyOptionWithNewKey(plainOptions, "feature_border_type", "border_type", &floatFeaturesBinarization, &seenKeys);
    CopyOption(plainOptions, "nan_mode", &floatFeaturesBinarization, &seenKeys);
    CopyPerFloatFeatureBinarization(plainOptions, "per_float_feature_binarization", &dataProcessingOptions, &seenKeys);

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

    for (const auto& [optionName, optionValue] : plainOptions.GetMap()) {
        if (seenKeys.contains(optionName)) {
            break;
        }

        const TString message = TStringBuilder()
                //TODO(kirillovs): this cast fixes structured binding problem in msvc 14.12 compilator
            << "Unknown option {" << static_cast<const TString&>(optionName) << '}'
            << " with value \"" << EscapeC(optionValue.GetStringRobust()) << '"';
        ythrow TCatBoostException() << message;
    }

    trainOptions["flat_params"] = plainOptions;

}

void NCatboostOptions::ConvertOptionsToPlainJson(
        const NJson::TJsonValue& options,
        const NJson::TJsonValue& outputOptions,
        NJson::TJsonValue* plainOptions)
{
    TSet<TString> seenKeys;

    NJson::TJsonValue& plainOptionsJson = *plainOptions;
    plainOptionsJson.SetType(NJson::JSON_MAP);

    NJson::TJsonValue OptionsCopy(options);
    NJson::TJsonValue OutputOptionsCopy(outputOptions);

    if (options.Has("loss_function")) {
        plainOptionsJson["loss_function"] = BuildMetricOptionDescription(options["loss_function"]);
        DeleteSeenOption(&OptionsCopy, "loss_function");
    }

    if (options.Has("metrics")) {
        if (options["metrics"].Has("eval_metric")) {
            plainOptionsJson["eval_metric"] = BuildMetricOptionDescription(options["metrics"]["eval_metric"]);
            DeleteSeenOption(&OptionsCopy["metrics"], "eval_metric");
        }
        if (options["metrics"].Has("custom_metrics")) {
            auto &result = plainOptionsJson["custom_metric"] = NJson::TJsonValue(NJson::JSON_ARRAY);
            for (auto &metric : options["metrics"]["custom_metrics"].GetArray()) {
                result.AppendValue(BuildMetricOptionDescription(metric));
            }
            DeleteSeenOption(&OptionsCopy["metrics"], "custom_metrics");
        }
        if (options["metrics"].Has("objective_metric")) {
            plainOptionsJson["objective_metric"] = BuildMetricOptionDescription(options["metrics"]["objective_metric"]);
            DeleteSeenOption(&OptionsCopy["metrics"], "objective_metric");
        }
        CB_ENSURE(OptionsCopy["metrics"].GetMapSafe().empty(), "some loss or metrics keys missed");
        DeleteSeenOption(&OptionsCopy, "metrics");
    }

    // outputOptions
    DeleteSeenOption(&OutputOptionsCopy, "train_dir");
    DeleteSeenOption(&OutputOptionsCopy, "name");
    DeleteSeenOption(&OutputOptionsCopy, "meta");
    DeleteSeenOption(&OutputOptionsCopy, "json_log");
    DeleteSeenOption(&OutputOptionsCopy, "profile_log");
    DeleteSeenOption(&OutputOptionsCopy, "learn_error_log");
    DeleteSeenOption(&OutputOptionsCopy, "test_error_log");
    DeleteSeenOption(&OutputOptionsCopy, "time_left_log");
    DeleteSeenOption(&OutputOptionsCopy, "result_model_file");
    DeleteSeenOption(&OutputOptionsCopy, "snapshot_file");
    DeleteSeenOption(&OutputOptionsCopy, "save_snapshot");
    DeleteSeenOption(&OutputOptionsCopy, "snapshot_interval");
    DeleteSeenOption(&OutputOptionsCopy, "verbose");
    DeleteSeenOption(&OutputOptionsCopy, "metric_period");
    DeleteSeenOption(&OutputOptionsCopy, "prediction_type");
    DeleteSeenOption(&OutputOptionsCopy, "output_columns");
    DeleteSeenOption(&OutputOptionsCopy, "allow_writing_files");
    DeleteSeenOption(&OutputOptionsCopy, "final_ctr_computation_mode");

    CopyOption(outputOptions, "use_best_model", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&OutputOptionsCopy, "use_best_model");

    CopyOption(outputOptions, "best_model_min_trees", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&OutputOptionsCopy, "best_model_min_trees");

    DeleteSeenOption(&OutputOptionsCopy, "eval_file_name");
    DeleteSeenOption(&OutputOptionsCopy, "fstr_regular_file");
    DeleteSeenOption(&OutputOptionsCopy, "fstr_internal_file");
    DeleteSeenOption(&OutputOptionsCopy, "fstr_type");
    DeleteSeenOption(&OutputOptionsCopy, "training_options_file");
    DeleteSeenOption(&OutputOptionsCopy, "model_format");
    DeleteSeenOption(&OutputOptionsCopy, "output_borders");
    DeleteSeenOption(&OutputOptionsCopy, "roc_file");
    CB_ENSURE(OutputOptionsCopy.GetMapSafe().empty(), "some output_options keys missed");

    // boosting options
    if (options.Has("boosting_options")) {
        const NJson::TJsonValue& boostingOptionsRef = options["boosting_options"];
        NJson::TJsonValue& OptionsCopyBoosting = OptionsCopy["boosting_options"];

        CopyOption(boostingOptionsRef, "iterations", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyBoosting, "iterations");

        CopyOption(boostingOptionsRef, "learning_rate", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyBoosting, "learning_rate");

        CopyOption(boostingOptionsRef, "fold_len_multiplier", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyBoosting, "fold_len_multiplier");

        CopyOption(boostingOptionsRef, "approx_on_full_history", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyBoosting, "approx_on_full_history");

        CopyOption(boostingOptionsRef, "fold_permutation_block", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyBoosting, "fold_permutation_block");

        CopyOption(boostingOptionsRef, "min_fold_size", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyBoosting, "min_fold_size");

        CopyOption(boostingOptionsRef, "permutation_count", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyBoosting, "permutation_count");

        CopyOption(boostingOptionsRef, "boosting_type", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyBoosting, "boosting_type");

        CopyOption(boostingOptionsRef, "data_partition", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyBoosting, "data_partition");

        if (boostingOptionsRef.Has("od_config")) {
            auto &odConfig = boostingOptionsRef["od_config"];
            NJson::TJsonValue &OptionsCopyOdConfig = OptionsCopyBoosting["od_config"];

            CopyOptionWithNewKey(odConfig, "type", "od_type", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&OptionsCopyOdConfig, "type");

            CopyOptionWithNewKey(odConfig, "stop_pvalue", "od_pval", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&OptionsCopyOdConfig, "stop_pvalue");

            CopyOptionWithNewKey(odConfig, "wait_iterations", "od_wait", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&OptionsCopyOdConfig, "wait_iterations");

            CB_ENSURE(OptionsCopyOdConfig.GetMapSafe().empty(), "some keys in boosting od_config options missed");
            DeleteSeenOption(&OptionsCopyBoosting, "od_config");
        }
        CB_ENSURE(OptionsCopyBoosting.GetMapSafe().empty(), "some keys in boosting options missed");
        DeleteSeenOption(&OptionsCopy, "boosting_options");
    }

    if (options.Has("tree_learner_options")) {
        auto &treeOptions = options["tree_learner_options"];
        NJson::TJsonValue &OptionsCopyTree = OptionsCopy["tree_learner_options"];

        CopyOption(treeOptions, "rsm", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "rsm");

        CopyOption(treeOptions, "leaf_estimation_iterations", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "leaf_estimation_iterations");

        CopyOption(treeOptions, "leaf_estimation_backtracking", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "leaf_estimation_backtracking");

        CopyOption(treeOptions, "depth", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "depth");

        CopyOption(treeOptions, "l2_leaf_reg", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "l2_leaf_reg");

        CopyOption(treeOptions, "bayesian_matrix_reg", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "bayesian_matrix_reg");

        CopyOption(treeOptions, "model_size_reg", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "model_size_reg");

        DeleteSeenOption(&OptionsCopyTree, "dev_score_calc_obj_block_size");

        DeleteSeenOption(&OptionsCopyTree, "dev_efb_max_buckets");

        CopyOption(treeOptions, "efb_max_conflict_fraction", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "efb_max_conflict_fraction");

        CopyOption(treeOptions, "random_strength", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "random_strength");

        CopyOption(treeOptions, "leaf_estimation_method", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "leaf_estimation_method");

        CopyOption(treeOptions, "grow_policy", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "grow_policy");

        CopyOption(treeOptions, "max_leaves", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "max_leaves");

        CopyOption(treeOptions, "min_data_in_leaf", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "min_data_in_leaf");

        CopyOption(treeOptions, "score_function", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "score_function");

        CopyOption(treeOptions, "fold_size_loss_normalization", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "fold_size_loss_normalization");

        CopyOption(treeOptions, "add_ridge_penalty_to_loss_function", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "add_ridge_penalty_to_loss_function");

        CopyOption(treeOptions, "sampling_frequency", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "sampling_frequency");

        DeleteSeenOption(&OptionsCopyTree, "dev_max_ctr_complexity_for_border_cache");

        CopyOption(treeOptions, "observations_to_bootstrap", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyTree, "observations_to_bootstrap");

        // bootstrap
        if (treeOptions.Has("bootstrap")) {
            auto &bootstrapOptions = treeOptions["bootstrap"];
            NJson::TJsonValue &OptionsCopyTreeBootstrap = OptionsCopyTree["bootstrap"];

            CopyOptionWithNewKey(bootstrapOptions, "type", "bootstrap_type", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&OptionsCopyTreeBootstrap, "type");

            CopyOption(bootstrapOptions, "bagging_temperature", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&OptionsCopyTreeBootstrap, "bagging_temperature");

            CopyOption(bootstrapOptions, "subsample", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&OptionsCopyTreeBootstrap, "subsample");

            CopyOption(bootstrapOptions, "mvs_head_fraction", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&OptionsCopyTreeBootstrap, "mvs_head_fraction");

            CB_ENSURE(OptionsCopyTreeBootstrap.GetMapSafe().empty(), "some bootstrap keys missed");
            DeleteSeenOption(&OptionsCopyTree, "bootstrap");
        }
        CB_ENSURE(OptionsCopyTree.GetMapSafe().empty(), "some tree_learner_options keys missed");
        DeleteSeenOption(&OptionsCopy, "tree_learner_options");
    }

    //feature evaluation options
    if (OptionsCopy.Has("model_based_eval_options")) {
        NJson::TJsonValue& OptionsCopyBasedEval = OptionsCopy["model_based_eval_options"];

        DeleteSeenOption(&OptionsCopyBasedEval, "features_to_evaluate");
        DeleteSeenOption(&OptionsCopyBasedEval, "offset");
        DeleteSeenOption(&OptionsCopyBasedEval, "experiment_count");
        DeleteSeenOption(&OptionsCopyBasedEval, "experiment_size");
        DeleteSeenOption(&OptionsCopyBasedEval, "baseline_model_snapshot");

        CB_ENSURE(OptionsCopyBasedEval.GetMapSafe().empty(), "some model_based_eval_options keys missed");
        DeleteSeenOption(&OptionsCopy, "model_based_eval_options");
    }

    //cat-features
    if (options.Has("cat_feature_params")) {
        auto &ctrOptions = options["cat_feature_params"];
        NJson::TJsonValue &OptionsCopyCtr = OptionsCopy["cat_feature_params"];

        ConcatenateCtrDescription(ctrOptions, "simple_ctrs", "simple_ctr", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyCtr, "simple_ctrs");

        ConcatenateCtrDescription(ctrOptions, "combinations_ctrs", "combinations_ctr", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyCtr, "combinations_ctrs");

        RemapPerFeatureCtrDescription(ctrOptions, "per_feature_ctrs", "per_feature_ctr", &plainOptionsJson);
        DeleteSeenOption(&OptionsCopyCtr, "per_feature_ctrs");


        if (ctrOptions.Has("target_binarization")) {
            auto &ctrTargetBinarization = ctrOptions["target_binarization"];
            NJson::TJsonValue &OptionsCopyCtrTargetBinarization = OptionsCopyCtr["target_binarization"];

            CopyOptionWithNewKey(ctrTargetBinarization, "border_count", "ctr_target_border_count", &plainOptionsJson,
                                 &seenKeys);
            DeleteSeenOption(&OptionsCopyCtrTargetBinarization, "border_count");
            DeleteSeenOption(&OptionsCopyCtr, "target_binarization");
        }

        CopyOption(ctrOptions, "max_ctr_complexity", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyCtr, "max_ctr_complexity");

        CopyOption(ctrOptions, "simple_ctr_description", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyCtr, "simple_ctr_description");

        CopyOption(ctrOptions, "tree_ctr_description", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyCtr, "tree_ctr_description");

        CopyOption(ctrOptions, "per_feature_ctr_description", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyCtr, "per_feature_ctr_description");

        CopyOption(ctrOptions, "counter_calc_method", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyCtr, "counter_calc_method");

        CopyOption(ctrOptions, "store_all_simple_ctr", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyCtr, "store_all_simple_ctr");

        CopyOption(ctrOptions, "one_hot_max_size", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyCtr, "one_hot_max_size");

        CopyOption(ctrOptions, "ctr_leaf_count_limit", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyCtr, "ctr_leaf_count_limit");

        CopyOption(ctrOptions, "ctr_history_unit", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyCtr, "ctr_history_unit");

        CB_ENSURE(OptionsCopyCtr.GetMapSafe().empty(), "some ctr options keys missed");
        DeleteSeenOption(&OptionsCopy, "cat_feature_params");
    }

    //data processing
    if (options.Has("data_processing_options")) {
        auto &dataProcessingOptions = options["data_processing_options"];
        NJson::TJsonValue &OptionsCopyDataProcessing = OptionsCopy["data_processing_options"];

        CopyOption(dataProcessingOptions, "ignored_features", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyDataProcessing, "ignored_features");

        CopyOption(dataProcessingOptions, "has_time", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyDataProcessing, "has_time");

        CopyOption(dataProcessingOptions, "allow_const_label", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyDataProcessing, "allow_const_label");

        CopyOption(dataProcessingOptions, "classes_count", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyDataProcessing, "classes_count");

        CopyOption(dataProcessingOptions, "class_names", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyDataProcessing, "class_names");

        CopyOption(dataProcessingOptions, "class_weights", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyDataProcessing, "class_weights");

        CopyOption(dataProcessingOptions, "gpu_cat_features_storage", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopyDataProcessing, "gpu_cat_features_storage");

        if (dataProcessingOptions.Has("float_features_binarization")) {
            auto &floatFeaturesBinarization = dataProcessingOptions["float_features_binarization"];
            NJson::TJsonValue &OptionsCopyDataProcessingFloatFeaturesBinarization = OptionsCopyDataProcessing["float_features_binarization"];

            CopyOption(floatFeaturesBinarization, "border_count", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&OptionsCopyDataProcessingFloatFeaturesBinarization, "border_count");

            CopyOptionWithNewKey(floatFeaturesBinarization, "border_type", "feature_border_type", &plainOptionsJson,
                                 &seenKeys);
            DeleteSeenOption(&OptionsCopyDataProcessingFloatFeaturesBinarization, "border_type");

            CopyOption(floatFeaturesBinarization, "nan_mode", &plainOptionsJson, &seenKeys);
            DeleteSeenOption(&OptionsCopyDataProcessingFloatFeaturesBinarization, "nan_mode");

            CB_ENSURE(OptionsCopyDataProcessingFloatFeaturesBinarization.GetMapSafe().empty(),
                      "some float features binarization options keys missed");
            DeleteSeenOption(&OptionsCopyDataProcessing, "float_features_binarization");
        }
        CB_ENSURE(OptionsCopyDataProcessing.GetMapSafe().empty(), "some data processing options keys missed");
        DeleteSeenOption(&OptionsCopy, "data_processing_options");
    }

    //system
    if (options.Has("system_options")) {
        auto& systemOptions = options["system_options"];
        NJson::TJsonValue& OptionsCopySystemOptions = OptionsCopy["system_options"];

        CopyOption(systemOptions, "thread_count", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopySystemOptions, "thread_count");

        CopyOption(systemOptions, "devices", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopySystemOptions, "devices");

        CopyOption(systemOptions, "used_ram_limit", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopySystemOptions, "used_ram_limit");

        CopyOption(systemOptions, "gpu_ram_part", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopySystemOptions, "gpu_ram_part");

        CopyOption(systemOptions, "pinned_memory_bytes", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopySystemOptions, "pinned_memory_bytes");

        CopyOption(systemOptions, "node_type", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopySystemOptions, "node_type");

        CopyOption(systemOptions, "node_port", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopySystemOptions, "node_port");

        CopyOption(systemOptions, "file_with_hosts", &plainOptionsJson, &seenKeys);
        DeleteSeenOption(&OptionsCopySystemOptions, "file_with_hosts");

        CB_ENSURE(OptionsCopySystemOptions.GetMapSafe().empty(), "some system options keys missed");
        DeleteSeenOption(&OptionsCopy, "system_options");
    }

    //rest
    CopyOption(options, "random_seed", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&OptionsCopy, "random_seed");

    CopyOption(options, "logging_level", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&OptionsCopy, "logging_level");

    CopyOption(options, "detailed_profile", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&OptionsCopy, "detailed_profile");

    CopyOption(options, "task_type", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&OptionsCopy, "task_type");

    CopyOption(options, "metadata", &plainOptionsJson, &seenKeys);
    DeleteSeenOption(&OptionsCopy, "metadata");

    DeleteSeenOption(&OptionsCopy, "flat_params");

    CB_ENSURE(OptionsCopy.GetMapSafe().empty(), "some options keys missed");
}

void NCatboostOptions::DeleteEmptyKeysInPlainJson(
        NJson::TJsonValue* plainOptionsJsonEfficient,
        bool CatFeaturesArePresent) {

    if ((*plainOptionsJsonEfficient)["od_type"].GetStringSafe() == ToString(EOverfittingDetectorType::None)) {
        DeleteSeenOption(plainOptionsJsonEfficient, "od_type");
        DeleteSeenOption(plainOptionsJsonEfficient, "od_wait");
        DeleteSeenOption(plainOptionsJsonEfficient, "od_pval");
    }

    if (plainOptionsJsonEfficient->Has("node_type")) {
        if ((*plainOptionsJsonEfficient)["node_type"].GetStringSafe() == ToString("SingleHost")) {
            DeleteSeenOption(plainOptionsJsonEfficient, "node_port");
            DeleteSeenOption(plainOptionsJsonEfficient, "file_with_hosts");
        }
    }

    if (!CatFeaturesArePresent) {
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
    }
}

