#include "plain_options_helper.h"
#include "loss_description.h"
#include "cat_feature_options.h"
#include "output_file_options.h"
#include <catboost/libs/logging/logging.h>

namespace NCatboostOptions {
    inline void CopyCtrDescription(const NJson::TJsonValue& options, const TString& srcKey,
                                   const TString& dstKey, NJson::TJsonValue* dst, TSet<TString>* seenKeys) {
        if (options.Has(srcKey)) {
            (*dst)[dstKey] = NJson::TJsonValue(NJson::JSON_ARRAY);

            const NJson::TJsonValue& ctrDescriptions = options[srcKey];

            if (ctrDescriptions.IsArray()) {
                for (const auto& ctr : ctrDescriptions.GetArraySafe()) {
                    (*dst)[dstKey].AppendValue(ParseCtrDescription(ctr.GetStringSafe()));
                }
            } else {
                (*dst)[dstKey].AppendValue(ParseCtrDescription(ctrDescriptions.GetStringSafe()));
            }
            seenKeys->insert(srcKey);
        }
    }

    inline void CopyPerFeatureCtrDescription(const NJson::TJsonValue& options, const TString& srcKey,
                                             const TString& dstKey, NJson::TJsonValue* dst, TSet<TString>* seenKeys) {
        if (options.Has(srcKey)) {
            NJson::TJsonValue& perFeatureCtrsMap = (*dst)[dstKey];
            perFeatureCtrsMap.SetType(NJson::JSON_MAP);
            const NJson::TJsonValue& ctrDescriptions = options[srcKey];
            CB_ENSURE(ctrDescriptions.IsArray());

            for (const auto& onePerFeatureCtrConfig : ctrDescriptions.GetArraySafe()) {
                auto perFeatureCtr = ParsePerFeatureCtrDescription(onePerFeatureCtrConfig.GetStringSafe());
                perFeatureCtrsMap[ToString<ui32>(perFeatureCtr.first)] = perFeatureCtr.second;
            }
            seenKeys->insert(srcKey);
        }
    }

    inline void CopyOption(const NJson::TJsonValue& options, const TString& key, NJson::TJsonValue* dst, TSet<TString>* seenKeys) {
        if (options.Has(key)) {
            (*dst)[key] = options[key];
            seenKeys->insert(key);
        }
    }

    inline void CopyOptionWithNewKey(const NJson::TJsonValue& options, const TString& srcKey,
                                     const TString& dstKey, NJson::TJsonValue* dst, TSet<TString>* seenKeys) {
        if (options.Has(srcKey)) {
            (*dst)[dstKey] = options[srcKey];
            seenKeys->insert(srcKey);
        }
    }

    void PlainJsonToOptions(const NJson::TJsonValue& plainOptions, NJson::TJsonValue* options, NJson::TJsonValue* outputOptions) {
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
            if (plainOptions.Has("custom_metric") && plainOptions.Has("custom_loss")) {
                ythrow TCatboostException() << "Error: don't set custom_metric and custom_loss at the same time. Could be used only one option";
            }
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
        CopyOption(plainOptions, "snapshot_save_interval_secs", &outputFilesJson, &seenKeys);
        CopyOption(plainOptions, "verbose", &outputFilesJson, &seenKeys);
        CopyOption(plainOptions, "metric_period", &outputFilesJson, &seenKeys);
        CopyOption(plainOptions, "prediction_type", &outputFilesJson, &seenKeys);
        CopyOption(plainOptions, "output_columns", &outputFilesJson, &seenKeys);
        CopyOption(plainOptions, "allow_writing_files", &outputFilesJson, &seenKeys);
        CopyOption(plainOptions, "final_ctr_computation_mode", &outputFilesJson, &seenKeys);
        CopyOption(plainOptions, "use_best_model", &outputFilesJson, &seenKeys);
        CopyOption(plainOptions, "eval_file_name", &outputFilesJson, &seenKeys);
        CopyOption(plainOptions, "fstr_regular_file", &outputFilesJson, &seenKeys);
        CopyOption(plainOptions, "fstr_internal_file", &outputFilesJson, &seenKeys);
        CopyOption(plainOptions, "model_format",  &outputFilesJson, &seenKeys);
        CopyOption(plainOptions, "output_borders",  &outputFilesJson, &seenKeys);


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
        CopyOption(plainOptions, "random_strength", &treeOptions, &seenKeys);
        CopyOption(plainOptions, "leaf_estimation_method", &treeOptions, &seenKeys);
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

        //cat-features
        auto& ctrOptions = trainOptions["cat_feature_params"];
        ctrOptions.SetType(NJson::JSON_MAP);

        if (plainOptions.Has("ctr_description")) {
            MATRIXNET_WARNING_LOG << "ctr_description option is deprecated and will be removed soon. Tree/Simple ctr option will override this" << Endl;
            CopyCtrDescription(plainOptions, "ctr_description", "simple_ctrs", &ctrOptions, &seenKeys);
            CopyCtrDescription(plainOptions, "ctr_description", "combinations_ctrs", &ctrOptions, &seenKeys);
        }
        CopyCtrDescription(plainOptions, "simple_ctr", "simple_ctrs", &ctrOptions, &seenKeys);
        CopyCtrDescription(plainOptions, "combinations_ctr", "combinations_ctrs", &ctrOptions, &seenKeys);
        CopyPerFeatureCtrDescription(plainOptions, "per_feature_ctr", "per_feature_ctrs", &ctrOptions, &seenKeys);

        CopyOption(plainOptions, "max_ctr_complexity", &ctrOptions, &seenKeys);
        CopyOption(plainOptions, "simple_ctr_description", &ctrOptions, &seenKeys);
        CopyOption(plainOptions, "tree_ctr_description", &ctrOptions, &seenKeys);
        CopyOption(plainOptions, "per_feature_ctr_description", &ctrOptions, &seenKeys);
        CopyOption(plainOptions, "counter_calc_method", &ctrOptions, &seenKeys);
        CopyOption(plainOptions, "store_all_simple_ctr", &ctrOptions, &seenKeys);
        CopyOption(plainOptions, "one_hot_max_size", &ctrOptions, &seenKeys);
        CopyOption(plainOptions, "ctr_leaf_count_limit", &ctrOptions, &seenKeys);

        //data processing
        auto& dataProcessingOptions = trainOptions["data_processing_options"];
        dataProcessingOptions.SetType(NJson::JSON_MAP);

        CopyOption(plainOptions, "ignored_features", &dataProcessingOptions, &seenKeys);
        CopyOption(plainOptions, "has_time", &dataProcessingOptions, &seenKeys);
        CopyOption(plainOptions, "allow_const_label", &dataProcessingOptions, &seenKeys);
        CopyOption(plainOptions, "allow_negative_weights", &dataProcessingOptions, &seenKeys);
        CopyOption(plainOptions, "classes_count", &dataProcessingOptions, &seenKeys);
        CopyOption(plainOptions, "class_names", &dataProcessingOptions, &seenKeys);
        CopyOption(plainOptions, "class_weights", &dataProcessingOptions, &seenKeys);
        CopyOption(plainOptions, "gpu_cat_features_storage", &dataProcessingOptions, &seenKeys);

        auto& floatFeaturesBinarization = dataProcessingOptions["float_features_binarization"];
        floatFeaturesBinarization.SetType(NJson::JSON_MAP);

        CopyOption(plainOptions, "border_count", &floatFeaturesBinarization, &seenKeys);
        CopyOptionWithNewKey(plainOptions, "feature_border_type", "border_type", &floatFeaturesBinarization, &seenKeys);
        CopyOption(plainOptions, "nan_mode", &floatFeaturesBinarization, &seenKeys);

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

        for (const auto& option : plainOptions.GetMap()) {
            if (!seenKeys.has(option.first)) {
                ythrow TCatboostException() << "Error: unknown option " << option.first << " with value " << option.second;
            }
        }

        trainOptions["flat_params"] = plainOptions;
    }
}
