#pragma once

#include "application_options.h"
#include "train_options.h"

#include <catboost/cuda/data/binarization_config.h>
#include <catboost/cuda/data/load_config.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/methods/oblivious_tree.h>
#include <catboost/cuda/methods/boosting.h>
#include <catboost/cuda/targets/target_options.h>
#include <library/getopt/small/last_getopt_opts.h>
#include <library/json/json_value.h>
#include <util/string/iterator.h>
#include <util/charset/utf8.h>
#include <cstdio>

namespace NCatboostCuda
{
    template<class TConfig>
    class TOptionsJsonConverter;


#define GET_FIELD(json_name, target_name, type)           \
    validKeys.insert(#json_name);                         \
    if (src.Has(#json_name)) {                           \
        target_name = src[#json_name].Get##type##Safe(); \
    }

#define GET_VECTOR_FIELD(json_name, target_name, type)                  \
    validKeys.insert(#json_name);                                       \
    if (src.Has(#json_name)) {                                         \
        target_name.clear();                                            \
        if (src[#json_name].IsArray()) {                               \
            for (const auto& value : src[#json_name].GetArraySafe()) { \
                target_name.push_back(value.Get##type##Safe());         \
            }                                                           \
        } else {                                                        \
            target_name.push_back(                                      \
                src[#json_name].Get##type##Safe());                    \
        }                                                               \
    }

#define GET_ENUM_FIELD(json_name, target_name, type)                      \
    validKeys.insert(#json_name);                                         \
    if (src.Has(#json_name)) {                                           \
        target_name = FromString<type>(src[#json_name].GetStringSafe()); \
    }

    template<class TDstContainer, class TSrcContainer>
    inline void Insert(const TSrcContainer& src, TDstContainer& dst)
    {
        for (auto& val : src) {
            dst.insert(val);
        }
    };

    template<>
    class TOptionsJsonConverter<TApplicationOptions>
    {
    public:


        static yset<TString> Load(const NJson::TJsonValue& src,
                                  TApplicationOptions& options)
        {

            yset<TString> validKeys;
            GET_FIELD(thread_count, options.NumThreads, Integer)
            GET_FIELD(gpu_ram_part, options.ApplicationConfig.GpuMemoryPartByWorker, Double)
            GET_FIELD(pinned_memory_size, options.ApplicationConfig.PinnedMemorySize, Integer)
            GET_FIELD(device_config, options.ApplicationConfig.DeviceConfig, String)
            return validKeys;
        }

        static void Save(const TApplicationOptions& options,
                         NJson::TJsonValue& dst)
        {
            dst["thread_count"] = options.NumThreads;
            dst["gpu_ram_part"] = options.ApplicationConfig.GpuMemoryPartByWorker;
            dst["pinned_memory_size"] = options.ApplicationConfig.PinnedMemorySize;
            dst["device_config"] = options.ApplicationConfig.DeviceConfig;
        }
    };


    template<>
    class TOptionsJsonConverter<TOverfittingDetectorOptions>
    {
//
    public:
        static yset<TString> Load(const NJson::TJsonValue& src,
                                  TOverfittingDetectorOptions& odOptions)
        {

            yset<TString> validKeys;
            GET_FIELD(od_pval, odOptions.AutoStopPValue, Double)
            GET_FIELD(od_wait, odOptions.IterationsWait, Integer)
            GET_ENUM_FIELD(od_type, odOptions.OverfittingDetectorType, EOverfittingDetectorType)
            return validKeys;
        }

        static void Save(const TOverfittingDetectorOptions& odOptions,
                         NJson::TJsonValue& dst)
        {
            dst["od_pval"] = odOptions.AutoStopPValue;
            dst["od_wait"] = odOptions.IterationsWait;
            dst["od_type"] = ToString<EOverfittingDetectorType>(odOptions.OverfittingDetectorType);
        }
    };

    template<>
    class TOptionsJsonConverter<TBoostingOptions>
    {
//
    public:
        static yset<TString> Load(const NJson::TJsonValue& src,
                                  TBoostingOptions& boostingOptions)
        {

            yset<TString> validKeys;
            GET_FIELD(iterations, boostingOptions.IterationCount, Integer)
            GET_FIELD(learning_rate, boostingOptions.Regularization, Double)
            GET_FIELD(fold_len_multiplier, boostingOptions.GrowthRate, Double)
            GET_FIELD(random_strength, boostingOptions.RandomStrength, Double)
            GET_FIELD(use_best_model, boostingOptions.UseBestModelFlag, Boolean)
            GET_FIELD(use_cpu_ram_for_cat_features, boostingOptions.UseCpuRamForCatFeaturesFlag, Boolean)
            GET_FIELD(fold_permutation_block_size, boostingOptions.PermutationBlockSize, Integer)
            GET_FIELD(has_time, boostingOptions.HasTimeFlag, Boolean)

            TOptionsJsonConverter<TOverfittingDetectorOptions>::Load(src, boostingOptions.OverfittingDetectorOptions);
            return validKeys;
        }

        static void Save(const TBoostingOptions& boostingOptions,
                         NJson::TJsonValue& dst)
        {
            dst["iterations"] = boostingOptions.IterationCount;
            dst["learning_rate"] = boostingOptions.Regularization;
            dst["permutation_count"] = boostingOptions.PermutationCount;
            dst["use_best_model"] = boostingOptions.UseBestModelFlag;
            dst["random_strength"] = boostingOptions.RandomStrength;
            dst["has_time"] = boostingOptions.HasTimeFlag;
            dst["fold_len_multiplier"] = boostingOptions.GrowthRate;
            dst["fold_permutation_block_size"] = boostingOptions.PermutationBlockSize;
            TOptionsJsonConverter<TOverfittingDetectorOptions>::Save(boostingOptions.OverfittingDetectorOptions, dst);
        }
    };

    template<>
    class TOptionsJsonConverter<TBinarizationConfiguration>
    {
    public:

        static void SetBinarizationForAllCtrs(TBinarizationConfiguration& configuration,
                                              int binarization)
        {
            CB_ENSURE(binarization <= 15, "Error: gpu supports <= 15 binarization for tree ctrs only");
            configuration.FreqCtrBinarization.Discretization = binarization;
            configuration.FreqTreeCtrBinarization.Discretization = binarization;
            configuration.DefaultTreeCtrBinarization.Discretization = binarization;
            configuration.DefaultCtrBinarization.Discretization = binarization;
        }

        static yset<TString> Load(const NJson::TJsonValue& src,
                                  TBinarizationConfiguration& binarizationConfiguration)
        {

            yset<TString> validKeys;
            int binarization = 15;
            GET_FIELD(ctr_border_count, binarization, Integer)
            SetBinarizationForAllCtrs(binarizationConfiguration, binarization);
            GET_FIELD(border_count, binarizationConfiguration.DefaultFloatBinarization.Discretization, Integer)
            GET_ENUM_FIELD(feature_border_type, binarizationConfiguration.DefaultFloatBinarization.BorderSelectionType,
                           EBorderSelectionType)
            return validKeys;
        }


        static void Save(const TBinarizationConfiguration& binarizationConfiguration,
                         NJson::TJsonValue& dst)
        {

            dst["ctr_border_count"] = binarizationConfiguration.DefaultCtrBinarization.Discretization;
            dst["border_count"] = binarizationConfiguration.DefaultFloatBinarization.Discretization;
            dst["feature_border_type"] = ToString<EBorderSelectionType>(
                    binarizationConfiguration.DefaultFloatBinarization.BorderSelectionType);
        }
    };

    template<>
    class TOptionsJsonConverter<TFeatureManagerOptions>
    {
    public:


        static yset<TString> Load(const NJson::TJsonValue& src,
                                  TFeatureManagerOptions& featureManagerOptions)
        {

            yset<TString> validKeys;

            GET_FIELD(max_ctr_complexity, featureManagerOptions.MaxTensorComplexity, Integer)
            GET_FIELD(one_hot_max_size, featureManagerOptions.OneHotLimit, Integer)
            yvector<int> ignoredFeatures;
            GET_VECTOR_FIELD(ignored_features, ignoredFeatures, Integer)
            for (auto f : ignoredFeatures) {
                featureManagerOptions.IgnoredFeatures.insert(f);
            }

            Insert(TOptionsJsonConverter<TBinarizationConfiguration>::Load(src, featureManagerOptions.BinarizationConfiguration), validKeys);
            return validKeys;
        }

        static void Save(const TFeatureManagerOptions& featureManagerOptions,
                         NJson::TJsonValue& dst)
        {
            dst["max_ctr_complexity"] = featureManagerOptions.MaxTensorComplexity;
            dst["one_hot_max_size"] = featureManagerOptions.OneHotLimit;

            for (auto f : featureManagerOptions.IgnoredFeatures) {
                dst["ignored_features"].AppendValue(f);
            }
            TOptionsJsonConverter<TBinarizationConfiguration>::Save(featureManagerOptions.BinarizationConfiguration,
                                                                    dst);
        }
    };


    template<>
    class TOptionsJsonConverter<TBootstrapConfig>
    {
    public:
        static yset<TString> Load(const NJson::TJsonValue& src,
                                  TBootstrapConfig& options)
        {
            yset<TString> validKeys;

            GET_ENUM_FIELD(bootstrap_type, options.BootstrapType, EBootstrapType)

            if (options.BootstrapType == EBootstrapType::Bayesian)
            {
                GET_FIELD(bagging_temperature, options.BaggingTemperature, Double)
            } else
            {
                GET_FIELD(sample_rate, options.BaggingTemperature, Double)
            }
            GET_FIELD(random_seed, options.Seed, Integer)
            return validKeys;

        }

        static void Save(const TBootstrapConfig& options,
                         NJson::TJsonValue& dst)
        {
            dst["bootstrap_type"] = ToString<EBootstrapType>(options.BootstrapType);
            if (options.BootstrapType == EBootstrapType::Bayesian)
            {
                dst["bagging_temperature"] = options.BaggingTemperature;
            } else
            {
                dst["sample_rate"] = options.TakenFraction;
            }
            dst["random_seed"] = options.Seed;
        }
    };


    template<>
    class TOptionsJsonConverter<TTargetOptions>
    {
    public:
        static yset<TString> Load(const NJson::TJsonValue& src,
                                  TTargetOptions& options)
        {
            yset<TString> validKeys;
            GET_ENUM_FIELD(loss_function, options.TargetType, ETargetFunction);
            if (options.TargetType == ETargetFunction::Logloss)
            {
                GET_FIELD(border, options.BinClassBorder, Double);
            }
            return validKeys;
        }

        static void Save(const TTargetOptions& options,
                         NJson::TJsonValue& dst)
        {

            dst["loss_function"] = ToString<ETargetFunction>(options.TargetType);
            if (options.TargetType == ETargetFunction::Logloss)
            {
                dst["border"] = options.BinClassBorder;
            }
        }
    };


    template<>
    class TOptionsJsonConverter<TObliviousTreeLearnerOptions>
    {
    public:
        static yset<TString> Load(const NJson::TJsonValue& src,
                                  TObliviousTreeLearnerOptions& options)
        {
            yset<TString> validKeys;
            GET_FIELD(depth, options.MaxDepth, Integer)
            GET_FIELD(l2_leaf_reg, options.L2Reg, Double)
            GET_FIELD(gradient_iterations, options.LeavesEstimationIters, Integer)

            TString method = "Newton";
            GET_FIELD(leaf_estimation_method, method, String)
            if (method == "Newton")
            {
                options.UseNewton = true;
            } else
            {
                CB_ENSURE(method == "Gradient");
                options.UseNewton = false;
            }
            auto tmp = TOptionsJsonConverter<TBootstrapConfig>::Load(src, options.BootstrapConfig);
            validKeys.insert(tmp.begin(), tmp.end());
            return validKeys;
        }

        static void Save(const TObliviousTreeLearnerOptions& options,
                         NJson::TJsonValue& dst)
        {
            if (options.UseNewton)
            {
                dst["leaf_estimation_method"] = "Newton";
            } else
            {
                dst["leaf_estimation_method"] = "Gradient";
            }
            dst["l2_leaf_reg"] = options.GetL2Reg();
            dst["gradient_iterations"] = options.LeavesEstimationIters;
            dst["depth"] = options.GetMaxDepth();
        }
    };

    template<>
    class TOptionsJsonConverter<TOutputFilesOptions>
    {
    public:
        static yset<TString> Load(const NJson::TJsonValue& src,
                                  TOutputFilesOptions& options)
        {
            yset<TString> validKeys;

            GET_FIELD(train_dir, options.TrainDir, String)
            GET_FIELD(learn_error_log, options.LearnErrorLogPath, String)
            GET_FIELD(test_error_log, options.TestErrorLogPath, String)
            GET_FIELD(name, options.Name, String)
            GET_FIELD(meta, options.MetaFile, String)
            GET_FIELD(time_left_log, options.TimeLeftLog, String)
            //TODO: snapshots support
            //GET_FIELD(snapshot_file, SnapshotFileName, String)
            return validKeys;
        }

        static void Save(const TOutputFilesOptions& options,
                         NJson::TJsonValue& dst)
        {
            if (!options.TrainDir.Empty())
            {
                dst["train_dir"] = options.TrainDir;
            }
            dst["learn_error_log"] = options.LearnErrorLogPath;
            dst["test_error_log"] = options.TestErrorLogPath;
            dst["name"] = options.Name;
            dst["meta"] = options.MetaFile;
            dst["time_left_log"] = options.TimeLeftLog;
        }
    };


#undef GET_FIELD
#undef GET_ENUM_FIELD
#undef GET_VECTOR_FIELD


    template<>
    class TOptionsJsonConverter<TTrainCatBoostOptions>
    {
    public:
        static void Load(const NJson::TJsonValue& src,
                         TTrainCatBoostOptions& trainCatboostOptions)
        {
            yset<TString> seenKeys;
            Insert(TOptionsJsonConverter<TApplicationOptions>::Load(src, trainCatboostOptions.ApplicationOptions),
                   seenKeys);
            Insert(TOptionsJsonConverter<TFeatureManagerOptions>::Load(src, trainCatboostOptions.FeatureManagerOptions),
                   seenKeys);
            Insert(TOptionsJsonConverter<TObliviousTreeLearnerOptions>::Load(src, trainCatboostOptions.TreeConfig),
                   seenKeys);
            Insert(TOptionsJsonConverter<TBoostingOptions>::Load(src, trainCatboostOptions.BoostingOptions), seenKeys);
            Insert(TOptionsJsonConverter<TTargetOptions>::Load(src, trainCatboostOptions.TargetOptions), seenKeys);
            Insert(TOptionsJsonConverter<TOutputFilesOptions>::Load(src, trainCatboostOptions.OutputFilesOptions),
                   seenKeys);

            for (const auto& keyVal : src.GetMap()) {
                CB_ENSURE(seenKeys.has(keyVal.first), "Unknown param: " << keyVal.first);
            }
        }

        static void Save(const TTrainCatBoostOptions& trainCatboostOptions,
                         NJson::TJsonValue& dst)
        {

            TOptionsJsonConverter<TApplicationOptions>::Save(trainCatboostOptions.ApplicationOptions, dst);
            TOptionsJsonConverter<TFeatureManagerOptions>::Save(trainCatboostOptions.FeatureManagerOptions, dst);
            TOptionsJsonConverter<TObliviousTreeLearnerOptions>::Save(trainCatboostOptions.TreeConfig, dst);
            TOptionsJsonConverter<TBoostingOptions>::Save(trainCatboostOptions.BoostingOptions, dst);
            TOptionsJsonConverter<TTargetOptions>::Save(trainCatboostOptions.TargetOptions, dst);
            TOptionsJsonConverter<TOutputFilesOptions>::Save(trainCatboostOptions.OutputFilesOptions, dst);

        }
    };
}
