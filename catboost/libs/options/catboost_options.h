#pragma once

#include "option.h"
#include "system_options.h"
#include "boosting_options.h"
#include "oblivious_tree_options.h"
#include "output_file_options.h"
#include "binarization_options.h"
#include "loss_description.h"
#include "data_processing_options.h"
#include "cat_feature_options.h"
#include "metric_options.h"

#include <library/json/json_reader.h>
#include <util/system/types.h>

namespace NCatboostOptions {
    class TCatBoostOptions {
    public:
        explicit TCatBoostOptions(ETaskType taskType)
            : SystemOptions("system_options", TSystemOptions(taskType))
            , BoostingOptions("boosting_options", TBoostingOptions(taskType))
            , ObliviousTreeOptions("tree_learner_options", TObliviousTreeLearnerOptions(taskType))
            , DataProcessingOptions("data_processing_options", TDataProcessingOptions(taskType))
            , LossFunctionDescription("loss_function", TLossDescription())
            , CatFeatureParams("cat_feature_params", TCatFeatureParams(taskType))
            , FlatParams("flat_params", NJson::TJsonValue(NJson::JSON_MAP))
            , RandomSeed("random_seed", GetCycleCount())
            , LoggingLevel("logging_level", ELoggingLevel::Verbose)
            , IsProfile("detailed_profile", false)
            , MetricOptions("metrics", TMetricOptions(), taskType)
            , TaskType("task_type", taskType)
        {
        }

        void Load(const NJson::TJsonValue& options);
        void Save(NJson::TJsonValue* options) const;

        bool operator==(const TCatBoostOptions& rhs) const {
            return std::tie(SystemOptions, BoostingOptions, ObliviousTreeOptions,  DataProcessingOptions,
                            LossFunctionDescription, CatFeatureParams, RandomSeed, LoggingLevel, IsProfile, MetricOptions, FlatParams) ==
                   std::tie(rhs.SystemOptions, rhs.BoostingOptions, rhs.ObliviousTreeOptions,
                            rhs.DataProcessingOptions, rhs.LossFunctionDescription, rhs.CatFeatureParams,
                            rhs.RandomSeed, rhs.LoggingLevel, rhs.IsProfile, rhs.MetricOptions, rhs.FlatParams);
        }

        bool operator!=(const TCatBoostOptions& rhs) const {
            return !(rhs == *this);
        }

        ETaskType GetTaskType() const {
            return TaskType.Get();
        }

        void Validate() const;

        void SetNotSpecifiedOptionsToDefaults() {
            SetLeavesEstimationDefault();
            SetCtrDefaults();

            if (DataProcessingOptions->HasTimeFlag) {
                BoostingOptions->PermutationCount = 1;
            }
        }

    public:
        TOption<TSystemOptions> SystemOptions;
        TOption<TBoostingOptions> BoostingOptions;
        TOption<TObliviousTreeLearnerOptions> ObliviousTreeOptions;
        TOption<TDataProcessingOptions> DataProcessingOptions;
        TOption<TLossDescription> LossFunctionDescription;
        TOption<TCatFeatureParams> CatFeatureParams;
        TOption<NJson::TJsonValue> FlatParams;

        TOption<ui64> RandomSeed;
        TOption<ELoggingLevel> LoggingLevel;
        TOption<bool> IsProfile;
        TCpuOnlyOption<TMetricOptions> MetricOptions;

    private:
        void ValidateCtr(const TCtrDescription& ctr, ELossFunction lossFunction, bool isTreeCtrs) const;

        void SetLeavesEstimationDefault();

        TCtrDescription CreateDefaultCounter(EProjectionType projectionType) const;

        void SetCtrDefaults();

        void SetDefaultPriorsIfNeeded(TVector<TCtrDescription>& ctrs) const {
            for (auto& ctr : ctrs) {
                if (!ctr.ArePriorsSet()) {
                    ctr.SetPriors(GetDefaultPriors(ctr.Type));
                }
            }
        }

        void ValidateCtrs(const TVector<TCtrDescription>& ctrDescription,
                          ELossFunction lossFunction,
                          bool isTreeCtrs) const {
            for (const auto& ctr : ctrDescription) {
                ValidateCtr(ctr, lossFunction, isTreeCtrs);
            }
        }

    private:
        TOption<ETaskType> TaskType;
    };

    inline ETaskType GetTaskType(const NJson::TJsonValue& source) {
        TOption<ETaskType> taskType("task_type", ETaskType::CPU);
        TJsonFieldHelper<decltype(taskType)>::Read(source, &taskType);
        return taskType.Get();
    }

    inline TCatBoostOptions LoadOptions(const NJson::TJsonValue& source) {
        //little hack. JSON parsing needs to known device_type
        TOption<ETaskType> taskType("task_type", ETaskType::CPU);
        TJsonFieldHelper<decltype(taskType)>::Read(source, &taskType);
        TCatBoostOptions options(taskType.Get());
        options.Load(source);
        return options;
    }

}

using TCatboostOptions = NCatboostOptions::TCatBoostOptions;

template <>
inline TString ToString<TCatboostOptions>(const TCatboostOptions& options) {
    NJson::TJsonValue json;
    options.Save(&json);
    return ToString(json);
}

template <>
inline TCatboostOptions FromString<TCatboostOptions>(const TString& str) {
    NJson::TJsonValue json;
    NJson::ReadJsonTree(TStringBuf(str), &json, true);
    return NCatboostOptions::LoadOptions(json);
}

inline TVector<int> GetOptionIgnoredFeatures(const NJson::TJsonValue& catBoostJsonOptions) {
    TVector<int> result;
    auto& dataProcessingOptions = catBoostJsonOptions["data_processing_options"];
    if (dataProcessingOptions.IsMap()) {
        auto& ignoredFeatures = dataProcessingOptions["ignored_features"];
        if (ignoredFeatures.IsArray()) {
            NCatboostOptions::TJsonFieldHelper<TVector<int>>::Read(ignoredFeatures, &result);
        }
    }
    return result;
}
