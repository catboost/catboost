#pragma once

#include "option.h"
#include "system_options.h"
#include "boosting_options.h"
#include "model_based_eval_options.h"
#include "oblivious_tree_options.h"
#include "output_file_options.h"
#include "binarization_options.h"
#include "loss_description.h"
#include "data_processing_options.h"
#include "cat_feature_options.h"
#include "metric_options.h"
#include "text_processing_options.h"
#include "pool_metainfo_options.h"

#include <util/system/types.h>
#include <util/system/datetime.h>

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    class TCatBoostOptions {
    public:
        explicit TCatBoostOptions(ETaskType taskType);

        void Load(const NJson::TJsonValue& options);
        void Save(NJson::TJsonValue* options) const;

        bool operator==(const TCatBoostOptions& rhs) const;

        bool operator!=(const TCatBoostOptions& rhs) const;

        ETaskType GetTaskType() const;

        void Validate() const;

        void SetNotSpecifiedOptionsToDefaults();

    public:
        TOption<TSystemOptions> SystemOptions;
        TOption<TBoostingOptions> BoostingOptions;
        TOption<TObliviousTreeLearnerOptions> ObliviousTreeOptions;
        TOption<TDataProcessingOptions> DataProcessingOptions;
        TOption<TLossDescription> LossFunctionDescription;
        TOption<TCatFeatureParams> CatFeatureParams;
        TOption<NJson::TJsonValue> FlatParams;
        TOption<NJson::TJsonValue> Metadata;
        TOption<TPoolMetaInfoOptions> PoolMetaInfoOptions;

        TOption<ui64> RandomSeed;
        TOption<ELoggingLevel> LoggingLevel;
        TOption<bool> IsProfile;
        TOption<TMetricOptions> MetricOptions;

        TGpuOnlyOption<TModelBasedEvalOptions> ModelBasedEvalOptions;

    private:
        void ValidateCtr(const TCtrDescription& ctr, ELossFunction lossFunction, bool isTreeCtrs) const;

        void SetLeavesEstimationDefault();

        TCtrDescription CreateDefaultCounter(EProjectionType projectionType) const;

        void SetCtrDefaults();

        void SetDefaultPriorsIfNeeded(TVector<TCtrDescription>& ctrs) const;

        void ValidateCtrs(
            const TVector<TCtrDescription>& ctrDescription,
            ELossFunction lossFunction,
            bool isTreeCtrs) const;

    private:
        TOption<ETaskType> TaskType;
    };

    ETaskType GetTaskType(const NJson::TJsonValue& source);

    ui32 GetThreadCount(const NJson::TJsonValue& source);

    TCatBoostOptions LoadOptions(const NJson::TJsonValue& source);

    bool IsParamsCompatible(TStringBuf firstSerializedParams, TStringBuf secondSerializedParams);

    constexpr bool IsSmallIterationCount(ui32 iterationCount) {
        return iterationCount < 200;
    }
}

using TCatboostOptions = NCatboostOptions::TCatBoostOptions;

TVector<ui32> GetOptionIgnoredFeatures(const NJson::TJsonValue& catBoostJsonOptions);
TVector<ui32> GetOptionFeaturesToEvaluate(const NJson::TJsonValue& catBoostJsonOptions);
