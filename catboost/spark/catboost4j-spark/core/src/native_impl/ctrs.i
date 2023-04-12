%{
#include <catboost/libs/data/ctrs.h>
#include <catboost/spark/catboost4j-spark/core/src/native_impl/ctrs.h>
%}

%include "data_provider.i"
%include "string.i"


%catches(yexception) NCB::TPrecomputedOnlineCtrMetaData::Append(NCB::TPrecomputedOnlineCtrMetaData& add);
%catches(yexception) NCB::TPrecomputedOnlineCtrMetaData::SerializeToJson() const;
%catches(yexception) NCB::TPrecomputedOnlineCtrMetaData::DeserializeFromJson(
    const TString& serializedJson
);

namespace NCB {
    struct TPrecomputedOnlineCtrMetaData {
    public:
        void Append(TPrecomputedOnlineCtrMetaData& add);

        // Use JSON as string to be able to use in JVM binding as well
        TString SerializeToJson() const;
        static TPrecomputedOnlineCtrMetaData DeserializeFromJson(
            const TString& serializedJson
        );
    };
}


%catches(yexception) GetCtrHelper(
    const NCatboostOptions::TCatBoostOptions& catBoostOptions,
    const NCB::TFeaturesLayout& layout,
    const TVector<float>& preprocessedLearnTarget,
    const TVector<i8>& serializedLabelConverter
);

%catches(yexception) ComputeTargetStatsForCtrs(
    const TCtrHelper& ctrHelper,
    const TVector<float>& preprocessedLearnTarget,
    NPar::TLocalExecutor* localExecutor
);

%catches(yexception) ComputeEstimatedCtrFeatures(
    const TCtrHelper& ctrHelper,
    const NCatboostOptions::TCatBoostOptions& catBoostOptions, // actually only catFeatureParams is used
    const TTargetStatsForCtrs& targetStats,
    const NCB::TQuantizedObjectsDataProviderPtr& learnData,
    const TVector<NCB::TQuantizedObjectsDataProviderPtr>& testData,
    NPar::TLocalExecutor* localExecutor,
    NCB::TEstimatedForCPUObjectsDataProviders* outputData,
    NCB::TPrecomputedOnlineCtrMetaData* outputMeta
);

%catches(yexception) TFinalCtrsCalcer::TFinalCtrsCalcer(
    TFullModel* modelWithoutCtrData, // moved into
    const NCatboostOptions::TCatBoostOptions* catBoostOptions,
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
    TVector<float>* preprocessedLearnTarget,  // moved into, can be empty if there's no target data
    TTargetStatsForCtrs* targetStatsForCtrs, // moved into
    const TCtrHelper& ctrHelper,
    NPar::TLocalExecutor* localExecutor
);

%catches(yexception) TFinalCtrsCalcer::GetCatFeatureFlatIndicesUsedForCtrs() const;
%catches(yexception) TFinalCtrsCalcer::ProcessForFeature(
    i32 catFeatureFlatIdx,
    const NCB::TQuantizedObjectsDataProviderPtr& learnData,
    const TVector<NCB::TQuantizedObjectsDataProviderPtr>& testData
);
%catches(yexception) TFinalCtrsCalcer::GetModelWithCtrData();

%include "ctrs.h"

