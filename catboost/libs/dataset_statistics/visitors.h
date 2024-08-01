#pragma once

#include "statistics_data_structures.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/visitor.h>
#include <catboost/private/libs/options/restrictions.h>

#include <library/cpp/json/writer/json_value.h>

#include <util/digest/numeric.h>
#include <util/generic/vector.h>
#include <util/stream/fwd.h>


namespace NCB {

class TDatasetStatisticsFullVisitor final : public IRawObjectsOrderDataVisitor {
public:
    TDatasetStatisticsFullVisitor(
        const TDataProviderBuilderOptions& options,
        bool isLocal,
        NPar::ILocalExecutor* /*localExecutor*/
    )
        : InBlock(false)
        , ObjectCount(0)
        , NextCursor(0)
        , Options(options)
        , InProcess(false)
        , ResultTaken(false)
        , IsLocal(isLocal)
    {}

    void SetCustomBorders(
        const TFeatureCustomBorders& customBorders,
        const TFeatureCustomBorders& targetCustomBorders
    ) {
        CustomBorders = customBorders;
        TargetCustomBorders = targetCustomBorders;
    }

    void SetConvertStringTargets(bool convertStringTargets) {
        ConvertStringTargets = convertStringTargets;
    }

    void Start(
        bool inBlock, // subset processing - Start/Finish is called for each block
        const TDataMetaInfo& metaInfo,
        bool haveUnknownNumberOfSparseFeatures,
        ui32 objectCount,
        EObjectsOrder /* objectsOrder */,
        TVector<TIntrusivePtr<IResourceHolder>> /* resourceHolders */
    ) override {
        CB_ENSURE(!InProcess, "Attempt to start new processing without finishing the last");
        CB_ENSURE(!haveUnknownNumberOfSparseFeatures, "Not supported");
        InProcess = true;
        ResultTaken = false;
        MetaInfo = metaInfo;

        InBlock = inBlock;

        ui32 prevTailSize = 0;
        if (InBlock) {
            CB_ENSURE(!metaInfo.HasPairs, "Pairs are not supported in block processing");

            prevTailSize = (NextCursor < ObjectCount) ? (ObjectCount - NextCursor) : 0;
            NextCursor = prevTailSize;
        } else {
            NextCursor = 0;
        }
        ObjectCount = objectCount + prevTailSize;
        CatFeatureCount = (size_t)metaInfo.FeaturesLayout->GetCatFeatureCount();
        if (MetaInfo.TargetType == ERawTargetType::String && ConvertStringTargets) {
            MetaInfo.TargetType = ERawTargetType::Float;
        }
        DatasetStatistics.Init(MetaInfo, CustomBorders, TargetCustomBorders);
//        MetaInfo.TargetType = ERawTargetType::String;
        FloatTarget.resize(metaInfo.TargetCount);
    }

    void StartNextBlock(ui32 blockSize) override {
        NextCursor += blockSize;
        if (DatasetStatistics.GroupwiseStats.Defined()) {
            DatasetStatistics.GroupwiseStats->Flush();
        }
    }

    // TCommonObjectsData
    void AddGroupId(ui32 localObjectIdx, TGroupId value) override {
        Y_UNUSED(localObjectIdx);
        DatasetStatistics.GroupwiseStats->Update(value);
    }
    void AddSubgroupId(ui32 localObjectIdx, TSubgroupId value) override {
        Y_UNUSED(localObjectIdx, value);
    }
    void AddTimestamp(ui32 localObjectIdx, ui64 value) override {
        Y_UNUSED(localObjectIdx, value);
    }
    void AddGroupId(ui32 localObjectIdx, const TString& value) override {
        Y_UNUSED(localObjectIdx, value);
    }
    void AddSubgroupId(ui32 localObjectIdx, const TString& value) override {
        Y_UNUSED(localObjectIdx, value);
    }
    void AddSampleId(ui32 localObjectIdx, const TString& value) override {
        DatasetStatistics.SampleIdStatistics.Update(value);
        Y_UNUSED(localObjectIdx);
    }

    // TRawObjectsData
    void AddFloatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, float feature) override {
        Y_ASSERT(false);
        DatasetStatistics.FeatureStatistics
            .FloatFeatureStatistics[GetInternalFeatureIdx<EFeatureType::Float>(flatFeatureIdx)]
            .Update(feature);
        Y_UNUSED(localObjectIdx);
    }
    void AddAllFloatFeatures(ui32 localObjectIdx, TConstArrayRef<float> features) override {
        for (auto perTypeFeatureIdx : xrange(features.size())) {
            DatasetStatistics.FeatureStatistics
                .FloatFeatureStatistics[TFloatFeatureIdx(perTypeFeatureIdx).Idx]
                .Update(features[perTypeFeatureIdx]);
        }
        Y_UNUSED(localObjectIdx);
    }
    void AddAllFloatFeatures(
        ui32 localObjectIdx,
        TConstPolymorphicValuesSparseArray<float, ui32> features) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(localObjectIdx, features);
    }

    // for sparse float features default value is always assumed to be 0.0f

    ui32 GetCatFeatureValue(ui32 flatFeatureIdx, TStringBuf feature) override {
        ui32 hashedValue = CalcCatFeatureHash(feature);
        Y_UNUSED(flatFeatureIdx);
        return hashedValue;
    }

    // localObjectIdx may be used as hint for sampling
    ui32 GetCatFeatureValue(ui32 /* localObjectIdx */, ui32 flatFeatureIdx, TStringBuf feature) override {
        return GetCatFeatureValue(flatFeatureIdx, feature);
    }

    void AddCatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) override {
        // ToDo Implement CatFeatureStatistics MLTOOLS-6678
         DatasetStatistics.FeatureStatistics
             .CatFeatureStatistics[GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx)]
             .Update(feature);
        Y_UNUSED(localObjectIdx);
    }
    void AddAllCatFeatures(ui32 localObjectIdx, TConstArrayRef<ui32> features) override {
        // ToDo Implement CatFeatureStatistics MLTOOLS-6678
        for (auto perTypeFeatureIdx : xrange(features.size())) {
            DatasetStatistics.FeatureStatistics
                .CatFeatureStatistics[TCatFeatureIdx(perTypeFeatureIdx).Idx]
                .Update(features[perTypeFeatureIdx]);
        }
        Y_UNUSED(localObjectIdx);
    }

    void AddAllCatFeatures(
        ui32 localObjectIdx,
        TConstPolymorphicValuesSparseArray<ui32, ui32> features
    ) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(localObjectIdx, features);
    }

    // for sparse data
    void AddCatFeatureDefaultValue(ui32 flatFeatureIdx, TStringBuf feature) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(flatFeatureIdx, feature);
    }

    void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) override {
        DatasetStatistics.FeatureStatistics.TextFeatureStatistics[
            GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx)
        ].Update(feature);
        Y_UNUSED(localObjectIdx);
    }
    void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, const TString& feature) override {
        DatasetStatistics.FeatureStatistics.TextFeatureStatistics[
            GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx)
        ].Update(feature);
        Y_UNUSED(localObjectIdx);
    }
    void AddAllTextFeatures(ui32 localObjectIdx, TConstArrayRef<TString> features) override {
        Y_UNUSED(localObjectIdx, features);
    }
    void AddAllTextFeatures(
        ui32 localObjectIdx,
        TConstPolymorphicValuesSparseArray<TString, ui32> features
    ) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(localObjectIdx, features);
    }

    void AddEmbeddingFeature(
        ui32 localObjectIdx,
        ui32 flatFeatureIdx,
        TMaybeOwningConstArrayHolder<float> feature
    ) override {
        Y_UNUSED(flatFeatureIdx, localObjectIdx, feature);
    }

    // TRawTargetData

    void AddTarget(ui32 localObjectIdx, const TString& value) override {
        if (!ConvertStringTargets) {
            DatasetStatistics.TargetsStatistics.Update(/* flatTargetIdx */ 0, value);
        } else {
            float fValue = FromString<float>(value);
            DatasetStatistics.TargetsStatistics.Update(/* flatTargetIdx */ 0, fValue);
            with_lock(TargetLock) {
                FloatTarget[0].push_back(fValue);
            }
        }
        Y_UNUSED(localObjectIdx);
    }
    void AddTarget(ui32 localObjectIdx, float value) override {
        if (MetaInfo.TargetType == ERawTargetType::Float) {
            DatasetStatistics.TargetsStatistics.Update(/* flatTargetIdx */ 0, value);
            with_lock(TargetLock) {
                FloatTarget[0].push_back(value);
            }
        } else {
            DatasetStatistics.TargetsStatistics.Update(/* flatTargetIdx */ 0, ui32(value));
        }
        Y_UNUSED(localObjectIdx);
    }
    void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, const TString& value) override {
        if (!ConvertStringTargets) {
            DatasetStatistics.TargetsStatistics.Update(flatTargetIdx, value);
        } else {
            float fValue = FromString<float>(value);
            DatasetStatistics.TargetsStatistics.Update(flatTargetIdx, fValue);
            with_lock(TargetLock) {
                FloatTarget[0].push_back(fValue);
            }
        }
        Y_UNUSED(localObjectIdx);
    }
    void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, float value) override {
        if (MetaInfo.TargetType == ERawTargetType::Float) {
            DatasetStatistics.TargetsStatistics.Update(flatTargetIdx, value);
            with_lock(TargetLock) {
                FloatTarget[flatTargetIdx].push_back(value);
            }
        } else {
            DatasetStatistics.TargetsStatistics.Update(flatTargetIdx, ui32(value));
        }
        Y_UNUSED(localObjectIdx);
    }
    void AddBaseline(ui32 localObjectIdx, ui32 baselineIdx, float value) override {
        Y_UNUSED(baselineIdx, localObjectIdx, value);
    }
    void AddWeight(ui32 localObjectIdx, float value) override {
        Y_UNUSED(localObjectIdx, value);
    }
    void AddGroupWeight(ui32 localObjectIdx, float value) override {
        Y_UNUSED(localObjectIdx, value);
    }

    void Finish() override {
        CB_ENSURE(InProcess, "Attempt to Finish without starting processing");
        CB_ENSURE(
            !IsLocal || NextCursor >= ObjectCount,
            "processed object count is less than than specified in metadata: " << NextCursor << "<"
            << ObjectCount);
        if (IsLocal) {
            DatasetStatistics.ObjectsCount = ObjectCount;
        } else {
            DatasetStatistics.ObjectsCount = NextCursor;
        }

        if (DatasetStatistics.GroupwiseStats.Defined()) {
            DatasetStatistics.GroupwiseStats->Flush();
        }

        if (ObjectCount != 0) {
            CATBOOST_INFO_LOG << "Object info sizes: " << ObjectCount << " "
                << MetaInfo.FeaturesLayout->GetExternalFeatureCount() << Endl;
        } else {
            // should this be an error?
            CATBOOST_ERROR_LOG << "No objects info loaded" << Endl;
        }

        InProcess = false;
    }

    // IDatasetVisitor

    void SetGroupWeights(TVector<float>&& groupWeights) override {
        Y_UNUSED(groupWeights);
    }

    // separate method because they can be loaded from a separate data source
    void SetBaseline(TVector<TVector<float>>&& baseline) override {
        Y_UNUSED(baseline);
    }

    void SetPairs(TRawPairsData&& pairs) override {
        Y_UNUSED(pairs);
    }

    void SetGraph(TRawPairsData&& pairs) override {
        Y_UNUSED(pairs);
        CB_ENSURE_INTERNAL(false, "Unsupported");
    }

    TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const override {
        return Nothing();
    }

    void SetTimestamps(TVector<ui64>&& timestamps) override {
        Y_UNUSED(timestamps);
    }

    void OutputResult(const TString& outputPath) const;

    const TDatasetStatistics& GetDatasetStatistics() const;
    const TVector<TVector<float>>& GetFloatTarget() const {
        return FloatTarget;
    }

private:
    template <EFeatureType FeatureType>
    ui32 GetInternalFeatureIdx(ui32 flatFeatureIdx) const {
        return MetaInfo.FeaturesLayout->GetExpandingInternalFeatureIdx<FeatureType>(flatFeatureIdx).Idx;
    }

private:
    bool InBlock;
    ui32 ObjectCount;
    ui32 NextCursor;

    TDataProviderBuilderOptions Options;

    bool InProcess;
    bool ResultTaken;

    bool IsLocal;

    TDatasetStatistics DatasetStatistics;
    TVector<TVector<float>> FloatTarget;
    TMutex TargetLock;
    TDataMetaInfo MetaInfo;
    ui32 CatFeatureCount;

    TFeatureCustomBorders CustomBorders;
    TFeatureCustomBorders TargetCustomBorders;
    bool ConvertStringTargets;
};

class TDatasetStatisticsOnlyGroupVisitor final : public IRawObjectsOrderDataVisitor {
public:
    explicit TDatasetStatisticsOnlyGroupVisitor(bool isLocal)
        : ObjectCount(0)
        , NextCursor(0)
        , InProcess(false)
        , IsLocal(isLocal)
    {}

    void Start(
        bool inBlock, // subset processing - Start/Finish is called for each block
        const TDataMetaInfo& /*metaInfo*/,
        bool /*haveUnknownNumberOfSparseFeatures*/,
        ui32 objectCount,
        EObjectsOrder /* objectsOrder */,
        TVector<TIntrusivePtr<IResourceHolder>> /* resourceHolders */
    ) override {
        CB_ENSURE(!InProcess, "Attempt to start new processing without finishing the last");
        InProcess = true;

        ui32 prevTailSize = 0;
        if (inBlock) {
            prevTailSize = (NextCursor < ObjectCount) ? (ObjectCount - NextCursor) : 0;
            NextCursor = prevTailSize;
        } else {
            NextCursor = 0;
        }
        ObjectCount = objectCount + prevTailSize;
    }

    void StartNextBlock(ui32 blockSize) override {
        NextCursor += blockSize;
        GroupwiseStats.Flush();
    }

    // TCommonObjectsData
    void AddGroupId(ui32 localObjectIdx, TGroupId value) override {
        Y_UNUSED(localObjectIdx);
        GroupwiseStats.Update(value);
    }
    void AddSubgroupId(ui32 localObjectIdx, TSubgroupId value) override {
        Y_UNUSED(localObjectIdx, value);
    }
    void AddTimestamp(ui32 localObjectIdx, ui64 value) override {
        Y_UNUSED(localObjectIdx, value);
    }
    void AddGroupId(ui32 localObjectIdx, const TString& value) override {
        Y_UNUSED(localObjectIdx, value);
    }
    void AddSubgroupId(ui32 localObjectIdx, const TString& value) override {
        Y_UNUSED(localObjectIdx, value);
    }
    void AddSampleId(ui32 localObjectIdx, const TString& value) override {
        Y_UNUSED(localObjectIdx, value);
    }

    // TRawObjectsData
    void AddFloatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, float feature) override {
        Y_UNUSED(localObjectIdx, flatFeatureIdx, feature);
    }
    void AddAllFloatFeatures(ui32 localObjectIdx, TConstArrayRef<float> features) override {
        Y_UNUSED(localObjectIdx, features);
    }
    void AddAllFloatFeatures(
        ui32 localObjectIdx,
        TConstPolymorphicValuesSparseArray<float, ui32> features) override {
        Y_UNUSED(localObjectIdx, features);
    }

    // for sparse float features default value is always assumed to be 0.0f

    ui32 GetCatFeatureValue(ui32 flatFeatureIdx, TStringBuf feature) override {
        Y_UNUSED(flatFeatureIdx, feature);
        return 0;
    }

    // localObjectIdx may be used as hint for sampling
    ui32 GetCatFeatureValue(ui32 /* localObjectIdx */, ui32 flatFeatureIdx, TStringBuf feature) override {
        Y_UNUSED(flatFeatureIdx, feature);
        return 0;
    }

    void AddCatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) override {
        Y_UNUSED(localObjectIdx, flatFeatureIdx, feature);
    }
    void AddAllCatFeatures(ui32 localObjectIdx, TConstArrayRef<ui32> features) override {
        Y_UNUSED(localObjectIdx, features);
    }

    void AddAllCatFeatures(
        ui32 localObjectIdx,
        TConstPolymorphicValuesSparseArray<ui32, ui32> features) override {
        Y_UNUSED(localObjectIdx, features);
    }

    // for sparse data
    void AddCatFeatureDefaultValue(ui32 flatFeatureIdx, TStringBuf feature) override {
        Y_UNUSED(flatFeatureIdx, feature);
    }

    void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) override {
        Y_UNUSED(localObjectIdx, flatFeatureIdx, feature);
    }
    void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, const TString& feature) override {
        Y_UNUSED(localObjectIdx, flatFeatureIdx, feature);
    }
    void AddAllTextFeatures(ui32 localObjectIdx, TConstArrayRef<TString> features) override {
        Y_UNUSED(localObjectIdx, features);
    }
    void AddAllTextFeatures(
        ui32 localObjectIdx,
        TConstPolymorphicValuesSparseArray<TString, ui32> features
    ) override {
        Y_UNUSED(localObjectIdx, features);
    }

    void AddEmbeddingFeature(
        ui32 localObjectIdx,
        ui32 flatFeatureIdx,
        TMaybeOwningConstArrayHolder<float> feature
    ) override {
        Y_UNUSED(flatFeatureIdx, localObjectIdx, feature);
    }

    // TRawTargetData

    void AddTarget(ui32 localObjectIdx, const TString& value) override {
        Y_UNUSED(localObjectIdx, value);
    }
    void AddTarget(ui32 localObjectIdx, float value) override {
        Y_UNUSED(localObjectIdx, value);
    }
    void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, const TString& value) override {
        Y_UNUSED(flatTargetIdx, localObjectIdx, value);
    }
    void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, float value) override {
        Y_UNUSED(flatTargetIdx, localObjectIdx, value);
    }
    void AddBaseline(ui32 localObjectIdx, ui32 baselineIdx, float value) override {
        Y_UNUSED(baselineIdx, localObjectIdx, value);
    }
    void AddWeight(ui32 localObjectIdx, float value) override {
        Y_UNUSED(localObjectIdx, value);
    }
    void AddGroupWeight(ui32 localObjectIdx, float value) override {
        Y_UNUSED(localObjectIdx, value);
    }

    void Finish() override {
        CB_ENSURE(InProcess, "Attempt to Finish without starting processing");
        CB_ENSURE(
            !IsLocal || NextCursor >= ObjectCount,
            "processed object count is less than than specified in metadata: " << NextCursor << "<"
            << ObjectCount);

        GroupwiseStats.Flush();

        if (ObjectCount != 0) {
            CATBOOST_INFO_LOG << "Objects processed: " << ObjectCount << Endl;
        } else {
            // should this be an error?
            CATBOOST_ERROR_LOG << "No objects processed" << Endl;
        }

        InProcess = false;
    }

    // IDatasetVisitor

    void SetGroupWeights(TVector<float>&& groupWeights) override {
        Y_UNUSED(groupWeights);
    }

    // separate method because they can be loaded from a separate data source
    void SetBaseline(TVector<TVector<float>>&& baseline) override {
        Y_UNUSED(baseline);
    }

    void SetPairs(TRawPairsData&& pairs) override {
        Y_UNUSED(pairs);
    }

    void SetGraph(TRawPairsData&& pairs) override {
        Y_UNUSED(pairs);
        CB_ENSURE(false, "Unsupported");
    }

    TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const override {
        return Nothing();
    }

    void SetTimestamps(TVector<ui64>&& timestamps) override {
        Y_UNUSED(timestamps);
    }

    void OutputResult(const TString& outputPath) const;

    const TGroupwiseStats& GetGroupwiseStats() const;

private:
    ui32 ObjectCount;
    ui32 NextCursor;

    bool InProcess;

    bool IsLocal;

    TGroupwiseStats GroupwiseStats;
};

}
