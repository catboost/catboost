#pragma once

#include "statistics_data_structures.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/visitor.h>
#include <catboost/private/libs/options/restrictions.h>

#include <library/cpp/json/writer/json_value.h>

#include <util/generic/vector.h>
#include <util/stream/fwd.h>

using namespace NCB;

class TDatasetStatisticsProviderBuilder: public IRawObjectsOrderDataVisitor {
public:
    TDatasetStatisticsProviderBuilder(
        const TDataProviderBuilderOptions& options,
        NPar::ILocalExecutor* localExecutor)
        : InBlock(false)
        , ObjectCount(0)
        , NextCursor(0)
        , Options(options)
        , LocalExecutor(localExecutor)
        , InProcess(false)
        , ResultTaken(false)
    {
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
        InProcess = true;
        ResultTaken = false;

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

        MetaInfo = metaInfo;
        if (haveUnknownNumberOfSparseFeatures) {
            // make a copy because it can be updated
            MetaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(*metaInfo.FeaturesLayout);
        }

        FeatureStatistics.Init(MetaInfo);
        TargetsStatistics.Init(MetaInfo);

        CatFeatureCount = (size_t)metaInfo.FeaturesLayout->GetCatFeatureCount();
    }

    void StartNextBlock(ui32 blockSize) override {
        NextCursor += blockSize;
    }

    // TCommonObjectsData
    void AddGroupId(ui32 localObjectIdx, TGroupId value) override {
        Y_UNUSED(localObjectIdx, value);
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
        FeatureStatistics
            .FloatFeatureStatistics[GetInternalFeatureIdx<EFeatureType::Float>(flatFeatureIdx)]
            .Update(feature);
        Y_UNUSED(localObjectIdx);
    }
    void AddAllFloatFeatures(ui32 localObjectIdx, TConstArrayRef<float> features) override {
        for (auto perTypeFeatureIdx : xrange(features.size())) {
            FeatureStatistics
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
        ui32 catFeatureIdx = GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx);

        int hashPartIdx = LocalExecutor->GetWorkerThreadId();
        CB_ENSURE(hashPartIdx < CB_THREAD_LIMIT, "Internal error: thread ID exceeds CB_THREAD_LIMIT");
        auto& catFeatureHashes = HashMapParts[hashPartIdx].CatFeatureHashes;
        catFeatureHashes.resize(CatFeatureCount);

        auto& catFeatureHash = catFeatureHashes[catFeatureIdx];

        ui32 hashedValue = CalcCatFeatureHash(feature);
        THashMap<ui32, TString>::insert_ctx insertCtx;
        if (!catFeatureHash.contains(hashedValue, insertCtx)) {
            catFeatureHash.emplace_direct(insertCtx, hashedValue, TString(feature));
        }
        return hashedValue;
    }

    // localObjectIdx may be used as hint for sampling
    ui32 GetCatFeatureValue(ui32 /* localObjectIdx */, ui32 flatFeatureIdx, TStringBuf feature) override {
        return GetCatFeatureValue(flatFeatureIdx, feature);
    }

    void AddCatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) override {
        // ToDo Implement CatFeatureStatistics MLTOOLS-6678
        // FeatureStatistics
        //     .CatFeatureStatistics[GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx)]
        //     .Update(feature);
        Y_UNUSED(localObjectIdx, flatFeatureIdx, feature);
    }
    void AddAllCatFeatures(ui32 localObjectIdx, TConstArrayRef<ui32> features) override {
        // ToDo Implement CatFeatureStatistics MLTOOLS-6678
        // for (auto perTypeFeatureIdx : xrange(features.size())) {
        //     FeatureStatistics
        //         .CatFeatureStatistics[TCatFeatureIdx(perTypeFeatureIdx).Idx]
        //         .Update(features[perTypeFeatureIdx]);
        // }
        Y_UNUSED(localObjectIdx, features);
    }

    void AddAllCatFeatures(
        ui32 localObjectIdx,
        TConstPolymorphicValuesSparseArray<ui32, ui32> features) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(localObjectIdx, features);
    }

    // for sparse data
    void AddCatFeatureDefaultValue(ui32 flatFeatureIdx, TStringBuf feature) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(flatFeatureIdx, feature);
    }

    void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(flatFeatureIdx, localObjectIdx, feature);
    }
    void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, const TString& feature) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(flatFeatureIdx, localObjectIdx, feature);
    }
    void AddAllTextFeatures(ui32 localObjectIdx, TConstArrayRef<TString> features) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(localObjectIdx, features);
    }
    void AddAllTextFeatures(
        ui32 localObjectIdx,
        TConstPolymorphicValuesSparseArray<TString, ui32> features) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(localObjectIdx, features);
    }

    void AddEmbeddingFeature(
        ui32 localObjectIdx,
        ui32 flatFeatureIdx,
        TMaybeOwningConstArrayHolder<float> feature) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(flatFeatureIdx, localObjectIdx, feature);
    }

    // TRawTargetData

    void AddTarget(ui32 localObjectIdx, const TString& value) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(localObjectIdx, value);
    }
    void AddTarget(ui32 localObjectIdx, float value) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(localObjectIdx, value);
    }
    void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, const TString& value) override {
        TargetsStatistics.Update(flatTargetIdx, value);
        Y_UNUSED(localObjectIdx);
    }
    void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, float value) override {
        TargetsStatistics.Update(flatTargetIdx, value);
        Y_UNUSED(localObjectIdx);
    }
    void AddBaseline(ui32 localObjectIdx, ui32 baselineIdx, float value) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(baselineIdx, localObjectIdx, value);
    }
    void AddWeight(ui32 localObjectIdx, float value) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(localObjectIdx, value);
    }
    void AddGroupWeight(ui32 localObjectIdx, float value) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(localObjectIdx, value);
    }

    void Finish() override {
        CB_ENSURE(InProcess, "Attempt to Finish without starting processing");
        CB_ENSURE(
            NextCursor >= ObjectCount,
            "processed object count is less than than specified in metadata");

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
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(groupWeights);
    }

    // separate method because they can be loaded from a separate data source
    void SetBaseline(TVector<TVector<float>>&& baseline) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(baseline);
    }

    void SetPairs(TRawPairsData&& pairs) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(pairs);
    }

    TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const override {
        return Nothing();
    }

    void SetTimestamps(TVector<ui64>&& timestamps) override {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(timestamps);
    }

private:
    template <EFeatureType FeatureType>
    ui32 GetInternalFeatureIdx(ui32 flatFeatureIdx) const {
        return MetaInfo.FeaturesLayout->GetExpandingInternalFeatureIdx<FeatureType>(flatFeatureIdx).Idx;
    }

    void GetResult() {
        if (CatFeatureCount) {
            if (!CatFeaturesHashToString) {
                CatFeaturesHashToString = MakeAtomicShared<TVector<THashMap<ui32, TString>>>();
            }
            CatFeaturesHashToString->resize(CatFeatureCount);
            for (const auto& part : HashMapParts) {
                if (part.CatFeatureHashes.empty()) {
                    continue;
                }
                for (auto catFeatureIdx : xrange(CatFeatureCount)) {
                    (*CatFeaturesHashToString)[catFeatureIdx].insert(
                        part.CatFeatureHashes[catFeatureIdx].begin(),
                        part.CatFeatureHashes[catFeatureIdx].end());
                }
            }
        }
    }

private:
    bool InBlock;
    ui32 ObjectCount;
    ui32 NextCursor;

    TDataProviderBuilderOptions Options;
    NPar::ILocalExecutor* LocalExecutor;

    bool InProcess;
    bool ResultTaken;

    struct THashPart {
        TVector<THashMap<ui32, TString>> CatFeatureHashes;
    };
    std::array<THashPart, CB_THREAD_LIMIT> HashMapParts;
    TAtomicSharedPtr<TVector<THashMap<ui32, TString>>> CatFeaturesHashToString; // [catFeatureIdx]

    TDataMetaInfo MetaInfo;
    ui32 CatFeatureCount;
    TFeatureStatistics FeatureStatistics;
    TTargetsStatistics TargetsStatistics;
    // ToDo: maybe add GroupStatistics

public:
    void OutputResult(const TString& outputPath) const;

    NJson::TJsonValue GetResult() const;
};
