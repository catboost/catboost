#pragma once

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/model/features.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/vector.h>
#include <util/string/vector.h>

struct TPoolColumnsMetaInfo {
    TVector<TColumn> Columns;

    ui32 CountColumns(const EColumn columnType) const;
    TVector<int> GetCategFeatures() const;
    void Validate() const;
    TVector<TString> GenerateFeatureIds(const TMaybe<TString>& header, char fieldDelimiter) const;
};

struct TPoolMetaInfo {
    ui32 FeatureCount = 0;
    ui32 BaselineCount = 0;

    bool HasGroupId = false;
    bool HasGroupWeight = false;
    bool HasSubgroupIds = false;
    bool HasDocIds = false;
    bool HasWeights = false;
    bool HasTimestamp = false;

    // set only for dsv format pools
    // TODO(akhropov): temporary, serialization details shouldn't be here
    TMaybe<TPoolColumnsMetaInfo> ColumnsInfo;

    TPoolMetaInfo() = default;

    explicit TPoolMetaInfo(TVector<TColumn>&& columns);

    void Swap(TPoolMetaInfo& other) {
        std::swap(FeatureCount, other.FeatureCount);
        std::swap(BaselineCount, other.BaselineCount);
        std::swap(HasGroupId, other.HasGroupId);
        std::swap(HasGroupWeight, other.HasGroupWeight);
        std::swap(HasSubgroupIds, other.HasSubgroupIds);
        std::swap(HasDocIds, other.HasDocIds);
        std::swap(HasWeights, other.HasWeights);
        std::swap(HasTimestamp, other.HasTimestamp);
        std::swap(ColumnsInfo, other.ColumnsInfo);
    }
};

namespace NCB {
    class IPoolBuilder {
    public:
        virtual void Start(const TPoolMetaInfo& poolMetaInfo,
                           int docCount,
                           const TVector<int>& catFeatureIds) = 0;
        virtual void StartNextBlock(ui32 blockSize) = 0;

        virtual float GetCatFeatureValue(const TStringBuf& feature) = 0;
        virtual void AddCatFeature(ui32 localIdx, ui32 featureId, const TStringBuf& feature) = 0;
        virtual void AddFloatFeature(ui32 localIdx, ui32 featureId, float feature) = 0;
        virtual void AddBinarizedFloatFeature(ui32 localIdx, ui32 featureId, ui8 binarizedFeature) = 0;
        virtual void AddAllFloatFeatures(ui32 localIdx, TConstArrayRef<float> features) = 0;
        virtual void AddTarget(ui32 localIdx, float value) = 0;
        virtual void AddWeight(ui32 localIdx, float value) = 0;
        virtual void AddQueryId(ui32 localIdx, TGroupId value) = 0;
        virtual void AddSubgroupId(ui32 localIdx, TSubgroupId value) = 0;
        virtual void AddBaseline(ui32 localIdx, ui32 offset, double value) = 0;
        virtual void AddDocId(ui32 localIdx, const TStringBuf& value) = 0;
        virtual void AddTimestamp(ui32 localIdx, ui64 value) = 0;
        virtual void SetFeatureIds(const TVector<TString>& featureIds) = 0;
        virtual void SetPairs(const TVector<TPair>& pairs) = 0;
        virtual void SetFloatFeatures(const TVector<TFloatFeature>& floatFeatures) = 0;
        virtual int GetDocCount() const = 0;
        virtual TConstArrayRef<float> GetWeight() const = 0;
        virtual void GenerateDocIds(int offset) = 0;
        virtual void Finish() = 0;
        virtual ~IPoolBuilder() = default;
    };
}
