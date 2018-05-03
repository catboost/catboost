#pragma once

#include "full_features.h"

#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/data_types/query.h>
#include <catboost/libs/helpers/query_info_helper.h>

#include <library/binsaver/bin_saver.h>

class TDataset {
public:
    TAllFeatures AllFeatures;
    TVector<TVector<double>> Baseline;
    TVector<float> Target;
    TVector<float> Weights;
    TVector<TGroupId> QueryId;
    TVector<ui32> SubgroupId;
    TVector<TQueryInfo> QueryInfo;
    TVector<TPair> Pairs;

    size_t GetSampleCount() const {
        return Target.size();
    }

    size_t GetQueryCount() const {
        return QueryInfo.size();
    }
    SAVELOAD(AllFeatures, Baseline, Target, Weights, QueryId, SubgroupId, QueryInfo, Pairs);
};

TDataset BuildDataset(const TPool& pool);

inline bool HaveGoodQueryIds(const TDataset& data) {
    return data.QueryId.size() == data.Target.size();
}

/// Update field `QueryInfo` of `data`
inline void UpdateQueryInfo(TDataset* data) {
    UpdateQueriesInfo(data->QueryId, data->SubgroupId, 0, data->GetSampleCount(), &data->QueryInfo);
    UpdateQueriesPairs(data->Pairs, /*invertedPermutation=*/{}, &data->QueryInfo);
}

using TDatasetPtrs = TVector<const TDataset*>;

inline size_t GetSampleCount(const TDatasetPtrs& dataPtrs) {
    size_t totalCount = 0;
    for (const TDataset* dataPtr : dataPtrs) {
        totalCount += dataPtr->GetSampleCount();
    }
    return totalCount;
}

inline bool HaveGoodQueryIds(const TDatasetPtrs& datas) {
    return AllOf(datas, [](const TDataset* data){
        return HaveGoodQueryIds(*data);
    });
}

// @return PrefixSum{ learnSampleCount, testSampleCount[0], ..., testSampleCount[num_tests - 1] }
inline TVector<size_t> CalcTestOffsets(size_t learnSampleCount, const TDatasetPtrs& testDataPtrs) {
    TVector<size_t> testOffsets(testDataPtrs.size() + 1);
    testOffsets[0] = learnSampleCount;
    for (int testIdx = 0; testIdx < testDataPtrs.ysize(); ++testIdx) {
        testOffsets[testIdx + 1] = testOffsets[testIdx] + testDataPtrs[testIdx]->GetSampleCount();
    }
    return testOffsets;
}

