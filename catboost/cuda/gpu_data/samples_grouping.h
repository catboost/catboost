#pragma once

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/slice.h>
#include <catboost/cuda/data/data_utils.h>
#include <catboost/libs/data_new/objects_grouping.h>
#include <catboost/libs/data_types/query.h>
#include <util/system/types.h>
#include <util/generic/vector.h>

namespace NCatboostCuda {
    class IQueriesGrouping {
    public:
        virtual ~IQueriesGrouping() {
        }
        virtual ui32 GetQueryCount() const = 0;

        virtual ui32 GetQueryOffset(ui32 id) const = 0;

        virtual ui32 GetQuerySize(ui32 id) const = 0;

        virtual ui32 GetQueryId(size_t line) const = 0;

        virtual ui32 NextQueryOffsetForLine(ui32 line) const = 0;
    };

    class TWithoutQueriesGrouping: public IQueriesGrouping {
    public:
        TWithoutQueriesGrouping(ui32 docCount)
            : DocCount(docCount)
        {
        }

        ui32 GetQueryCount() const override {
            return DocCount;
        }

        ui32 GetQueryOffset(ui32 id) const override {
            return Min<ui32>(id, DocCount);
        }

        ui32 GetQuerySize(ui32 id) const override {
            Y_UNUSED(id);
            return 1;
        }

        ui32 GetQueryId(size_t line) const override {
            return Min<size_t>(line, DocCount);
        }

        ui32 NextQueryOffsetForLine(ui32 line) const override {
            return Min<ui32>(line + 1, DocCount);
        }

    private:
        ui32 DocCount;
    };

    //zero-based group indices
    class TQueriesGrouping: public IQueriesGrouping {
    public:
        TQueriesGrouping(TConstArrayRef<ui32> objectPermutation,
                         const NCB::TObjectsGrouping& objectsGrouping,
                         TConstArrayRef<TQueryInfo> groupInfos,
                         bool hasPairs) {
            QueryIds.resize(objectPermutation.size());

            size_t atLeastTwoDocQueriesCount = 0;

            {
                ui32 groupIdx = 0;
                ui32 groupStartIdx = 0;
                while (groupStartIdx < objectPermutation.size()) {
                    ui32 srcGroupIdx = objectsGrouping.GetGroupIdxForObject(objectPermutation[groupStartIdx]);
                    const auto& groupInfo = groupInfos[srcGroupIdx];
                    CB_ENSURE(groupInfo.GetSize(), "Error: empty group");
                    QuerySizes.push_back(groupInfo.GetSize());
                    QueryOffsets.push_back(groupStartIdx);
                    for (ui32 j = 0; j < groupInfo.GetSize(); ++j) {
                        QueryIds[groupStartIdx + j] = groupIdx;
                    }
                    atLeastTwoDocQueriesCount += groupInfo.GetSize() > 1;

                    if (hasPairs) {
                        QueryPairOffsets.push_back(FlatQueryPairs.size());
                        for (auto winnerId : xrange(groupInfo.Competitors.size())) {
                            for (const auto& localPair : groupInfo.Competitors[winnerId]) {
                                uint2 gpuPair;
                                gpuPair.x = groupStartIdx + winnerId;
                                gpuPair.y = groupStartIdx + localPair.Id;
                                FlatQueryPairs.push_back(gpuPair);
                                QueryPairWeights.push_back(localPair.Weight);
                            }
                        }
                    }
                    groupStartIdx += groupInfo.GetSize();
                    ++groupIdx;
                }
            }
            CB_ENSURE(atLeastTwoDocQueriesCount, "Error: all groups have size 1");
        }

        ui32 GetQueryCount() const override {
            return QuerySizes.size();
        }

        ui32 GetQueryOffset(ui32 id) const override {
            return id < QueryOffsets.size() ? QueryOffsets[id] : QueryIds.size();
        }

        ui32 GetQuerySize(ui32 id) const override {
            return QuerySizes[id];
        }

        ui32 GetQueryId(size_t line) const override {
            return line < QueryIds.size() ? QueryIds[line] : QuerySizes.size();
        }

        ui32 NextQueryOffsetForLine(ui32 line) const override {
            ui32 gid = GetQueryId(line);
            if (gid + 1 < QueryOffsets.size()) {
                return QueryOffsets[gid + 1];
            }
            return QueryIds.size();
        }

        ui32 GetQueryPairOffset(ui32 groupId) const {
            return groupId < QueryPairOffsets.size() ? QueryPairOffsets[groupId] : FlatQueryPairs.size();
        };

        const TVector<uint2>& GetFlatQueryPairs() const {
            return FlatQueryPairs;
        }

        const TVector<float>& GetQueryPairWeights() const {
            return QueryPairWeights;
        }

        void SetSubgroupIds(TVector<ui32>&& groupIds) {
            SubgroupIds = std::move(groupIds);
        }

        bool HasSubgroupIds() const {
            return SubgroupIds.size();
        }

        const ui32* GetSubgroupIds(ui32 queryId) const {
            CB_ENSURE(HasSubgroupIds());
            auto offset = GetQueryOffset(queryId);
            return SubgroupIds.data() + offset;
        }

    private:
        TVector<ui32> QuerySizes;
        TVector<ui32> QueryOffsets;
        //doc to query ids map
        TVector<ui32> QueryIds;
        TVector<ui32> SubgroupIds;

        TVector<ui32> QueryPairOffsets;
        TVector<uint2> FlatQueryPairs;
        TVector<float> QueryPairWeights;
    };

}
