#pragma once

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/slice.h>
#include <catboost/cuda/data/data_utils.h>
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
            return id;
        }

        ui32 GetQuerySize(ui32 id) const override {
            Y_UNUSED(id);
            return 1;
        }

        ui32 GetQueryId(size_t line) const override {
            return line;
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
        TQueriesGrouping(const TVector<TGroupId>& queryIds,
                         const THashMap<TGroupId, TVector<TPair>>& pairs) {
            auto groupedSamples = GroupSamples(queryIds);
            QuerySizes = ComputeGroupSizes(groupedSamples);
            QueryOffsets = ComputeGroupOffsets(groupedSamples);
            QueryIds.resize(queryIds.size());

            TVector<TGroupId> inverseGids;
            {
                ui32 cursor = 0;
                for (ui32 i = 0; i < QuerySizes.size(); ++i) {
                    CB_ENSURE(QuerySizes[i], "Error: empty group");
                    inverseGids.push_back(queryIds[cursor]);
                    for (ui32 j = 0; j < QuerySizes[i]; ++j) {
                        QueryIds[cursor++] = i;
                    }
                }
            }

            if (pairs.size()) {
                for (ui32 i = 0; i < inverseGids.size(); ++i) {
                    TGroupId gid = inverseGids[i];
                    QueryPairOffsets.push_back(FlatQueryPairs.size());
                    if (pairs.has(gid)) {
                        ui32 queryOffset = QueryOffsets[i];

                        const auto& groupPairs = pairs.at(gid);
                        for (auto& localPair : groupPairs) {
                            uint2 gpuPair;
                            gpuPair.x = queryOffset + localPair.WinnerId;
                            gpuPair.y = queryOffset + localPair.LoserId;
                            FlatQueryPairs.push_back(gpuPair);
                            CB_ENSURE(localPair.Weight > 0, "Error: pair weight should be positive " << gpuPair.x << " / " << gpuPair.y);
                            QueryPairWeights.push_back(localPair.Weight);
                        }
                    }
                }
            }
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
            return ~SubgroupIds + offset;
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
