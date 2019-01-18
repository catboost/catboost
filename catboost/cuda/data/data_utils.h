#pragma once

#include <catboost/cuda/cuda_lib/helpers.h>

#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/helpers/cpu_random.h>

#include <util/system/types.h>
#include <util/generic/array_ref.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/random/shuffle.h>

#include <numeric>

namespace NCatboostCuda {
    void GroupSamples(TConstArrayRef<TGroupId> qid, TVector<TVector<ui32>>* qdata);

    inline TVector<TVector<ui32>> GroupSamples(TConstArrayRef<TGroupId> qid) {
        TVector<TVector<ui32>> qdata;
        GroupSamples(qid, &qdata);
        return qdata;
    }

    TVector<ui32> ComputeGroupOffsets(const TVector<TVector<ui32>>& queries);

    template <class TIndicesType>
    inline void Shuffle(ui64 seed, ui32 blockSize, ui32 sampleCount, TVector<TIndicesType>* orderPtr) {
        TRandom rng(seed);
        rng.Advance(10);
        auto& order = *orderPtr;
        order.resize(sampleCount);
        std::iota(order.begin(), order.end(), 0);

        if (blockSize == 1) {
            ::Shuffle(order.begin(), order.begin() + sampleCount, rng);
        } else {
            const auto blocksCount = static_cast<ui32>(::NHelpers::CeilDivide(order.size(), blockSize));
            TVector<ui32> blocks(blocksCount);
            std::iota(blocks.begin(), blocks.end(), 0);
            ::Shuffle(blocks.begin(), blocks.end(), rng);

            ui32 cursor = 0;
            for (ui32 i = 0; i < blocksCount; ++i) {
                const ui32 blockStart = blocks[i] * blockSize;
                const ui32 blockEnd = Min<ui32>(blockStart + blockSize, order.size());
                for (ui32 j = blockStart; j < blockEnd; ++j) {
                    order[cursor++] = j;
                }
            }
        }
    }

    template <class TIndicesType>
    inline void QueryConsistentShuffle(ui64 seed, ui32 blockSize, TConstArrayRef<TGroupId> queryIds, TVector<TIndicesType>* orderPtr) {
        auto grouppedQueries = GroupSamples(queryIds);
        auto offsets = ComputeGroupOffsets(grouppedQueries);
        TVector<ui32> order;
        Shuffle(seed, blockSize, grouppedQueries.size(), &order);
        auto& docwiseOrder = *orderPtr;
        docwiseOrder.resize(queryIds.size());

        ui32 cursor = 0;
        for (ui32 i = 0; i < order.size(); ++i) {
            const auto queryid = order[i];
            ui32 queryOffset = offsets[queryid];
            ui32 querySize = grouppedQueries[queryid].size();
            for (ui32 doc = 0; doc < querySize; ++doc) {
                docwiseOrder[cursor++] = queryOffset + doc;
            }
        }
    }

}
