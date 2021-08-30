#pragma once

#include <catboost/cuda/cuda_lib/helpers.h>

#include <catboost/private/libs/data_types/groupid.h>
#include <catboost/private/libs/data_types/query.h>
#include <catboost/libs/helpers/cpu_random.h>

#include <util/generic/array_ref.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/random/shuffle.h>

#include <util/system/types.h>
#include <util/system/yassert.h>

#include <numeric>

namespace NCatboostCuda {

    template <class TIndicesType>
    inline void Shuffle(ui64 seed, ui32 blockSize, ui32 sampleCount, TVector<TIndicesType>* orderPtr) {
        TRandom rng(seed);
        rng.Advance(10);
        auto& order = *orderPtr;
        order.yresize(sampleCount);
        std::iota(order.begin(), order.end(), 0);

        if (blockSize == 1) {
            ::Shuffle(order.begin(), order.begin() + sampleCount, rng);
        } else {
            const auto blocksCount = static_cast<ui32>(::NHelpers::CeilDivide(order.size(), blockSize));
            TVector<ui32> blocks;
            blocks.yresize(blocksCount);
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
    inline void GenerateQueryDocsOrder(ui64 seed, ui32 blockSize, TConstArrayRef<TGroupBounds> groupBounds, TVector<TIndicesType>* orderPtr) {
        TVector<ui32> order;
        Shuffle(seed, blockSize, groupBounds.size(), &order);
        auto& docwiseOrder = *orderPtr;
        // we assume groupBounds are sorted in ascending order
        docwiseOrder.yresize(groupBounds.back().End);

        ui32 cursor = 0;
        for (ui32 groupId = 0; groupId < order.size(); ++groupId) {
            const auto group = groupBounds[order[groupId]];
            std::iota(docwiseOrder.begin() + cursor, docwiseOrder.begin() + cursor + group.GetSize(), group.Begin);
            cursor += group.GetSize();
        }
        Y_ASSERT(cursor == docwiseOrder.size());
    }

    // TODO(kirillovs): this function is used only for cuda unittests, maybe remove later
    template <class TIndicesType>
    inline void QueryConsistentShuffle(ui64 seed, ui32 blockSize, TConstArrayRef<TGroupId> queryIds, TVector<TIndicesType>* orderPtr) {
        GenerateQueryDocsOrder(seed, blockSize, GroupSamples<TGroupId>(queryIds), orderPtr);
    }
}
