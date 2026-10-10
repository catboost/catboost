#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/math_utils.h>
#include <catboost/private/libs/options/catboost_options.h>

#include <util/random/shuffle.h>

#include <numeric>

namespace NCB {
    // CUDA dynamic_boosting.h::GetPermutationBlockSize. IntLog2 is ceil(log2),
    // so e.g. a configured block of 65 initially becomes 128, not 64.
    inline ui32 GetMetalFeatureParallelBlockSize(ui32 rows, ui32 configuredBlockSize) {
        if (rows < 50000) return 1;
        const ui32 suggested = configuredBlockSize ? configuredBlockSize : 64;
        ui64 block = ui64(1) << IntLog2(suggested);
        while (block * 128 > rows) block >>= 1;
        return static_cast<ui32>(block);
    }

    // Host-only counterpart of CUDA data_utils.h::Shuffle, sharing CatBoost's
    // actual MT19937-64 and rejection-sampled Fisher-Yates implementation.
    inline TVector<ui32> MakeMetalFeatureParallelBlockOrder(ui32 size, ui32 permutation, ui32 blockSize) {
        CB_ENSURE(blockSize > 0, "Metal FeatureParallel history block size must be positive");
        TVector<ui32> result(size);
        std::iota(result.begin(), result.end(), 0);
        if (!permutation || !size) return result;
        const ui32 seed = 1664525u * permutation + 1013904223u + blockSize;
        TRandom random(seed);
        random.Advance(10);
        if (blockSize == 1) {
            ::Shuffle(result.begin(), result.end(), random);
            return result;
        }
        const ui32 blockCount = (ui64(size) + blockSize - 1) / blockSize;
        TVector<ui32> blocks(blockCount);
        std::iota(blocks.begin(), blocks.end(), 0);
        ::Shuffle(blocks.begin(), blocks.end(), random);
        ui32 cursor = 0;
        for (ui32 block : blocks) {
            const ui64 begin = ui64(block) * blockSize;
            const ui32 end = Min<ui64>(begin + blockSize, size);
            for (ui32 row = begin; row < end; ++row) result[cursor++] = row;
        }
        return result;
    }

    inline TVector<ui32> MakeMetalFeatureParallelGroupedHistoryOrder(
            ui32 rows, TConstArrayRef<TGroupBounds> groups, ui32 permutation, ui32 blockSize) {
        if (!permutation) return MakeMetalFeatureParallelBlockOrder(rows, 0, blockSize);
        ui32 expectedBegin = 0;
        for (const auto& group : groups) {
            CB_ENSURE(group.Begin == expectedBegin && group.Begin < group.End && group.End <= rows,
                      "Metal FeatureParallel group bounds must partition the Pool");
            expectedBegin = group.End;
        }
        CB_ENSURE(expectedBegin == rows, "Metal FeatureParallel group bounds do not cover the Pool");
        const auto groupOrder = MakeMetalFeatureParallelBlockOrder(groups.size(), permutation, blockSize);
        TVector<ui32> result(rows);
        ui32 cursor = 0;
        for (ui32 group : groupOrder) {
            CB_ENSURE(groups[group].Begin < groups[group].End && groups[group].End <= rows &&
                      ui64(cursor) + groups[group].GetSize() <= rows,
                      "Metal FeatureParallel group bounds are invalid");
            for (ui32 row = groups[group].Begin; row < groups[group].End; ++row) result[cursor++] = row;
        }
        CB_ENSURE(cursor == rows, "Metal FeatureParallel group histories do not cover the Pool");
        return result;
    }

    inline TVector<ui32> MakeMetalFeatureParallelHistoryOrder(
            const TTrainingDataProvider& data, ui32 permutation, ui32 blockSize) {
        const ui32 rows = data.GetObjectCount();
        if (!permutation || !data.MetaInfo.HasGroupId || data.ObjectsGrouping->IsTrivial())
            return MakeMetalFeatureParallelBlockOrder(rows, permutation, blockSize);
        return MakeMetalFeatureParallelGroupedHistoryOrder(
            rows, data.ObjectsGrouping->GetNonTrivialGroups(), permutation, blockSize);
    }

    inline TVector<TVector<ui32>> MakeMetalFeatureParallelHistoryOrders(
            const TTrainingDataProvider& data, const NCatboostOptions::TCatBoostOptions& options) {
        // FeatureParallel honors the provider's order directly. Shared Pool
        // preprocessing marks has_time providers Ordered before this stage.
        const ui32 count = data.ObjectsData->GetOrder() == EObjectsOrder::Ordered
            ? 1 : options.BoostingOptions->PermutationCount.Get();
        CB_ENSURE(count > 0, "Metal FeatureParallel requires at least one permutation");
        const auto& configured = options.BoostingOptions->PermutationBlockSize;
        const ui32 block = GetMetalFeatureParallelBlockSize(data.GetObjectCount(),
                                                           configured.IsSet() ? configured.Get() : 64);
        CB_ENSURE(ui64(count) * data.GetObjectCount() * sizeof(ui32) <= (ui64(1) << 30),
                  "Metal FeatureParallel history orders exceed the 1 GiB limit");
        TVector<TVector<ui32>> result;
        for (ui32 p = 0; p < count; ++p) result.push_back(MakeMetalFeatureParallelHistoryOrder(data, p, block));
        return result;
    }

    // Consume the shared, mutable FeatureParallel RNG at the CUDA call site.
    // Do not reseed this helper per iteration: bootstrap and weak-learner RNG
    // consumers between these calls are part of the same persistent stream.
    template <class TRandomLike>
    inline ui32 ChooseMetalFeatureParallelPermutation(ui32 permutationCount, TRandomLike& random) {
        CB_ENSURE(permutationCount > 0, "Metal FeatureParallel permutation count must be positive");
        const ui32 learnCount = permutationCount > 1 ? permutationCount - 1 : 1;
        return learnCount > 1 ? static_cast<ui32>(random.NextUniformL() % (learnCount - 1)) : 0;
    }
}
