#pragma once

#include <catboost/libs/data/objects_grouping.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/metal/native/metal_ordered_kernels.h>
#include <catboost/metal/native/metal_ordered_trainer.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>

namespace NCB {
    struct TMetalOrderedStochasticShape {
        TVector<ui32> WeakSeedCounts;
        ui32 LeafTaskCount = 0;
        // [estimateEnd, qualityEnd, cursorOffset, permutation] for every
        // learning prefix followed by the independent full-estimation task.
        TVector<ui32> Descriptors;
        ui32 CursorCount = 0;
    };

    // Reconstruct the runtime's fold and stochastic-oracle geometry without
    // creating a Metal session. Completed snapshots need the same validation
    // as snapshots that continue training. Share the actual host fold builder
    // with the runtime rather than inferring counts from an untrusted snapshot.
    inline TMetalOrderedStochasticShape MakeMetalOrderedStochasticShape(
            const CBMOrderedParams& params,
            const TObjectsGrouping& grouping,
            TConstArrayRef<TVector<ui32>> permutationOrders,
            double groupFoldGrowth,
            ui32 yetiComponents,
            bool combination) {
        // This shared header also declares the embedded shader string. Mark
        // it referenced without compiling, allocating or dispatching a shader.
        (void)CBMMetalOrderedSource;
        CB_ENSURE(params.rows >= 4 && params.rows <= (1u << 24) &&
                  grouping.GetObjectCount() == params.rows && grouping.GetGroupCount() >= 4 &&
                  params.permutations >= 1 && params.permutations <= 64 &&
                  yetiComponents >= 1 && yetiComponents <= 128 &&
                  (combination ? params.objective == 19 : params.objective == 17 && yetiComponents == 1),
                  "Invalid Metal Ordered stochastic shape dimensions or objective");
        CB_ENSURE(permutationOrders.empty() ? params.permutations == 1 :
                  permutationOrders.size() == params.permutations,
                  "Metal Ordered stochastic histories differ from the permutation count");
        CB_ENSURE(ui64(params.permutations) * params.rows * sizeof(ui32) <= (1ull << 30),
                  "Metal Ordered stochastic histories exceed the 1 GiB limit");

        TMetalOrderedStochasticShape result;
        result.WeakSeedCounts.resize(params.permutations, 1);
        const ui32 learnCount = params.permutations > 1 ? params.permutations - 1 : 1;
        const double growth = groupFoldGrowth ? groupFoldGrowth : params.fold_growth;
        ui64 cursorCount = 0;
        ui64 leafTaskCount = 1; // Independent full-estimation task is always active.

        for (ui32 permutation = 0; permutation < params.permutations; ++permutation) {
            const TConstArrayRef<ui32> order = permutationOrders.empty()
                ? TConstArrayRef<ui32>() : MakeConstArrayRef(permutationOrders[permutation]);
            CB_ENSURE(permutationOrders.empty() || order.size() == params.rows,
                      "Metal Ordered stochastic history has a different row count");
            const auto sourceRow = [&](ui32 position) {
                return permutationOrders.empty() ? position : order[position];
            };
            TVector<bool> seen(grouping.GetGroupCount(), false);
            std::vector<ui32> groupEnds;
            groupEnds.reserve(grouping.GetGroupCount());
            ui32 position = 0;
            while (position < params.rows) {
                const ui32 first = sourceRow(position);
                CB_ENSURE(first < params.rows, "Metal Ordered stochastic history row is out of range");
                const ui32 group = grouping.GetGroupIdxForObject(first);
                const auto bounds = grouping.GetGroup(group);
                const ui32 size = bounds.GetSize();
                CB_ENSURE(first == bounds.Begin && !seen[group] && ui64(position) + size <= params.rows,
                          "Metal Ordered stochastic histories must preserve each complete group once");
                seen[group] = true;
                for (ui32 row = 0; row < size; ++row) {
                    CB_ENSURE(sourceRow(position + row) == first + row,
                              "Metal Ordered stochastic histories must preserve within-group row order");
                }
                position += size;
                groupEnds.push_back(position);
            }
            if (permutation >= learnCount) continue;

            // ordered.h supplies explicit offsets for both YetiRank and a
            // Combination containing YetiRank, including singleton groups.
            // Thus these paths use grouped folds even for a trivial grouping.
            const auto folds = CBMCreateGroupedOrderedFolds(params.rows, groupEnds, growth, params.min_fold_size);
            CB_ENSURE(!folds.empty(), "Metal Ordered stochastic history has no prefix folds");
            const ui64 weakSeeds = ui64(folds.size()) * 2 * yetiComponents;
            CB_ENSURE(weakSeeds <= Max<ui32>(), "Metal Ordered stochastic weak seed count overflows");
            result.WeakSeedCounts[permutation] = static_cast<ui32>(weakSeeds);
            for (const auto& fold : folds) {
                CB_ENSURE(cursorCount + fold.QualityEnd <= Max<ui32>(),
                          "Metal Ordered stochastic cursor count overflows");
                result.Descriptors.insert(result.Descriptors.end(), {
                    fold.EstimateEnd, fold.QualityEnd, static_cast<ui32>(cursorCount), permutation});
                cursorCount += fold.QualityEnd;
                const auto prefixEnd = std::lower_bound(groupEnds.begin(), groupEnds.end(), fold.EstimateEnd);
                CB_ENSURE(prefixEnd != groupEnds.end() && *prefixEnd == fold.EstimateEnd,
                          "Metal Ordered stochastic prefix cuts a group");
                const ui32 prefixGroups = static_cast<ui32>(prefixEnd - groupEnds.begin()) + 1;
                // Pure Yeti skips estimation on an all-singleton prefix.
                // Combination still estimates every prefix for its other
                // components, even when its Yeti contribution is identically zero.
                if (combination || prefixGroups < fold.EstimateEnd) ++leafTaskCount;
            }
        }
        CB_ENSURE(cursorCount + params.rows <= Max<ui32>() &&
                  leafTaskCount * yetiComponents <= Max<ui32>(),
                  "Metal Ordered stochastic final task shape overflows");
        result.Descriptors.insert(result.Descriptors.end(), {
            params.rows, params.rows, static_cast<ui32>(cursorCount), params.permutations - 1});
        result.CursorCount = static_cast<ui32>(cursorCount + params.rows);
        result.LeafTaskCount = static_cast<ui32>(leafTaskCount * yetiComponents);
        return result;
    }
}
