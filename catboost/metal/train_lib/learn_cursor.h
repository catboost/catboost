#pragma once

#include <catboost/libs/helpers/array_subset.h>

#include <util/generic/vector.h>

namespace NCB {
    // Native cursors and snapshots retain prepared-data order. Only the
    // transient Python result is scattered back to the caller's Pool order.
    inline void RestoreMetalLearnCursorOrder(
        const TArraySubsetIndexing<ui32>& preparedToOriginal,
        TVector<TVector<double>>* cursor)
    {
        if (cursor->empty()) {
            return;
        }
        const ui32 rows = preparedToOriginal.Size();
        TVector<ui8> seen(rows, 0);
        preparedToOriginal.ForEach([&](ui32, ui32 original) {
            CB_ENSURE(original < rows && !seen[original], "Metal learn row order must be a permutation");
            seen[original] = 1;
        });
        TVector<TVector<double>> reordered(cursor->size(), TVector<double>(rows));
        for (size_t dim = 0; dim < cursor->size(); ++dim) {
            CB_ENSURE((*cursor)[dim].size() == rows, "Metal learn row order differs from cursor dimensions");
            preparedToOriginal.ForEach([&](ui32 prepared, ui32 original) {
                reordered[dim][original] = (*cursor)[dim][prepared];
            });
        }
        cursor->swap(reordered);
    }
}
