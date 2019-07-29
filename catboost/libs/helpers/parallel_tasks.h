#pragma once

#include "map_merge.h"

#include <library/dot_product/dot_product.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

#include <functional>


namespace NCB {

    // tasks is not const because elements of tasks are cleared after execution
    void ExecuteTasksInParallel(TVector<std::function<void()>>* tasks, NPar::TLocalExecutor* localExecutor);

    template <typename TNumber>
    inline TNumber L2NormSquared(
        TConstArrayRef<TNumber> array,
        NPar::TLocalExecutor* localExecutor
    ) {
        TNumber result = 0;
        NCB::MapMerge(
            localExecutor,
            TSimpleIndexRangesGenerator<int>(TIndexRange<int>(array.size()), /*blockSize*/10000),
            /*mapFunc*/[&](NCB::TIndexRange<int> partIndexRange, TNumber* output) {
                Y_ASSERT(!partIndexRange.Empty());
                *output = DotProduct(
                    array.data() + partIndexRange.Begin,
                    array.data() + partIndexRange.Begin,
                    partIndexRange.GetSize()
                );
            },
            /*mergeFunc*/[](TNumber* output, TVector<TNumber>&& addVector) {
                for (TNumber addItem : addVector) {
                    *output += addItem;
                }
            },
            &result
        );
        return result;
    }

    template <typename TNumber>
    inline void FillRank2(
        TNumber value,
        int rowCount,
        int columnCount,
        TVector<TVector<TNumber>>* dst,
        NPar::TLocalExecutor* localExecutor
    ) {
        constexpr int minimumElementCount = 1000;
        dst->resize(rowCount);
        if (rowCount * columnCount < minimumElementCount) {
            for (auto& dimension : *dst) {
                dimension.yresize(columnCount);
                Fill(dimension.begin(), dimension.end(), value);
            }
        } else if (columnCount < rowCount * minimumElementCount) {
            NPar::ParallelFor(
                *localExecutor,
                0,
                rowCount,
                [=] (int rowIdx) {
                    (*dst)[rowIdx].yresize(columnCount);
                    Fill((*dst)[rowIdx].begin(), (*dst)[rowIdx].end(), value);
                });
        } else {
            for (auto& dimension : *dst) {
                dimension.yresize(columnCount);
                const auto dimensionRef = MakeArrayRef(dimension);
                NPar::ParallelFor(
                    *localExecutor,
                    0,
                    columnCount,
                    [=] (int idx) {
                        dimensionRef[idx] = value;
                    });
            }
        }
    }
}
