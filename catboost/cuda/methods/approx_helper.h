#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <vector>

namespace NCatboostCuda {
    template <class TMapping, class TFloat>
    inline void ReadApproxInCpuFormat(const TCudaBuffer<TFloat, TMapping>& cursor, bool oneMoreApproxAsZeroes, TVector<TVector<double>>* pointOnCpuPtr) {
        auto& pointOnCpu = *pointOnCpuPtr;
        const ui32 columnCount = cursor.GetColumnCount();
        pointOnCpu.resize(columnCount);
        TVector<float> point;
        cursor.Read(point);
        const ui64 docCount = cursor.GetObjectsSlice().Size();
        CB_ENSURE(point.size() == docCount * columnCount, "BUG: report to catboost team");

        for (ui32 column = 0; column < cursor.GetColumnCount(); ++column) {
            pointOnCpu[column].resize(docCount);
            for (size_t i = 0; i < docCount; ++i) {
                pointOnCpu[column][i] = point[column * docCount + i];
            }
        }

        if (oneMoreApproxAsZeroes) {
            //make it zeroes for CPU compatibility
            pointOnCpu.resize(columnCount + 1);
            pointOnCpu[columnCount].clear();
            pointOnCpu[columnCount].resize(docCount);
        }
    }
}
