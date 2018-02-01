#pragma once

#include <util/generic/utility.h>
#include <util/system/yassert.h>

inline void InitElementRange(int partNumber, int partsCount,
                                    int elementCount,
                                    int* elemStartIdx, int* elemEndIdx) {
    const int minGroupsInTask = elementCount / partsCount;
    *elemStartIdx = minGroupsInTask * partNumber + Min(partNumber, elementCount % partsCount);

    *elemEndIdx = *elemStartIdx + minGroupsInTask;
    if (partNumber < elementCount % partsCount) {
        (*elemEndIdx)++;
    }

    Y_ASSERT(*elemStartIdx <= *elemEndIdx);
}
