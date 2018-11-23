#pragma once

#include <util/generic/utility.h>
#include <util/system/yassert.h>

inline void InitElementRange(
    ui32 partNumber,
    ui32 partsCount,
    ui32 elementCount,
    ui32* elemStartIdx,
    ui32* elemEndIdx
) {
    const ui32 minGroupsInTask = elementCount / partsCount;
    *elemStartIdx = minGroupsInTask * partNumber + Min(partNumber, elementCount % partsCount);

    *elemEndIdx = *elemStartIdx + minGroupsInTask;
    if (partNumber < elementCount % partsCount) {
        (*elemEndIdx)++;
    }

    Y_ASSERT(*elemStartIdx <= *elemEndIdx);
}
