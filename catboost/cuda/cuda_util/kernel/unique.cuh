#pragma once

namespace NKernelHost {
    template <typename T> void  CreateUniqueMasks(const T* sortedKeys, ui32 * masks, const ui32 size);

    template <typename T> void  GatherUnique(T* sortedKeys, ui32 * masks, ui32* prefixSum, ui32 size, ui32 uniqueSize, T* tmp);
};
