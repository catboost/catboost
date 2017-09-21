#pragma once

namespace NKernelHost {
    template <typename T> void  CreateUniqueMasks(const T* sortedKeys, uint * masks, const uint size);
    
    template <typename T> void  GatherUnique(T* sortedKeys, uint * masks, uint* prefixSum, uint size, uint uniqueSize, T* tmp);
};