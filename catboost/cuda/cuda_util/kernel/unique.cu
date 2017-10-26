#include <util/system/types.h>
#include "unique.cuh"

namespace NKernelHost {

template <typename T>
__global__ void GatherUnique2TmpImpl(T* keys, ui32* masks, ui32* map, ui32 size,  ui32 uniqueSize, T* tmp)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        const ui32 mask = masks[i];
        if (mask) {
            tmp[map[i]-1] = keys[i];
        }
    }

}

template <typename T>
__global__ void GatherUniqueImpl(T* keys, ui32 uniqueSize, T* tmp)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < uniqueSize) {
        keys[i] = tmp[i];
    }
}


template <typename T> void GatherUnique(T* keys, ui32* masks, ui32* prefixSum, ui32 size, ui32 uniqSize, T* tmp)
{
    if (size > 0) {
        {
            const ui32 blockSize = 512;
            const ui32 numBlocks = (size + blockSize - 1) / (blockSize);
            GatherUnique2TmpImpl << <numBlocks, blockSize>>>(keys, masks, prefixSum, size, uniqSize,tmp);
        }

        {
            const ui32 blockSize = 512;
            const ui32 numBlocks = (uniqSize + blockSize - 1) / (blockSize);
            GatherUniqueImpl<< <numBlocks, blockSize>>>(keys, uniqSize, tmp);
        }
    }
}

template <typename T>
__global__ void CreateUniqueMasksImpl(const T* keys, ui32 size, ui32* result)
{
    const ui32 tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;


    __shared__ T local[513];

    if (i < size) {
        const T val = keys[i];
        local[tid+1] = val;
    }
    if (tid == 0) {
        local[0] = i > 0 ? keys[i-1] : 0;
    }
    __syncthreads();

    if (i == 0) {
        result[i] = 1;
    } else if (i < size) {
        const ui32 cur = local[tid+1];
        const ui32 prev = local[tid];
        result[i] = (cur == prev ? 0 : 1);
    }

}

template <typename T> void CreateUniqueMasks(const T* keys, ui32* masks, const ui32 size)
{
    if (size > 0) {
        const ui32 blockSize = 512;
        const ui32 numBlocks = (size + blockSize - 1) / (blockSize);
        CreateUniqueMasksImpl << <numBlocks, blockSize>>>(keys, size, masks);
    }
}


template void CreateUniqueMasks<ui32>(const ui32* keys, ui32* masks, const ui32 size);
template void GatherUnique<ui32>(ui32* keys, ui32* masks, ui32* prefixSum, ui32 size, ui32 uniqSize,ui32* tmp);

}
