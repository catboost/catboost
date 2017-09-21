#include "unique.cuh"

namespace NKernelHost {

template <typename T>
__global__ void GatherUnique2TmpImpl(T* keys, uint* masks, uint* map, uint size,  uint uniqueSize, T* tmp)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        const uint mask = masks[i];
        if (mask) {
            tmp[map[i]-1] = keys[i];
        }
    }

}

template <typename T>
__global__ void GatherUniqueImpl(T* keys, uint uniqueSize, T* tmp)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < uniqueSize) {
        keys[i] = tmp[i];
    }
}


template <typename T> void GatherUnique(T* keys, uint* masks, uint* prefixSum, uint size, uint uniqSize, T* tmp)
{
    if (size > 0) {
        {
            const uint blockSize = 512;
            const uint numBlocks = (size + blockSize - 1) / (blockSize);
            GatherUnique2TmpImpl << <numBlocks, blockSize>>>(keys, masks, prefixSum, size, uniqSize,tmp);
        }
        
        {
            const uint blockSize = 512;
            const uint numBlocks = (uniqSize + blockSize - 1) / (blockSize);
            GatherUniqueImpl<< <numBlocks, blockSize>>>(keys, uniqSize, tmp);
        }
    }
}
    
template <typename T>
__global__ void CreateUniqueMasksImpl(const T* keys, uint size, uint* result)
{
    const uint tid = threadIdx.x;
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
        const uint cur = local[tid+1];
        const uint prev = local[tid];
        result[i] = (cur == prev ? 0 : 1);
    }
    
}

template <typename T> void CreateUniqueMasks(const T* keys, uint* masks, const uint size)
{
    if (size > 0) {
        const uint blockSize = 512;
        const uint numBlocks = (size + blockSize - 1) / (blockSize);
        CreateUniqueMasksImpl << <numBlocks, blockSize>>>(keys, size, masks);
    }
}


template void CreateUniqueMasks<uint>(const uint* keys, uint* masks, const uint size);
template void GatherUnique<uint>(uint* keys, uint* masks, uint* prefixSum, uint size, uint uniqSize,uint* tmp);

}
