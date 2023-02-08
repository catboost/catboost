#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

//TODO(noxoomo): compare with CUB routines and pour implementation (cub temp storage is not flexible enough to
namespace NKernel {

    template <typename T, ui32 BLOCK_SIZE>
    __forceinline__ __device__ void InplaceInclusiveScan(T *data, ui32 tid) {
        T val = data[tid];
        __syncthreads();
        // assume n <= 2048
        if (BLOCK_SIZE > 1) {
            if (tid >= 1) { val += data[tid - 1]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        if (BLOCK_SIZE > 2) {
            if (tid >= 2) { val += data[tid - 2]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        if (BLOCK_SIZE > 4) {
            if (tid >= 4) { val += data[tid - 4]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        if (BLOCK_SIZE > 8) {
            if (tid >= 8) { val += data[tid - 8]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        if (BLOCK_SIZE > 16) {
            if (tid >= 16) { val += data[tid - 16]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        if (BLOCK_SIZE > 32) {
            if (tid >= 32) { val += data[tid - 32]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        if (BLOCK_SIZE > 64) {
            if (tid >= 64) { val += data[tid - 64]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        if (BLOCK_SIZE > 128) {
            if (tid >= 128) { val += data[tid - 128]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        if (BLOCK_SIZE > 256) {
            if (tid >= 256) { val += data[tid - 256]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        if (BLOCK_SIZE > 512) {
            if (tid >= 512) { val += data[tid - 512]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        if (BLOCK_SIZE > 1024) {
            if (tid >= 1024) { val += data[tid - 1024]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
    }


    template <typename K, typename V, int BLOCK_SIZE, bool CMP_LESS>
    __forceinline__ __device__ void SortKeyValues(K* keys, V* values) {
        int i = threadIdx.x;
        for (int length = 1; length < BLOCK_SIZE; length <<= 1) {
            K iKey = keys[i];
            V iVal = values[i];
            int ii = i & (length - 1); // index in our sequence in 0..length-1
            int sibling = (i - ii) ^ length; // beginning of the sibling sequence
            int pos = 0;
            if (sibling & length) {
                for (int inc = length; inc > 0; inc >>= 1) { // increment for dichotomic search
                    int j = sibling + min(pos + inc, length) - 1;
                    K jKey = keys[j];
                    if (CMP_LESS)
                        pos += jKey < iKey ? inc : 0;
                    else
                        pos += jKey > iKey ? inc : 0;
                }
            } else {
                for (int inc = length; inc > 0; inc >>= 1) { // increment for dichotomic search
                    int j = sibling + min(pos + inc, length) - 1;
                    K jKey = keys[j];
                    if (CMP_LESS)
                        pos += jKey <= iKey ? inc : 0;
                    else
                        pos += jKey >= iKey ? inc : 0;
                }
            }
            pos = min(pos, length);
            int bits = (length << 1) - 1; // mask for destination
            int dest = ((ii + pos) & bits) | (i & ~bits); // destination index in merged sequence
            __syncthreads();
            keys[dest] = iKey;
            values[dest] = iVal;
            __syncthreads();
        }
    }

    template <typename K, typename V, int BLOCK_SIZE, bool CMP_LESS>
    __forceinline__ __device__ void SortKeyValues2(K* keys, V* values) {
        int i = threadIdx.x;
        for (int length = 1; length < BLOCK_SIZE; length <<= 1) {
            K iKey = keys[i];
            V iVal = values[i];
            int ii = i & (length - 1); // index in our sequence in 0..length-1
            int sibling = (i - ii) ^ length; // beginning of the sibling sequence
            int pos = 0;
            for (int inc = length; inc > 0; inc >>= 1) { // increment for dichotomic search
                int j = sibling + pos + inc - 1;
                K jKey = keys[j];
                if (CMP_LESS)
                    pos += ((jKey < iKey) || (jKey == iKey && j < i)) ? inc : 0;
                else
                    pos += ((jKey > iKey) || (jKey == iKey && j < i)) ? inc : 0;
                pos = min(pos, length);
            }
            int bits = (length << 1) - 1; // mask for destination
            int dest = ((ii + pos) & bits) | (i & ~bits); // destination index in merged sequence
            __syncthreads();
            keys[dest] = iKey;
            values[dest] = iVal;
            __syncthreads();
        }
    }

    template <ui32 BLOCK_SIZE>
    __device__ uint4 RadixSortRank4(uint4& pred, ui32* sdata) {
        sdata[threadIdx.x] = pred.x;
        sdata[threadIdx.x + BLOCK_SIZE] = pred.y;
        sdata[threadIdx.x + 2 * BLOCK_SIZE] = pred.z;
        sdata[threadIdx.x + 3 * BLOCK_SIZE] = pred.w;
        __syncthreads();

        ui32 val0 = sdata[threadIdx.x * 4];
        ui32 val1 = val0 + sdata[threadIdx.x * 4 + 1];
        ui32 val2 = val1 + sdata[threadIdx.x * 4 + 2];
        ui32 val3 = val2 + sdata[threadIdx.x * 4 + 3];
        __syncthreads();
        sdata[threadIdx.x] = val3;
        InplaceInclusiveScan<ui32, BLOCK_SIZE>(sdata, threadIdx.x);
        ui32 tmp = (threadIdx.x ? sdata[threadIdx.x - 1] : 0);
        __syncthreads();
        sdata[threadIdx.x * 4] = val0 + tmp;
        sdata[threadIdx.x * 4 + 1] = val1 + tmp;
        sdata[threadIdx.x * 4 + 2] = val2 + tmp;
        sdata[threadIdx.x * 4 + 3] = val3 + tmp;
        __syncthreads();

        ui32 numTrue = sdata[4 * BLOCK_SIZE - 1];
        uint4 r;
        r.x = pred.x ? (sdata[threadIdx.x] - 1) : (numTrue + threadIdx.x - sdata[threadIdx.x]);
        r.y = pred.y ? (sdata[threadIdx.x] - 1) : (numTrue + threadIdx.x + BLOCK_SIZE - sdata[threadIdx.x]);
        r.z = pred.z ? (sdata[threadIdx.x + 2 * BLOCK_SIZE] - 1) : (numTrue + threadIdx.x + BLOCK_SIZE * 2 - sdata[threadIdx.x + 2 * BLOCK_SIZE]);
        r.w = pred.w ? (sdata[threadIdx.x + 3 * BLOCK_SIZE] - 1) : (numTrue + threadIdx.x + BLOCK_SIZE * 3 - sdata[threadIdx.x + 3 * BLOCK_SIZE]);
        __syncthreads();
        return r;
    }

    template <ui32 BLOCK_SIZE, bool CMP_LESS, ui32 START_BIT, ui32 NUM_BITS, typename V>
    __device__ void RadixSortSingleBlock4(uint4& key, V& val, ui32* data) {
        ui32 tid = threadIdx.x;
        ui32* ptrx = data + tid;
        ui32* ptry = data + tid + BLOCK_SIZE;
        ui32* ptrz = data + tid + BLOCK_SIZE * 2;
        ui32* ptrw = data + tid + BLOCK_SIZE * 3;
        for (ui32 shift = START_BIT; shift < (START_BIT + NUM_BITS); shift++) {
            uint4 lsb;
            lsb.x = ((key.x >> shift) & 1) ^ CMP_LESS;
            lsb.y = ((key.y >> shift) & 1) ^ CMP_LESS;
            lsb.z = ((key.z >> shift) & 1) ^ CMP_LESS;
            lsb.w = ((key.w >> shift) & 1) ^ CMP_LESS;

            *ptrx = lsb.x;
            *ptry = lsb.y;
            *ptrz = lsb.z;
            *ptrw = lsb.w;
            __syncthreads();

            ui32 val0 = data[tid * 4];
            ui32 val1 = val0 + data[tid * 4 + 1];
            ui32 val2 = val1 + data[tid * 4 + 2];
            ui32 val3 = val2 + data[tid * 4 + 3];
            __syncthreads();
            data[tid] = val3;
            InplaceInclusiveScan<ui32, BLOCK_SIZE>(data, tid);
            ui32 tmp = (tid ? data[tid - 1] : 0);
            __syncthreads();
            data[tid * 4] = val0 + tmp;
            data[tid * 4 + 1] = val1 + tmp;
            data[tid * 4 + 2] = val2 + tmp;
            data[tid * 4 + 3] = val3 + tmp;
            __syncthreads();

            ui32 numTrue = data[4 * BLOCK_SIZE - 1];
            uint4 r;
            r.x = lsb.x ? ((*ptrx) - 1) : (numTrue + tid - (*ptrx));
            r.y = lsb.y ? ((*ptry) - 1) : (numTrue + tid + BLOCK_SIZE - (*ptry));
            r.z = lsb.z ? ((*ptrz) - 1) : (numTrue + tid + BLOCK_SIZE * 2 - (*ptrz));
            r.w = lsb.w ? ((*ptrw) - 1) : (numTrue + tid + BLOCK_SIZE * 3 - (*ptrw));
            __syncthreads();
            //        uint4 r = RadixSortRank4<BlockSize>(lsb, data);

            data[r.x] = key.x;
            data[r.y] = key.y;
            data[r.z] = key.z;
            data[r.w] = key.w;
            __syncthreads();
            key.x = *ptrx;
            key.y = *ptry;
            key.z = *ptrz;
            key.w = *ptrw;
            __syncthreads();

            data[r.x] = val.x;
            data[r.y] = val.y;
            data[r.z] = val.z;
            data[r.w] = val.w;
            __syncthreads();
            val.x = *ptrx;
            val.y = *ptry;
            val.z = *ptrz;
            val.w = *ptrw;
            __syncthreads();
        }
    }


    template <ui32 BLOCK_SIZE, bool CMP_LESS, ui32 START_BIT, ui32 NUM_BITS>
    __device__ void RadixSortSingleBlock4(uint4& key, ui32* data) {
        ui32 tid = threadIdx.x;
        ui32* ptrx = data + tid;
        ui32* ptry = data + tid + BLOCK_SIZE;
        ui32* ptrz = data + tid + BLOCK_SIZE * 2;
        ui32* ptrw = data + tid + BLOCK_SIZE * 3;

        for (ui32 shift = START_BIT; shift < (START_BIT + NUM_BITS); shift++) {
            uint4 lsb;
            lsb.x = ((key.x >> shift) & 1) ^ CMP_LESS;
            lsb.y = ((key.y >> shift) & 1) ^ CMP_LESS;
            lsb.z = ((key.z >> shift) & 1) ^ CMP_LESS;
            lsb.w = ((key.w >> shift) & 1) ^ CMP_LESS;

            *ptrx = lsb.x;
            *ptry = lsb.y;
            *ptrz = lsb.z;
            *ptrw = lsb.w;
            __syncthreads();

            ui32 val0 = data[tid * 4];
            ui32 val1 = val0 + data[tid * 4 + 1];
            ui32 val2 = val1 + data[tid * 4 + 2];
            ui32 val3 = val2 + data[tid * 4 + 3];
            __syncthreads();
            data[tid] = val3;
            InplaceInclusiveScan<ui32, BLOCK_SIZE>(data, tid);
            ui32 tmp = (tid ? data[tid - 1] : 0);
            __syncthreads();
            data[tid * 4] = val0 + tmp;
            data[tid * 4 + 1] = val1 + tmp;
            data[tid * 4 + 2] = val2 + tmp;
            data[tid * 4 + 3] = val3 + tmp;
            __syncthreads();

            ui32 numTrue = data[4 * BLOCK_SIZE - 1];
            uint4 r;
            r.x = lsb.x ? ((*ptrx) - 1) : (numTrue + tid - (*ptrx));
            r.y = lsb.y ? ((*ptry) - 1) : (numTrue + tid + BLOCK_SIZE - (*ptry));
            r.z = lsb.z ? ((*ptrz) - 1) : (numTrue + tid + BLOCK_SIZE * 2 - (*ptrz));
            r.w = lsb.w ? ((*ptrw) - 1) : (numTrue + tid + BLOCK_SIZE * 3 - (*ptrw));
            __syncthreads();
            //        uint4 r = RadixSortRank4<BlockSize>(lsb, data);

            data[r.x] = key.x;
            data[r.y] = key.y;
            data[r.z] = key.z;
            data[r.w] = key.w;
            __syncthreads();
            key.x = *ptrx;
            key.y = *ptry;
            key.z = *ptrz;
            key.w = *ptrw;
            __syncthreads();
        }
    }

}
