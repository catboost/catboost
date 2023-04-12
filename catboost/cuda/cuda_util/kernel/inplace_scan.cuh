#pragma once

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

    template <typename T>
    __forceinline__ __device__

    void InplaceInclusiveScanN(T *data, ui32 n, ui32 tid) {
        T val = data[tid];
        __syncthreads();
        // assume n <= 2048
        if (n > 1) {
            if (tid < n && tid >= 1) { val += data[tid - 1]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        else { return; }
        if (n > 2) {
            if (tid < n && tid >= 2) { val += data[tid - 2]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        else { return; }
        if (n > 4) {
            if (tid < n && tid >= 4) { val += data[tid - 4]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        else { return; }
        if (n > 8) {
            if (tid < n && tid >= 8) { val += data[tid - 8]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        else { return; }
        if (n > 16) {
            if (tid < n && tid >= 16) { val += data[tid - 16]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        else { return; }
        if (n > 32) {
            if (tid < n && tid >= 32) { val += data[tid - 32]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        else { return; }
        if (n > 64) {
            if (tid < n && tid >= 64) { val += data[tid - 64]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        else { return; }
        if (n > 128) {
            if (tid < n && tid >= 128) { val += data[tid - 128]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        else { return; }
        if (n > 256) {
            if (tid < n && tid >= 256) { val += data[tid - 256]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        else { return; }
        if (n > 512) {
            if (tid < n && tid >= 512) { val += data[tid - 512]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        else { return; }
        if (n > 1024) {
            if (tid < n && tid >= 1024) { val += data[tid - 1024]; }
            __syncthreads();
            data[tid] = val;
            __syncthreads();
        }
        else { return; }
    }


//warning: works only for warp, no sync
    template <typename T>
    __forceinline__ __device__ void InclusiveScanInWarp(volatile T *data, ui32 tid) {
        __syncwarp();
        T val = data[tid];
        if (tid >= 1) {
            val += data[tid - 1];
        }
        __syncwarp();

        data[tid] = val;
        if (tid >= 2) {
            val += data[tid - 2];
        }
        __syncwarp();
        data[tid] = val;
        if (tid >= 4) {
            val += data[tid - 4];
        }
        __syncwarp();

        data[tid] = val;
        if (tid >= 8) {
            val += data[tid - 8];
        }
        __syncwarp();

        data[tid] = val;
        if (tid >= 16) {
            val += data[tid - 16];
        }
        __syncwarp();
        data[tid] = val;
        __syncwarp();
    }

}
