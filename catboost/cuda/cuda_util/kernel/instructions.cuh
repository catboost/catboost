#pragma once
#include <cub/thread/thread_load.cuh>


__forceinline__ __device__ ui32 bfe(ui32 a, ui32 start, ui32 length)
{
    ui32 res;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(start), "r"(length));
    return res;
}

__forceinline__ __device__ ui32 load_noncached(const ui32 * ptr)
{
    ui32 res;
    asm("ld.global.cg.u32 %0, [%1];" : "=r"(res) : "l"(ptr));
    return res;
}

__forceinline__ __device__ uint2 load_noncached(const uint2* ptr)
{
    uint2 res;
    asm("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(res.x), "=r"(res.y) : "l"(ptr));
    return res;
}

__forceinline__ __device__ float load_noncached(const float * ptr)
{
    float res;
    asm("ld.global.cg.f32 %0, [%1];" : "=f"(res) : "l"(ptr));
    return res;
}

__device__ __forceinline__ void PrefetchL1(void *ptr) {
    asm("prefetch.global.L1 [%0];"::"l"(ptr));
}


template <typename T>
__forceinline__ __device__ T const_load(const T* ptr) {
    return cub::ThreadLoad<cub::LOAD_CS>(ptr);
}
