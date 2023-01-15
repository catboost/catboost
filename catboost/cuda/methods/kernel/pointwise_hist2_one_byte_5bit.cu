#include "pointwise_hist2_one_byte_templ.cuh"
#include "split_properties_helpers.cuh"
#include "compute_point_hist2_loop.cuh"
#include <cooperative_groups.h>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    template <>
    struct TLoadEntriesTrait<0, false> {
        constexpr static ELoadType LoadType() {
            #if __CUDA_ARCH__ < 700
            return ELoadType::OneElement;
            #else
            return ELoadType::FourElements;
            #endif
        }
    };

    template <>
    struct TLoadEntriesTrait<0, true> {
        constexpr static ELoadType LoadType() {
            #if __CUDA_ARCH__ < 520
            return ELoadType::OneElement;
            #elif __CUDA_ARCH__ < 700
            return ELoadType::TwoElements;
            #else
            return ELoadType::FourElements;
            #endif
        }
    };

    template <>
    struct TDeclarePassInnerOuterBitsTrait<0> {
        constexpr static int Inner() {
            return 0;
        }

        constexpr static int Outer() {
            return 0;
        }
    };

    template <int BLOCK_SIZE>
    struct TPointHist<0, 0, BLOCK_SIZE> {
        float* __restrict__ Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpId = (threadIdx.x / 32);
            const int warpOffset = 1024 * warpId;
            const int blocks = 4;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << 3));
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHist(float* buff) {

            const int HIST_SIZE = 32 * BLOCK_SIZE;

            #pragma unroll 8
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE) {
                buff[i] = 0;
            }

            Buffer = buff + SliceOffset();
            __syncthreads();
        }

        __forceinline__ __device__ void Add(float val, float* dst) {
            dst[0] += val;
        }

        __forceinline__ __device__ void AddPoint(ui32 ci,
                                                 const float t,
                                                 const float w) {

            thread_block_tile<8> syncTile = tiled_partition<8>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const float stat1 = flag ? t : w;
            const float stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = ((2 * i + threadIdx.x) & 6);
                const int bin = (ci >> (24 - (f << 2))) & 255;
                const bool pass = bin != 32;
                int offset = f + 32 * (bin & 31);
                const int offset1 = offset + flag;
                const float add1 = pass ? stat1 : 0.0f;
                const int offset2 = offset + !flag;
                const float add2 = pass ? stat2 : 0.0f;

                syncTile.sync();
                Buffer[offset1] += add1;
                syncTile.sync();
                Buffer[offset2] += add2;
            }
        }

        __forceinline__ __device__ void AddPoint2(uint2 ci,
                                                  const float2 t,
                                                  const float2 w) {

            thread_block_tile<8> syncTile = tiled_partition<8>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const float2 stat1 = flag ? t : w;
            const float2 stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int f = ((2 * i + threadIdx.x) & 6);
                const int bin1 = (ci.x >> (24 - (f << 2))) & 255;
                const int bin2 = (ci.y >> (24 - (f << 2))) & 255;

                const float passx = bin1 != 32 ? 1.0f : 0.0f;
                const float passy = bin2 != 32 ? 1.0f : 0.0f;

                int offsetx = f + 32 * (bin1 & 31) + flag;
                int offsety = f + 32 * (bin2 & 31) + flag;

                syncTile.sync();
                Buffer[offsetx] += passx * stat1.x;
                Buffer[offsety] += passy * stat1.y;

                offsetx += flag ? -1 : 1;
                offsety += flag ? -1 : 1;

                syncTile.sync();

                Buffer[offsetx] += passx * stat2.x;
                Buffer[offsety] += passy * stat2.y;
            }
        }

        __forceinline__ __device__ void AddPoint4(uint4 ci, const float4 t, const float4 w) {
            //don't change anything without performance tests, nvcc is so awesome, that little change of code could slow everything by 5-10%
            thread_block_tile<8> syncTile = tiled_partition<8>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const float4 stat1 = flag ? t : w;
            const float4 stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int f = ((2 * i + threadIdx.x) & 6);
                const ui32 shift = static_cast<ui32>(24 - (f << 2));
                f += flag;

                const int binx = (ci.x >> shift) & 255;
                const int biny = (ci.y >> shift) & 255;
                const int binz = (ci.z >> shift) & 255;
                const int binw = (ci.w >> shift) & 255;


                const float passx = binx != 32 ? 1.0f : 0.0f;
                const float passy = biny != 32 ? 1.0f : 0.0f;
                const float passz = binz != 32 ? 1.0f : 0.0f;
                const float passw = binw != 32 ? 1.0f : 0.0f;

                float* buffer = Buffer + f;


                int offsetx = (binx & 31) << 5;
                int offsety = (biny & 31) << 5;
                int offsetz = (binz & 31) << 5;
                int offsetw = (binw & 31) << 5;

                syncTile.sync();

                buffer[offsetx] += passx * stat1.x;
                buffer[offsety] += passy * stat1.y;
                buffer[offsetz] += passz * stat1.z;
                buffer[offsetw] += passw * stat1.w;


                offsetx += flag ? -1 : 1;
                offsety += flag ? -1 : 1;
                offsetz += flag ? -1 : 1;
                offsetw += flag ? -1 : 1;


                syncTile.sync();

                buffer[offsetx] += passx * stat2.x;
                buffer[offsety] += passy * stat2.y;
                buffer[offsetz] += passz * stat2.z;
                buffer[offsetw] += passw * stat2.w;
            }
        }


        //After reduce we store histograms by blocks: 256 floats (4 x 2 x 32)
        // for first 32 bins; than 256 floats for second 32 bins, etc
        __forceinline__ __device__ void Reduce() {
            Buffer -= SliceOffset();
            __syncthreads();

            {
                const int warpHistSize = 1024;

                for (int start = threadIdx.x; start < warpHistSize; start += BLOCK_SIZE) {
                    float sum = 0;
//                    12 iterations at 32-bin
                    #pragma unroll 12
                    for (int i = start; i < 32 * BLOCK_SIZE; i += warpHistSize) {
                        sum += Buffer[i];
                    }
                    Buffer[warpHistSize + start] = sum;
                }
            }

            __syncthreads();

            if (threadIdx.x < 256) {
                const int w = threadIdx.x & 1;
                const int f = threadIdx.x / 64;
                float sum = 0.0f;
                const int fold = (threadIdx.x >> 1) & 31;
                const int maxFoldCount = 32;

                if (fold < maxFoldCount) {
                    const int innerHistCount = 4;
                    const volatile float* __restrict__ src = Buffer
                                                    + 1024  //warpHistSize
                                                    + 32 * fold
                                                    + 2 * f
                                                    + w;

                    #pragma unroll
                    for (int inWarpHist = 0; inWarpHist < innerHistCount; ++inWarpHist) {
                        sum += src[(inWarpHist << 3)];
                    }

                    Buffer[2 * (maxFoldCount * f + fold) + w] = sum;
                }
            }
            __syncthreads();
        }
    };

    template <>
    struct TUnrollsTrait<0, ELoadType::FourElements> {
        constexpr static int Outer() {
            return 1;
        }
    };




    DEFINE_NON_BINARY(5)
}
