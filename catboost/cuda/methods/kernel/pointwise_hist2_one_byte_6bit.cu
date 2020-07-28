#include "pointwise_hist2_one_byte_templ.cuh"
#include "split_properties_helpers.cuh"
#include <cooperative_groups.h>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    template <>
    struct TLoadEntriesTrait<1, false> {
        constexpr static ELoadType LoadType() {
            #if __CUDA_ARCH__ < 700
            return ELoadType::OneElement;
            #else
            return ELoadType::FourElements;
            #endif
        }
    };


    template <>
    struct TLoadEntriesTrait<1, true> {
        constexpr static ELoadType LoadType() {
            #if __CUDA_ARCH__ < 520
            return ELoadType::OneElement;
            #elif __CUDA_ARCH__  < 700
            return ELoadType::TwoElements;
            #else
            return ELoadType::FourElements;
            #endif
        }
    };


    template <>
    struct TDeclarePassInnerOuterBitsTrait<1> {
        constexpr static int Inner() {
            return 1;
        }

        constexpr static int Outer() {
            return 0;
        }
    };


    template <int BLOCK_SIZE>
    struct TPointHist<0, 1, BLOCK_SIZE> {
        float* __restrict__ Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpId = (threadIdx.x / 32);
            const int warpOffset = 1024 * warpId;
            const int blocks = 2;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << 4));
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


        __forceinline__ __device__ void AddPoint(ui32 ci,
                                                 const float t,
                                                 const float w) {

            thread_block_tile<16> syncTile = tiled_partition<16>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const float stat1 = flag ? t : w;
            const float stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = ((2 * i + threadIdx.x) & 6);
                const int bin = (ci >> (24 - (f << 2))) & 255;
                const float pass = bin != 64 ? 1.0f : 0.0f;
                int offset = f +  16 * (bin & 62) + 8 * (bin & 1);

                const bool writeFirstFlag = threadIdx.x & 8;

                const float val1 = pass * stat1;
                const float val2 = pass * stat2;

                offset += flag;

                syncTile.sync();

                if (writeFirstFlag) {
                    Buffer[offset] += val1;
                }

                syncTile.sync();

                if (!writeFirstFlag) {
                    Buffer[offset] += val1;
                }

                offset = flag ? offset - 1 : offset + 1;

                syncTile.sync();

                if (writeFirstFlag) {
                    Buffer[offset] += val2;
                }

                syncTile.sync();

                if (!writeFirstFlag) {
                    Buffer[offset] += val2;
                }
            }
        }

        #if __CUDA_ARCH < 700
        __forceinline__ __device__ void AddPoint2(uint2 ci, const float2 t, const float2 w) {
            AddPoint(ci.x, t.x, w.x);
            AddPoint(ci.y, t.y, w.y);
        }
        #else
        __forceinline__ __device__ void AddPoint2(uint2 ci, const float2 t, const float2 w) {

            thread_block_tile<32> syncTile = tiled_partition<32>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const float2 stat1 = flag ? t : w;
            const float2 stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int f = ((2 * i + threadIdx.x) & 6);
                const ui32 shift = static_cast<ui32>(24 - (f << 2));
                f += flag;

                const int binx = (ci.x >> shift) & 255;
                const int biny = (ci.y >> shift) & 255;

                const float passx = binx != 64;
                const float passy = biny != 64;

                float* buffer = Buffer + f;

                syncTile.sync();

                int offsetx = 16 * (binx & 62) + 8 * (binx & 1);
                int offsety = 16 * (biny & 62) + 8 * (biny & 1);

                const bool writeFirstFlag = threadIdx.x & 8;

                const float valx = passx * stat1.x;
                const float valy = passy * stat1.y;


                if (writeFirstFlag) {
                    buffer[offsetx] += valx;
                    buffer[offsety] += valy;
                }

                syncTile.sync();

                if (!writeFirstFlag) {
                    buffer[offsetx] += valx;
                    buffer[offsety] += valy;
                }

                const float val2x = passx * stat2.x;
                const float val2y = passy * stat2.y;

                syncTile.sync();

                offsetx += flag ? -1 : 1;
                offsety += flag ? -1 : 1;

                if (writeFirstFlag) {
                    buffer[offsetx] += val2x;
                    buffer[offsety] += val2y;
                }

                syncTile.sync();

                if (!writeFirstFlag) {
                    buffer[offsetx] += val2x;
                    buffer[offsety] += val2y;
                }
            }
        }
        #endif

        __forceinline__ __device__ void AddPoint4(uint4 ci, const float4 t, const float4 w) {
            //don't change anything without performance tests, cuda so awesome, that little change of code could slow everything by 5-10%
            thread_block_tile<32> syncTile = tiled_partition<32>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const float4 stat1 = flag ? t : w;
            const float4 stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int f = ((2 * i + threadIdx.x) & 6);
                const int shift = (24 - (f << 2));
                f += flag;

                const int binx = (ci.x >> shift) & 255;
                const int biny = (ci.y >> shift) & 255;
                const int binz = (ci.z >> shift) & 255;
                const int binw = (ci.w >> shift) & 255;

                const bool passx = binx != 64;
                const bool passy = biny != 64;
                const bool passz = binz != 64;
                const bool passw = binw != 64;

                const float valx = passx ? stat1.x : 0.0f;
                const float valy = passy ? stat1.y : 0.0f;
                const float valz = passz ? stat1.z : 0.0f;
                const float valw = passw ? stat1.w : 0.0f;

                const float val2x = passx ? stat2.x : 0.0f;
                const float val2y = passy ? stat2.y : 0.0f;
                const float val2z = passz ? stat2.z : 0.0f;
                const float val2w = passw ? stat2.w : 0.0f;

                float* buffer = Buffer + f;
                const int offsetx = 16 * (binx & 62) + 8 * (binx & 1);
                const int offsety = 16 * (biny & 62) + 8 * (biny & 1);
                const int offsetz = 16 * (binz & 62) + 8 * (binz & 1);
                const int offsetw = 16 * (binw & 62) + 8 * (binw & 1);

                const bool writeFirstFlag = threadIdx.x & 8;

                syncTile.sync();

                if (writeFirstFlag) {
                    buffer[offsetx] += valx;
                    buffer[offsety] += valy;
                    buffer[offsetz] += valz;
                    buffer[offsetw] += valw;
                }

                syncTile.sync();

                if (!writeFirstFlag) {
                    buffer[offsetx] += valx;
                    buffer[offsety] += valy;
                    buffer[offsetz] += valz;
                    buffer[offsetw] += valw;
                }

                const int ptrShift = flag ? -1 : 1;
                buffer += ptrShift;

                syncTile.sync();

                if (writeFirstFlag) {
                    buffer[offsetx] += val2x;
                    buffer[offsety] += val2y;
                    buffer[offsetz] += val2z;
                    buffer[offsetw] += val2w;
                }

                syncTile.sync();

                if (!writeFirstFlag) {
                    buffer[offsetx] += val2x;
                    buffer[offsety] += val2y;
                    buffer[offsetz] += val2z;
                    buffer[offsetw] += val2w;
                }
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
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                const int fold0 = (threadIdx.x >> 1) & 31;

                const int maxFoldCount = 64;

                {
                    const int innerHistCount = 2;
                    const volatile float* __restrict__ src = Buffer
                                                    + 1024  //warpHistSize
                                                    + 8 * (fold0 & 1)
                                                    + 32 * (fold0 >> 1)
                                                    + w;

                    #pragma unroll
                    for (int inWarpHist = 0; inWarpHist < innerHistCount; ++inWarpHist) {
                        sum0 += src[2 * f + (inWarpHist << 4)];
                        sum1 += src[2 * f + (inWarpHist << 4) + 512];
                    }

                    Buffer[2 * (maxFoldCount * f + fold0) + w] = sum0;
                    Buffer[2 * (maxFoldCount * f + fold0 + 32) + w] = sum1;
                }
            }
            __syncthreads();
        }
    };



    DEFINE_NON_BINARY(6)
}
