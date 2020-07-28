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
    struct TLoadEntriesTrait<2, false> {
        constexpr static ELoadType LoadType() {
            #if __CUDA_ARCH__ < 520
            return ELoadType::OneElement;
            #else
            return ELoadType::FourElements;
            #endif

        }
    };

    template <>
    struct TLoadEntriesTrait<2, true> {
        constexpr static ELoadType LoadType() {
            #if __CUDA_ARCH__ < 520
            return ELoadType::OneElement;
            #elif __CUDA_ARCH__  < 700
            return ELoadType::TwoElements;
            #else
            return  ELoadType::FourElements;
            #endif
        }
    };



    template <>
    struct TDeclarePassInnerOuterBitsTrait<2> {
        constexpr static int Inner() {
            return 2;
        }

        constexpr static int Outer() {
            return 0;
        }
    };

    template <int BLOCK_SIZE>
    struct TPointHist<0, 2, BLOCK_SIZE> {
        float* __restrict__ Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpId = (threadIdx.x / 32);
            const int warpOffset = 1024 * warpId;
            return warpOffset;
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
            thread_block_tile<32> syncTile = tiled_partition<32>(this_thread_block());
            const bool flag = threadIdx.x & 1;

            const float stat1 = flag ? t : w;
            const float stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = ((2 * i + threadIdx.x) & 6);
                const int bin = (ci >> (24 - (f << 2))) & 255;
                const float pass = bin != 128;
                int offset = f;
                offset += 8 * (bin & 127);
//
                const int writeTime = (threadIdx.x >> 3) & 3;

                const float val1 = pass * stat1;
                offset += flag;

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (k == writeTime) {
                        Buffer[offset] += val1;
                    }
                    syncTile.sync();
                }

                const float val2 = pass * stat2;
                offset = flag ? offset - 1 : offset + 1;

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (k == writeTime) {
                        Buffer[offset] += val2;
                    }
                    syncTile.sync();
                }
            }
        }

        #if __CUDA_ARCH__ < 700
        __forceinline__ __device__ void AddPoint2(uint2 bin, const float2 t, const float2 w) {
            AddPoint(bin.x, t.x, w.x);
            AddPoint(bin.y, t.y, w.y);
        }
        #else
        __forceinline__ __device__ void AddPoint2(uint2 ci, const float2 t, const float2 w) {
            const bool flag = threadIdx.x & 1;

            const float2 stat1 = flag ? t : w;
            const float2 stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = ((2 * i + threadIdx.x) & 6);

                const int binx = (ci.x >> (24 - (f << 2))) & 255;
                const int biny = (ci.y >> (24 - (f << 2))) & 255;

                float* buffer = Buffer + f + flag;

                int offsetx = 8 * (binx & 127);
                int offsety = 8 * (biny & 127);

                const bool passx = binx != 128;
                const bool passy = biny != 128;

                const float val1x = passx ? stat1.x : 0.0f;
                const float val1y = passy ? stat1.y : 0.0f;

                const float val2x = passx ? stat2.x : 0.0f;
                const float val2y = passy ? stat2.y : 0.0f;

                const int writeTime = (threadIdx.x >> 3) & 3;

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (k == writeTime) {
                        buffer[offsetx] += val1x;
                        buffer[offsety] += val1y;
                    }
                    __syncwarp();
                }


                buffer += flag ? -1 : 1;

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (k == writeTime) {
                        buffer[offsetx] += val2x;
                        buffer[offsety] += val2y;
                    }
                    __syncwarp();
                }
            }
        }
        #endif

        __forceinline__ __device__ void AddPoint4(uint4 ci, const float4 t, const float4 w) {
            thread_block_tile<32> syncTile = tiled_partition<32>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const float4 stat1 = flag ? t : w;
            const float4 stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = ((2 * i + threadIdx.x) & 6);

                const int binx = (ci.x >> (24 - (f << 2))) & 255;
                const int biny = (ci.y >> (24 - (f << 2))) & 255;
                const int binz = (ci.z >> (24 - (f << 2))) & 255;
                const int binw = (ci.w >> (24 - (f << 2))) & 255;

                float* buffer = Buffer + f + flag;

                int offsetx = 8 * (binx & 127);
                int offsety = 8 * (biny & 127);
                int offsetz = 8 * (binz & 127);
                int offsetw = 8 * (binw & 127);

                const bool passx = binx != 128;
                const bool passy = biny != 128;
                const bool passz = binz != 128;
                const bool passw = binw != 128;

                const float val1x = passx ? stat1.x : 0.0f;
                const float val1y = passy ? stat1.y : 0.0f;
                const float val1z = passz ? stat1.z : 0.0f;
                const float val1w = passw ? stat1.w : 0.0f;

                const float val2x = passx ? stat2.x : 0.0f;
                const float val2y = passy ? stat2.y : 0.0f;
                const float val2z = passz ? stat2.z : 0.0f;
                const float val2w = passw ? stat2.w : 0.0f;

                const int writeTime = (threadIdx.x >> 3) & 3;

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (k == writeTime) {
                        buffer[offsetx] += val1x;
                        buffer[offsety] += val1y;
                        buffer[offsetz] += val1z;
                        buffer[offsetw] += val1w;
                    }
                    syncTile.sync();
                }

                buffer += flag ? - 1 : 1;

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (k == writeTime) {
                        buffer[offsetx] += val2x;
                        buffer[offsety] += val2y;
                        buffer[offsetz] += val2z;
                        buffer[offsetw] += val2w;
                    }
                    syncTile.sync();
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
                const int fold0 = (threadIdx.x >> 1) & 31;

                const int maxFoldCount = 128;

                {
                    const volatile float* __restrict__ src = Buffer
                                                             + 1024  //warpHistSize
                                                             + 2 * f
                                                             + w;

                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        int fold = fold0 + 32 * k;
                        Buffer[2 * (maxFoldCount * f + fold) + w] = src[8 * fold];
                    }
                }
            }
            __syncthreads();
        }
    };



    DEFINE_NON_BINARY(7)
}
