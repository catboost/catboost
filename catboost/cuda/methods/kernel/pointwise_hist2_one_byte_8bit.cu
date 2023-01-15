#include "pointwise_hist2_one_byte_templ.cuh"
#include <cooperative_groups.h>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    template <>
    struct TLoadEntriesTrait<3, false> {
        constexpr static ELoadType LoadType() {
            return ELoadType::OneElement;
        }
    };

    template <>
    struct TLoadEntriesTrait<3, true> {
        constexpr static ELoadType LoadType() {
            #if __CUDA_ARCH__ < 520
            return ELoadType::OneElement;
            #else
            return ELoadType::TwoElements;
            #endif
        }
    };

    template <>
    struct TDeclarePassInnerOuterBitsTrait<3> {
        constexpr static int Inner() {
            return 1;
        }

        constexpr static int Outer() {
            return 2;
        }
    };


    template <int BLOCK_SIZE>
    struct TPointHist<2, 1, BLOCK_SIZE> {
        constexpr static int OUTER_HIST_BITS_COUNT = 2;
        constexpr static int INNER_HIST_BITS_COUNT = 1;
        float* __restrict__ Buffer;

        float mostRecentStat1[4];
        float mostRecentStat2[4];
        uchar mostRecentBin[4];

        __forceinline__ __device__ int SliceOffset() {

            const int maxBlocks = BLOCK_SIZE * 32 / (1024 << OUTER_HIST_BITS_COUNT);
            static_assert(OUTER_HIST_BITS_COUNT <= 2, "Error: assume 12 warps, so limited by 128-bin histogram per warp");
            static_assert(OUTER_HIST_BITS_COUNT > 0 && INNER_HIST_BITS_COUNT > 0, "This histogram is specialized for 255 bin count");

            const int warpId = (threadIdx.x / 32) % maxBlocks;
            const int warpOffset = (1024 << OUTER_HIST_BITS_COUNT) * warpId;
            const int blocks = 4 >> INNER_HIST_BITS_COUNT;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << (INNER_HIST_BITS_COUNT + 3)));
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
            #pragma unroll
            for (int f = 0; f < 4; ++f) {
                mostRecentBin[f] = 0;
                mostRecentStat1[f] = 0;
                mostRecentStat2[f] = 0;
            }
        }

        __forceinline__ __device__ void Add(float val, float* dst) {
            atomicAdd(dst, val);
        }

        __forceinline__ __device__ void AddPoint(ui32 ci, const float t, const float w) {
            const bool flag = threadIdx.x & 1;

            const float stat1 = flag ? t : w;
            const float stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = ((2 * i + threadIdx.x) & 6);
                const uchar bin = bfe(ci, 24 - (f << 2), 8);

                if (bin != mostRecentBin[i]) {
                    int offset = f;
                    const uchar mask = (1 << INNER_HIST_BITS_COUNT) - 1;
                    offset += 8 * (mostRecentBin[i] & mask);
                    offset += 32 * ((mostRecentBin[i] >> INNER_HIST_BITS_COUNT));

                    offset += flag;
                    Add(mostRecentStat1[i], Buffer + offset);
                    offset = flag ? offset - 1 : offset + 1;
                    Add(mostRecentStat2[i], Buffer + offset);

                    mostRecentBin[i] = bin;
                    mostRecentStat1[i] = 0;
                    mostRecentStat2[i] = 0;
                }

                {
                    mostRecentStat1[i] += stat1;
                    mostRecentStat2[i] += stat2;
                }
            }
        }

        __forceinline__ __device__ void AddPoint2(uint2 bin, const float2 t, const float2 w) {
            AddPoint(bin.x, t.x, w.x);
            AddPoint(bin.y, t.y, w.y);
        }

        __forceinline__ __device__ void AddPoint4(uint4 ci, const float4 t, const float4 w) {
            AddPoint(ci.x, t.x, w.x);
            AddPoint(ci.y, t.y, w.y);
            AddPoint(ci.z, t.z, w.z);
            AddPoint(ci.w, t.w, w.w);
        }

        //After reduce we store histograms by blocks: 256 floats (4 x 2 x 32)
        // for first 32 bins; than 256 floats for second 32 bins, etc
        __forceinline__ __device__ void Reduce() {
            {
                const bool flag = threadIdx.x & 1;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const short f = ((2 * i + threadIdx.x) & 6);
                    int offset = f;
                    const int mask = (1 << INNER_HIST_BITS_COUNT) - 1;
                    offset += 8 * (mostRecentBin[i] & mask);
                    offset += 32 * ((mostRecentBin[i] >> INNER_HIST_BITS_COUNT));

                    Add(mostRecentStat1[i], Buffer + offset + flag);
                    Add(mostRecentStat2[i], Buffer + offset + !flag);
                }
            }


            Buffer -= SliceOffset();
            __syncthreads();

            {
                const int warpHistSize = 1024 << OUTER_HIST_BITS_COUNT;
                const int maxBlocks = BLOCK_SIZE * 32 / (1024 << OUTER_HIST_BITS_COUNT);

                for (int start = threadIdx.x; start < warpHistSize; start += BLOCK_SIZE) {
                    float sum = 0;
//                    12 iterations at 32-bin
                    #pragma unroll maxBlocks
                    for (int i = start; i < 32 * BLOCK_SIZE; i += warpHistSize) {
                        sum += Buffer[i];
                    }
                    Buffer[warpHistSize + start] = sum;
                }
            }

            __syncthreads();

            if (threadIdx.x < 256) {
                const int w = threadIdx.x & 1;

                float sum[4];

                const int maxFoldCount = (1 << (5 + INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT));
                for (int fold = (threadIdx.x >> 1); fold < maxFoldCount; fold += 128) {

                    #pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        sum[f] = 0;
                    }

                    const int innerHistCount = 4 >> INNER_HIST_BITS_COUNT;
                    const int lowBitMask = (1 << INNER_HIST_BITS_COUNT) - 1;
                    const float* __restrict__ src = Buffer
                                                    + (1024 << OUTER_HIST_BITS_COUNT)  //warpHistSize
                                                    + 8 * (fold & lowBitMask)
                                                    + 32 * (fold >> INNER_HIST_BITS_COUNT)
                                                    + w;

                    #pragma unroll
                    for (int inWarpHist = 0; inWarpHist < innerHistCount; ++inWarpHist) {
                        #pragma unroll
                        for (int f = 0; f < 4; ++f) {
                            sum[f] += src[2 * f + (inWarpHist << (3 + INNER_HIST_BITS_COUNT))];
                        }
                    }

                    #pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        Buffer[2 * (maxFoldCount * f + fold) + w] = sum[f];
                    }
                }
            }
            __syncthreads();
        }
    };

    DEFINE_NON_BINARY(8)
}
