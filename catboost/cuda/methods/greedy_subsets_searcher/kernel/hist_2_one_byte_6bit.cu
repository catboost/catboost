#include "hist.cuh"
#include "hist_2_one_byte_base.cuh"

#include "tuning_policy_enums.cuh"
#include <cooperative_groups.h>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    template <int BlockSize>
    struct TPointHist2OneByte<6, BlockSize> : public TPointHist2OneByteBase<TPointHist2OneByte<6, BlockSize>, BlockSize> {
        using TParent = TPointHist2OneByteBase<TPointHist2OneByte<6, BlockSize>, BlockSize>;
        using  TPointHist2OneByteBase<TPointHist2OneByte<6, BlockSize>, BlockSize>::Histogram;

        __forceinline__ __device__ TPointHist2OneByte(float* buffer)
            : TPointHist2OneByteBase<TPointHist2OneByte<6, BlockSize>, BlockSize>(buffer) {

        }

        __forceinline__ __device__ int SliceOffset() {
            const int warpId = (threadIdx.x / 32);
            const int warpOffset = 1024 * warpId;
            const int blocks = 2;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << 4));
            return warpOffset + innerHistStart;
        }

        template <int N>
        __forceinline__ __device__ void AddPointsImpl(const ui32* ci,
                                                      const float* s1,
                                                      const float* s2) {

            thread_block_tile<16> syncTile = tiled_partition<16>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            float stat1[N];
            float stat2[N];

            #pragma unroll
            for (int k = 0; k < N; ++k) {
                stat1[k] = flag ? s2[k] : s1[k];
                stat2[k] = flag ? s1[k] : s2[k];
            }


            float val1[N];
            float val2[N];

            int offset[N];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = ((2 * i + threadIdx.x) & 6);

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    const int bin = (ci[k] >> (24 - (f << 2))) & 255;
                    const float pass = bin != 64 ? 1.0f : 0.0f;

                    val1[k] = pass * stat1[k];
                    val2[k] = pass * stat2[k];

                    offset[k] = f +  16 * (bin & 62) + 8 * (bin & 1) + flag;
                }

                const bool writeFirstFlag = threadIdx.x & 8;

                syncTile.sync();

                if (writeFirstFlag) {
                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        Histogram[offset[k]] += val1[k];
                    }
                }

                syncTile.sync();

                if (!writeFirstFlag) {
                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        Histogram[offset[k]] += val1[k];
                    }
                }

                int shift = flag ? -1 : 1;
                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    offset[k] += shift;
                }

                syncTile.sync();


                if (writeFirstFlag) {
                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        Histogram[offset[k]] += val2[k];
                    }
                }

                syncTile.sync();

                if (!writeFirstFlag) {
                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        Histogram[offset[k]] += val2[k];
                    }
                }
            }
        }

        static constexpr int MaxBits()  {
            return 6;
        }

        __forceinline__ __device__ void Reduce() {
            TParent::ReduceToOneWarp();

            if (threadIdx.x < 256) {
                const int isSecondStat = threadIdx.x & 1;
                const int f = threadIdx.x / 64;
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                const int fold0 = (threadIdx.x >> 1) & 31;
                const int maxFoldCount = 64;

                {
                    const int innerHistCount = 2;
                    const volatile float* __restrict__ src =  Histogram
                        + 2048 //warpHistSize
                        + 2 * f
                        + 8 * (fold0 & 1)
                        + 32 * (fold0 >> 1)
                        + isSecondStat;

                    #pragma unroll
                    for (int inWarpHist = 0; inWarpHist < innerHistCount; ++inWarpHist) {
                        sum0 += src[(inWarpHist << 4)];
                        sum1 += src[(inWarpHist << 4) + 512];
                    }

                    Histogram[maxFoldCount * 4 * isSecondStat + maxFoldCount * f + fold0] = sum0;
                    Histogram[maxFoldCount * 4 * isSecondStat + maxFoldCount * f + fold0 + 32] = sum1;
                }
            }
            __syncthreads();
        }
    };

    DefineHist2Pass(6)



}
