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
    struct TPointHist2OneByte<5, BlockSize> : public TPointHist2OneByteBase<TPointHist2OneByte<5, BlockSize>, BlockSize> {
        using TParent = TPointHist2OneByteBase<TPointHist2OneByte<5, BlockSize>, BlockSize>;
        using  TPointHist2OneByteBase<TPointHist2OneByte<5, BlockSize>, BlockSize>::Histogram;

        __forceinline__ __device__ TPointHist2OneByte(float* buffer)
        : TPointHist2OneByteBase<TPointHist2OneByte<5, BlockSize>, BlockSize>(buffer) {

        }

        __forceinline__ __device__ int SliceOffset() {
            const int warpId = (threadIdx.x / 32);
            const int warpOffset = 1024 * warpId;
            const int blocks = 4;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << 3));
            return warpOffset + innerHistStart;
        }


        template <int N>
        __forceinline__ __device__ void AddPointsImpl(const ui32* ci,
                                                      const float* s1,
                                                      const float* s2) {

            thread_block_tile<8> syncTile = tiled_partition<8>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            float stat1[N];
            float stat2[N];

            #pragma unroll
            for (int k = 0; k < N; ++k) {
                stat1[k] = flag ? s2[k] : s1[k];
                stat2[k] = flag ? s1[k] : s2[k];
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = ((2 * i + threadIdx.x) & 6);

                int offsets[N];
                bool pass[N];

                #pragma unroll
                for (int k =0; k < N; ++k) {
                    const int bin = (ci[k] >> (24 - (f << 2))) & 255;
                    offsets[k] = f + 32 * (bin & 31);
                    pass[k] = bin != 32;
                }

                syncTile.sync();

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    int offset = offsets[k];
                    const int offset1 = offset + flag;
                    const float add1 = pass[k] ? stat1[k] : 0.0f;
                    Histogram[offset1] += add1;
                }

                syncTile.sync();

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    int offset = offsets[k];
                    const int offset2 = offset + !flag;
                    const float add2 = pass[k] ? stat2[k] : 0.0f;

                    Histogram[offset2] += add2;
                }
            }
        }

        static constexpr int MaxBits()  {
            return 5;
        }

        __forceinline__ __device__ void Reduce() {
            TParent::ReduceToOneWarp();

            if (threadIdx.x < 256) {
                const int isSecondStat = threadIdx.x & 1;
                const int f = threadIdx.x / 64;
                float sum = 0.0f;
                const int fold = (threadIdx.x >> 1) & 31;
                const int maxFoldCount = 32;

                if (fold < maxFoldCount) {
                    const int innerHistCount = 4;
                    const volatile float* __restrict__ src =  Histogram
                                                             + 2048 //warpHistSize
                                                             + 32 * fold
                                                             + 2 * f
                                                             + isSecondStat;

                    #pragma unroll
                    for (int inWarpHist = 0; inWarpHist < innerHistCount; ++inWarpHist) {
                        sum += src[(inWarpHist << 3)];
                    }

                   Histogram[maxFoldCount * 4 * isSecondStat + maxFoldCount * f + fold] = sum;
                }
            }
            __syncthreads();
        }
    };


    DefineHist2Pass(5)



}
