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
    struct TPointHist2OneByte<7, BlockSize> : public TPointHist2OneByteBase<TPointHist2OneByte<7, BlockSize>, BlockSize> {
        using TParent = TPointHist2OneByteBase<TPointHist2OneByte<7, BlockSize>, BlockSize>;
        using  TPointHist2OneByteBase<TPointHist2OneByte<7, BlockSize>, BlockSize>::Histogram;

        __forceinline__ __device__ TPointHist2OneByte(float* buffer)
            : TPointHist2OneByteBase<TPointHist2OneByte<7, BlockSize>, BlockSize>(buffer) {

        }

        __forceinline__ __device__ int SliceOffset() {
            const int warpId = (threadIdx.x / 32);
            const int warpOffset = 1024 * warpId;
            return warpOffset;
        }

        template <int N>
        __forceinline__ __device__ void AddPointsImpl(const ui32* ci,
                                                      const float* s1,
                                                      const float* s2) {

            thread_block_tile<32> syncTile = tiled_partition<32>(this_thread_block());

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
                    const bool pass = bin != 128;
                    val1[k] = pass * stat1[k];
                    val2[k] = pass * stat2[k];
                    offset[k] = f + 8 * (bin & 127) + flag;
                }

                const int writeTime = (threadIdx.x >> 3) & 3;

                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (t > 0) {
                        syncTile.sync();
                    }

                    if (t == writeTime) {
                        #pragma unroll
                        for (int k = 0; k < N; ++k) {
                            Histogram[offset[k]] += val1[k];
                        }
                    }
                }

                int shift = flag ? -1 : 1;

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    offset[k] += shift;
                }

                syncTile.sync();

                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (t == writeTime) {
                        #pragma unroll
                        for (int k = 0; k < N; ++k) {
                            Histogram[offset[k]] += val2[k];
                        }
                    }
                    syncTile.sync();
                }
            }
        }

        static constexpr int MaxBits()  {
            return 7;
        }

        __forceinline__ __device__ void Reduce() {
            TParent::ReduceToOneWarp();

            if (threadIdx.x < 256) {
                const int isSecondStat = threadIdx.x & 1;
                const int f = threadIdx.x / 64;
                const int fold0 = (threadIdx.x >> 1) & 31;
                const int maxFoldCount = 128;

                const volatile float* __restrict__ src =  Histogram
                    + 2048 //warpHistSize
                    + 2 * f
                    + isSecondStat;



                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int fold = fold0 + 32 * k;
                    Histogram[maxFoldCount * 4 * isSecondStat + maxFoldCount * f + fold] =  src[8 * fold];
                }
            }
            __syncthreads();
        }
    };


    DefineHist2Pass(7)


}
