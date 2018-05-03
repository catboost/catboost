#include "pairwise_hist.cuh"
#include "split_properties_helpers.cuh"

#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

namespace NKernel {

   __forceinline__  __device__ uint2 ZeroPair() {
        uint2 pair;
        pair.x = pair.y = 0;
        return pair;
    }

    template <int STRIPE_SIZE, int HIST_BLOCK_COUNT, int N, int OUTER_UNROLL, int BLOCKS_PER_FEATURE, typename THist>
    __forceinline__  __device__ void ComputePairHistogram(ui32 offset,
                                                          const ui32* __restrict cindex,
                                                          int dsSize,
                                                          const uint2* __restrict pairs,
                                                          const float* __restrict weight,
                                                          float*  histogram) {

        THist hist(histogram);

        int i = (threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32;
        pairs += offset;
        weight += offset;

        //all operations should be warp-aligned
        //first: first warp make memory access aligned. it load first 32 - offset % 32 elements.
        {
            int lastId = min(dsSize, 32 - (offset & 31));

            if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0) {
                const uint2 pair = i < lastId ? __ldg(pairs + i) : ZeroPair();
                const ui32 ci1 = i < lastId ? __ldg(cindex + pair.x) : 0;
                const ui32 ci2 = i < lastId ? __ldg(cindex + pair.y) : 0;
                const float w = i < lastId ? __ldg(weight + i) : 0;
                hist.AddPair(ci1, ci2,  w);
            }
            dsSize = max(dsSize - lastId, 0);

            pairs += lastId;
            weight += lastId;
        }

        //now lets align end
        const int unalignedTail = (dsSize & 31);

        if (unalignedTail != 0) {
            if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0)
            {
                const int tailOffset = dsSize - unalignedTail;
                const uint2 pair = i < unalignedTail ? __ldg(pairs + tailOffset + i) : ZeroPair();
                const ui32 ci1 = i < unalignedTail ? __ldg(cindex + pair.x) : 0;
                const ui32 ci2 = i < unalignedTail ? __ldg(cindex + pair.y) : 0;
                const float w = i < unalignedTail ? __ldg(weight + tailOffset + i) : 0;
                hist.AddPair(ci1, ci2, w);
            }
        }
        dsSize -= unalignedTail;

        if (blockIdx.x % BLOCKS_PER_FEATURE == 0 && dsSize <= 0) {
            __syncthreads();
            hist.Reduce();
            return;
        }

        pairs += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;
        weight += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;

        dsSize = max(dsSize - (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE, 0);
        const int stripe = STRIPE_SIZE * BLOCKS_PER_FEATURE;

        const int iteration_count = (dsSize - i + stripe - 1)  / stripe;
        const int blocked_iteration_count = ((dsSize - (i | 31) + stripe - 1) / stripe) / N;

        if (dsSize) {
            pairs += i;
            weight += i;

            #pragma unroll OUTER_UNROLL
            for (int j = 0; j < blocked_iteration_count; ++j) {
                ui32 local_ci[N * 2];
                float local_w[N];

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    uint2 p = __ldg(pairs + stripe * k);
                    local_ci[k] = __ldg(cindex + p.x);
                    local_ci[k + N] = __ldg(cindex + p.y);
                    local_w[k] = __ldg(weight + stripe * k);
                }

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    hist.AddPair(local_ci[k], local_ci[k + N], local_w[k]);
                }

                pairs += stripe * N;
                weight += stripe * N;
            }

            for (int k = blocked_iteration_count * N; k < iteration_count; ++k) {
                const uint2 p = __ldg(pairs);
                const ui32 ci1 = __ldg(cindex + p.x);
                const ui32 ci2 = __ldg(cindex + p.y);
                const float w = __ldg(weight);
                hist.AddPair(ci1, ci2, w);

                pairs += stripe;
                weight += stripe;
            }
            hist.Reduce();
        }
        __syncthreads();
    }


}
