#pragma once
#include "split_properties_helpers.cuh"

#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

namespace NKernel {



    template <ui32 STRIPE_SIZE, ui32 OUTER_UNROLL, ui32 N, ui32 HIST_BLOCK_COUNT, ui32 BLOCKS_PER_FEATURE, typename THist>
    __forceinline__ __device__ void ComputeHistogram(const ui32* __restrict__ indices, ui32 offset, ui32 dsSize,
                                                     const float* __restrict__ target, const float* __restrict__ weight,
                                                     const ui32* __restrict__ cindex, float* __restrict__ result) {

        weight += offset;
        target += offset;
        indices += offset;

        THist hist(result);

        if (dsSize  == 0) {
            return;
        }

        ui32 i = (threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32;

        //all operations should be warp-aligned
        //first: first warp make memory access aligned. it load first 32 - offset % 32 elements.
        {
            ui32 lastId = min(dsSize, 32 - (offset & 31));

            if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0) {
                const ui32 index = i < lastId ? __ldg(indices + i) : 0;
                const ui32 ci = i < lastId ? __ldg(cindex + index) : 0;
                const float w = i < lastId ? __ldg(weight + i) : 0;
                const float wt = i < lastId ? __ldg(target + i) : 0;
                hist.AddPoint(ci, wt, w);
            }
            dsSize = dsSize > lastId ? dsSize - lastId : 0;

            indices += lastId;
            target += lastId;
            weight += lastId;
        }

        //now lets align end
        const ui32 unalignedTail = (dsSize & 31);

        if (unalignedTail != 0) {
            if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0)
            {
                const ui32 tailOffset = dsSize - unalignedTail;
                const ui32 index = i < unalignedTail ? __ldg(indices + tailOffset + i) : 0;
                const ui32 ci = i < unalignedTail ? __ldg(cindex + index) : 0;
                const float w = i < unalignedTail ? __ldg(weight + tailOffset + i) : 0;
                const float wt = i < unalignedTail ? __ldg(target + tailOffset + i) : 0;
                hist.AddPoint(ci, wt, w);
            }
        }
        dsSize -= unalignedTail;

        if (blockIdx.x % BLOCKS_PER_FEATURE == 0 && dsSize <= 0) {
            __syncthreads();
            hist.Reduce();
            return;
        }

        indices += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;
        target += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;
        weight += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;

        dsSize = dsSize > (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE ? dsSize - (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE : 0;
        const ui32 stripe = STRIPE_SIZE * BLOCKS_PER_FEATURE;

        if (dsSize) {

            ui32 iteration_count = dsSize > i ? (dsSize - i + (stripe - 1)) / stripe : 0;
            ui32 blocked_iteration_count = dsSize > (i | 31) ? ((dsSize - (i | 31) + (stripe - 1)) / stripe) / N : 0;

            weight += i;
            target += i;
            indices += i;

#pragma unroll OUTER_UNROLL
            for (ui32 j = 0; j < blocked_iteration_count; ++j) {
                ui32 local_index[N];
#pragma unroll
                for (ui32 k = 0; k < N; k++) {
                    local_index[k] = __ldg(indices + stripe * k);
                }

                ui32 local_ci[N];
                float local_w[N];
                float local_wt[N];

#pragma unroll
                for (ui32 k = 0; k < N; ++k) {
                    local_ci[k] = __ldg(cindex + local_index[k]);
                    local_w[k] = __ldg(weight + stripe * k);
                    local_wt[k] = __ldg(target + stripe * k);
                }

                indices += stripe * N;
                target += stripe * N;
                weight += stripe * N;

#pragma unroll
                for (ui32 k = 0; k < N; ++k) {
                    hist.AddPoint(local_ci[k], local_wt[k], local_w[k]);
                }
            }

            for (ui32 k = blocked_iteration_count * N; k < iteration_count; ++k) {
                const int index = __ldg(indices);
                ui32 ci = __ldg(cindex + index);
                float w = __ldg(weight);
                float wt = __ldg(target);

                indices += stripe;
                target += stripe;
                weight += stripe;
                hist.AddPoint(ci, wt, w);
            }
            __syncthreads();

            hist.Reduce();
        }
    }


    template <ui32 STRIPE_SIZE, ui32 OUTER_UNROLL, ui32 HIST_BLOCK_COUNT, ui32 BLOCKS_PER_FEATURE, typename THist>
    __forceinline__ __device__ void ComputeHistogram2(
            const ui32* __restrict__ indices,
            ui32 offset, ui32 dsSize,
            const float* __restrict__ target, const float* __restrict__ weight,
            const ui32* __restrict__ cindex, float* __restrict__ result) {

        weight += offset;
        target += offset;
        indices += offset;

        THist hist(result);

        if (dsSize) {
            //first: first warp make memory access aligned. it load first 32 - offset % 32 elements.
            {
                ui32 lastId = min(dsSize, 128 - (offset & 127));
                ui32 colId = (threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32;

                if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0)
                {
                    for (; colId < 128; colId += blockDim.x / HIST_BLOCK_COUNT)
                    {
                        const ui32 index = colId < lastId ? __ldg(indices + colId) : 0;
                        const ui32 ci = colId < lastId  ? __ldg(cindex + index) : 0;
                        const float w = colId < lastId  ? __ldg(weight + colId) : 0;
                        const float wt = colId < lastId  ? __ldg(target + colId) : 0;
                        hist.AddPoint(ci, wt, w);
                    }
                }

                dsSize = dsSize > lastId ? dsSize - lastId : 0;

                indices += lastId;
                target += lastId;
                weight += lastId;
            }

            //now lets align end
            const ui32 unalignedTail = (dsSize & 63);

            if (unalignedTail != 0) {
                if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0)
                {
                    ui32 colId = (threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32;
                    const ui32 tailOffset = dsSize - unalignedTail;

                    for (; colId < 64; colId += blockDim.x / HIST_BLOCK_COUNT)
                    {
                        const ui32 index = colId < unalignedTail ? __ldg(indices + tailOffset + colId) : 0;
                        const ui32 ci = colId < unalignedTail ? __ldg(cindex + index) : 0;
                        const float w = colId < unalignedTail ? __ldg(weight + tailOffset + colId) : 0;
                        const float wt = colId < unalignedTail ? __ldg(target + tailOffset + colId) : 0;
                        hist.AddPoint(ci, wt, w);
                    }
                }
            }

            dsSize -= unalignedTail;

            if (dsSize <= 0) {
                if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0) {
                    __syncthreads();
                    hist.Reduce();
                }
                return;
            }


            indices += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 2;
            target += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 2;
            weight += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 2;

            constexpr ui32 stripe = STRIPE_SIZE * BLOCKS_PER_FEATURE * 2;
            dsSize = dsSize > (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 2 ? dsSize - (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 2 : 0;

            if (dsSize) {
                ui32 iterCount;
                {
                    const ui32 i = 2 * ((threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32);
                    weight += i;
                    target += i;
                    indices += i;
                    iterCount = dsSize > i ? (dsSize - i + (stripe - 1)) / stripe : 0;
                }

                #pragma unroll OUTER_UNROLL
                for (ui32 j = 0; j < iterCount; ++j) {
                    const uint2 localIndices = __ldg((uint2*) indices);
                    uint2 bin;
                    bin.x = __ldg(cindex + localIndices.x);
                    bin.y = __ldg(cindex + localIndices.y);
                    const float2 localTarget = __ldg((float2* )(target));
                    const float2 localWeight = __ldg((float2* )(weight));

                    indices += stripe;
                    target += stripe;
                    weight += stripe;

                    hist.AddPoint2(bin, localTarget, localWeight);
                }
                __syncthreads();
                hist.Reduce();
            }
        }
    }






    template <int STRIPE_SIZE, int OUTER_UNROLL, int HIST_BLOCK_COUNT, int BLOCKS_PER_FEATURE, typename THist>
    __forceinline__ __device__ void ComputeHistogram4(
            const ui32* __restrict__ indices,
            ui32 offset, ui32 dsSize,
            const float* __restrict__ target, const float* __restrict__ weight,
            const ui32* __restrict__ cindex, float* __restrict__ result) {

        weight += offset;
        target += offset;
        indices += offset;

        THist hist(result);

        if (dsSize) {
            //first: first warp make memory access aligned. it load first 32 - offset % 32 elements.
            {
                ui32 lastId = min(dsSize, 128 - (offset & 127));
                ui32 colId = (threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32;

                if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0)
                {
                    for (; colId < 128; colId += blockDim.x / HIST_BLOCK_COUNT)
                    {
                        const ui32 index = colId < lastId ? __ldg(indices + colId) : 0;
                        const ui32 ci = colId < lastId  ? __ldg(cindex + index) : 0;
                        const float w = colId < lastId  ? __ldg(weight + colId) : 0;
                        const float wt = colId < lastId  ? __ldg(target + colId) : 0;
                        hist.AddPoint(ci, wt, w);
                    }
                }

                dsSize = dsSize > lastId ? dsSize - lastId : 0;

                indices += lastId;
                target += lastId;
                weight += lastId;
            }

            //now lets align end
            const ui32 unalignedTail = (dsSize & 127);

            if (unalignedTail != 0) {
                if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0)
                {
                    ui32 colId = (threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32;
                    const ui32 tailOffset = dsSize - unalignedTail;

                    for (; colId < 128; colId += blockDim.x / HIST_BLOCK_COUNT)
                    {
                        const ui32 index = colId < unalignedTail ? __ldg(indices + tailOffset + colId) : 0;
                        const ui32 ci = colId < unalignedTail ? __ldg(cindex + index) : 0;
                        const float w = colId < unalignedTail ? __ldg(weight + tailOffset + colId) : 0;
                        const float wt = colId < unalignedTail ? __ldg(target + tailOffset + colId) : 0;
                        hist.AddPoint(ci, wt, w);
                    }
                }
            }

            dsSize -= unalignedTail;

            if (dsSize <= 0) {
                if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0) {
                    __syncthreads();
                    hist.Reduce();
                }
                return;
            }


            indices += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 4;
            target += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 4;
            weight += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 4;

            constexpr ui32 stripe = STRIPE_SIZE * BLOCKS_PER_FEATURE * 4;
            dsSize = dsSize > (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 4 ? dsSize - (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 4 : 0;

            __syncthreads();

            if (dsSize) {
                ui32 iterCount;
                {
                    const ui32 i = 4 * ((threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32);
                    weight += i;
                    target += i;
                    indices += i;
                    iterCount = dsSize > i ? (dsSize - i + (stripe - 1)) / stripe : 0;
                }

                #pragma unroll OUTER_UNROLL
                for (ui32 j = 0; j < iterCount; ++j) {
                    const uint4 localIndices = __ldg((uint4*) indices);
                    uint4 bin;
                    bin.x =  __ldg(cindex + localIndices.x);
                    bin.y =  __ldg(cindex + localIndices.y);
                    bin.z =  __ldg(cindex + localIndices.z);
                    bin.w =  __ldg(cindex + localIndices.w);
                    const float4 localTarget = __ldg((float4*)(target));
                    const float4 localWeight = __ldg((float4*)(weight));


                    indices += stripe;
                    target += stripe;
                    weight += stripe;

                    hist.AddPoint4(bin, localTarget, localWeight);
                }
                __syncthreads();
                hist.Reduce();
            }
        }
    }


}
