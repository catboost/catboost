#pragma once
#include <cooperative_groups.h>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    template<int BLOCK_SIZE>
    struct TPointHistHalfByte {
        float* Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 512 * (threadIdx.x / 32);
            const int innerHistStart = threadIdx.x & 16;
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHistHalfByte(float* buff) {
            const int HIST_SIZE = 16 * BLOCK_SIZE;
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE) {
                buff[i] = 0;
            }
            __syncthreads();

            Buffer = buff + SliceOffset();
        }

        __forceinline__ __device__ void AddPoint(ui32 ci, const float t, const float w) {
            thread_block_tile<32> syncTile = tiled_partition<32>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            #if __CUDA_ARCH__ >= 700
            const int UNROLL = 4;
            #else
            const int UNROLL = 8;
            #endif

            #pragma unroll UNROLL
            for (int i = 0; i < 8; i++) {
                const short f = (threadIdx.x + (i << 1)) & 14;
                short bin = bfe(ci, 28 - (f << 1), 4);
                bin <<= 5;
                bin += f;
                const int offset0 = bin + flag;
                const int offset1 = bin + !flag;
                syncTile.sync();
                Buffer[offset0] += (flag ? t : w);
                syncTile.sync();
                Buffer[offset1] += (flag ? w : t);
            }
        }

        __forceinline__ __device__ void AddPoint2(uint2 bin, const float2 t, const float2 w) {
            AddPoint(bin.x, t.x, w.x);
            AddPoint(bin.y, t.y, w.y);
        }

        __device__ void Reduce() {
            Buffer -= SliceOffset();
            const int warpCount = BLOCK_SIZE >> 5;

            {
                const int fold = (threadIdx.x >> 5) & 15;
                const int sumOffset = threadIdx.x & 31;


                float sum = 0.0;
                if (threadIdx.x < 512)
                {
                    float* __restrict__ buffer = const_cast<float*>(Buffer);

                    #pragma unroll
                    for (int warpId = 0; warpId < warpCount; ++warpId)
                    {
                        const int warpOffset = 512 * warpId;
                        sum += buffer[warpOffset + sumOffset + 32 * fold];
                    }
                }
                __syncthreads();

                if (threadIdx.x < 512) {
                    Buffer[threadIdx.x] = sum;
                }
            }

            __syncthreads();
            const int fold = (threadIdx.x >> 4) & 15;
            float sum = 0.0f;

            if (threadIdx.x < 256)
            {
                const int histEntryId = (threadIdx.x & 15);
                sum = Buffer[32 * fold + histEntryId] + Buffer[32 * fold + histEntryId + 16];
            }

            __syncthreads();

            if (threadIdx.x < 256) {
                Buffer[threadIdx.x] = sum;
            }

            __syncthreads();
        }
    };

}
