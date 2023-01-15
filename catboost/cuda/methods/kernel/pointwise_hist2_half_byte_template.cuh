#pragma once
#include <cooperative_groups.h>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    template <int BlockSize>
    struct TPointHistHalfByte {
        float* Buffer;
        thread_block_tile<32> SyncTile;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 512 * (threadIdx.x / 32);
            const int innerHistStart = threadIdx.x & 16;
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHistHalfByte(float* buff)
        : SyncTile(tiled_partition<32>(this_thread_block())) {
            const int HIST_SIZE = 16 * BlockSize;
            for (int i = threadIdx.x; i < HIST_SIZE; i += BlockSize) {
                buff[i] = 0;
            }
            __syncthreads();

            Buffer = buff + SliceOffset();
        }

        __forceinline__ __device__ void AddPoint(ui32 ci, const float t, const float w) {
            const bool flag = threadIdx.x & 1;
            const float addFirst = flag ? t : w;
            const float addSecond = flag ? w : t;

            const int shift = threadIdx.x & 14;
            const ui32 bins = RotateRight(ci, 2 * shift);

            #if __CUDA_ARCH__ < 700
            #pragma unroll
            #endif
            for (int i = 0; i < 8; i++) {
                const int f = (shift + (i << 1)) & 14;
                int offset = (bins >> (28 - 4 * i)) & 15;
                offset <<= 5;
                offset += f;

                SyncTile.sync();
                Buffer[offset + flag] += addFirst;

                SyncTile.sync();
                Buffer[offset + !flag] += addSecond;
            }
        }

        __forceinline__ __device__ void AddPoint2(uint2 bin, const float2 t, const float2 w) {
            AddPoint(bin.x, t.x, w.x);
            AddPoint(bin.y, t.y, w.y);
        }

        __device__ void Reduce() {
            Buffer -= SliceOffset();
            const int warpCount = BlockSize >> 5;

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

            if (threadIdx.x < 256) {
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
