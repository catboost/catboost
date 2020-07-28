#pragma once

#include "tuning_policy_enums.cuh"
#include <cooperative_groups.h>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    template <int BlockSize, class TImpl>
    struct TPointHistHalfByteBase {
        float* Histogram;

        static constexpr int GetHistSize() {
            return BlockSize * 16;
        }

        static constexpr int AddPointsBatchSize() {
            return TLoadSize<LoadSize()>::Size();
        }

        static constexpr int Unroll(ECIndexLoadType type) {
            return TImpl::Unroll(type);
        }

        static constexpr int GetBlockSize() {
            return BlockSize;
        }

        static constexpr ELoadSize LoadSize() {
            #if __CUDA_ARCH__ < 500
            return ELoadSize::FourElements;
            #elif __CUDA_ARCH__ < 700
            return ELoadSize::TwoElements;
            #else
            return ELoadSize::FourElements;
            #endif
        }

        static constexpr int BlockLoadSize(ECIndexLoadType indexLoadType) {
            return TLoadSize<LoadSize()>::Size() * BlockSize * Unroll(indexLoadType);
        }

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 512 * (threadIdx.x / 32);
            const int innerHistStart = threadIdx.x & 24;
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHistHalfByteBase(float* buff) {
            const int histSize = 16 * BlockSize;

            for (int i = threadIdx.x; i < histSize; i += BlockSize) {
                buff[i] = 0;
            }
            __syncthreads();

            Histogram = buff + SliceOffset();
        }

        __forceinline__ __device__ void AddPoint(ui32 ci, const float t) {
            thread_block_tile<8> addToHistTile = tiled_partition<8>(this_thread_block());

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int f = (threadIdx.x + i) & 7;
                int bin = (ci >> (28 - 4 * f)) & 15;//bfe(ci, 28 - 4 * f, 4);
                bin <<= 5;
                bin += f;
                Histogram[bin] += t;
                addToHistTile.sync();
            }
        }

        template <int N>
        __forceinline__ __device__ void AddPointsImpl(const ui32* ci, const float* t) {

            thread_block_tile<8> addToHistTile = tiled_partition<8>(this_thread_block());

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int f = (threadIdx.x + i) & 7;
                int bin[N];
                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    bin[k] = (ci[k] >> (28 - 4 * f)) & 15;
                    bin[k] <<= 5;
                    bin[k] += f;
                }

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    Histogram[bin[k]] += t[k];
                }
                addToHistTile.sync();
            }
        }

        template <int N>
        __forceinline__ __device__ void AddPoints(const ui32* ci, const float* t) {
            const int NN = AddPointsBatchSize();
            static_assert(N % NN == 0, "Error: incorrect stripe size");

            #pragma unroll
            for (int k = 0; k < N; k += NN) {
                AddPointsImpl<NN>(ci + k, t + k);
            }
        }


        __forceinline__ __device__ void Reduce() {
            Histogram -= SliceOffset();

            __syncthreads();
            {
                const int histSize = 16 * BlockSize;
                float sum = 0;

                if (threadIdx.x < 512) {
                    for (int i = threadIdx.x; i < histSize; i += 512) {
                        sum += Histogram[i];
                    }
                }
                __syncthreads();

                if (threadIdx.x < 512) {
                    Histogram[threadIdx.x] = sum;
                }
                __syncthreads();
            }

            const int fold = (threadIdx.x >> 3) & 15;
            float sum = 0.0f;

            if (threadIdx.x < 128) {
                const int featureId = threadIdx.x & 7;

                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    sum += Histogram[32 * fold + featureId + 8 * group];
                }
            }

            __syncthreads();

            if (threadIdx.x < 128) {
                //featureId + 8 * fold
                Histogram[threadIdx.x] = sum;
            }

            __syncthreads();
        }

    };



}
