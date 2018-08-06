#include "hist.cuh"

#include "tuning_policy_enums.cuh"
#include <cooperative_groups.h>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    template<class TImpl, int BlockSize>
    struct TPointHist2OneByteBase {
        float* Histogram;

        static constexpr int GetHistSize() {
            return BlockSize * 32;
        }

        static constexpr int AddPointsBatchSize() {
            return TLoadSize<LoadSize()>::Size();
        }

        static constexpr int Unroll(ECIndexLoadType) {
            #if __CUDA_ARCH__ < 700
            const int NN = 1;
            #else
            const int NN = 2;
            #endif
            return NN;
        }

        static constexpr int GetBlockSize() {
            return BlockSize;
        }

        static constexpr ELoadSize LoadSize() {
            #if __CUDA_ARCH__ < 500
            return ELoadSize::OneElement;
            #else
            return ELoadSize::FourElements;
            #endif
        }

        static constexpr int BlockLoadSize(ECIndexLoadType indexLoadType) {
            return TLoadSize<LoadSize()>::Size() * BlockSize * Unroll(indexLoadType);
        }

        __forceinline__ __device__ TPointHist2OneByteBase(float* hist) {
            const int histSize = 32 * BlockSize;

            #pragma unroll 8
            for (int i = threadIdx.x; i < histSize; i += BlockSize) {
                hist[i] = 0;
            }

            Histogram = hist + static_cast<TImpl*>(this)->SliceOffset();

            __syncthreads();
        }


        template <int N>
        __forceinline__ __device__ void AddPoints(const ui32* ci, const float* s1, const float* s2) {
            const int NN = AddPointsBatchSize();
            static_assert(N % NN == 0, "Error: incorrect stripe size");

            #pragma unroll
            for (int k = 0; k < N; k += NN) {
                static_cast<TImpl*>(this)->AddPointsImpl<NN>(ci + k, s1 + k, s2 + k);
            }
        }

        __forceinline__ __device__ void ReduceToOneWarp() {
            Histogram -=  static_cast<TImpl*>(this)->SliceOffset();
            __syncthreads();
            {
                const int warpHistSize = 1024;
                for (int start = threadIdx.x; start < warpHistSize; start += BlockSize) {
                    float sum = 0;
                    //12 iterations
                    #pragma unroll 12
                    for (int i = start; i < 32 * BlockSize; i += warpHistSize) {
                        sum += Histogram[i];
                    }
                    Histogram[2048 + start] = sum;
                }
            }
            __syncthreads();
        }



        __forceinline__ __device__  int DstOffset(int statId, int statCount,
                                                  TFeatureInBlock& group,
                                                  int fCount,
                                                  int leafId,
                                                  int leafCount) {
                const int deviceOffset = group.GroupOffset * statCount * leafCount;
                const int entriesPerLeaf = statCount * group.GroupSize;
                return deviceOffset + leafId * entriesPerLeaf + statId * group.GroupSize + group.FoldOffsetInGroup;
        }

        __forceinline__ __device__  void AddToGlobalMemory(int statId, int statCount, int blockCount,
                                                           const TFeatureInBlock* features,
                                                           int fCount,
                                                           int leafId, int leafCount,
                                                           float* binSums) {

            if (threadIdx.x < 256) {
                const int isSecondStatFlag = threadIdx. x >= 128;
                const int fid = (threadIdx.x & 127) / 32;
                const int firstFoldIdx = threadIdx.x & 31;
                const int histSize = 1 << TImpl::MaxBits();

                #pragma unroll 4
                if (fid < fCount) {
                    TFeatureInBlock group = features[fid];
                    float* dst = binSums + DstOffset(statId + isSecondStatFlag, statCount, group, fCount, leafId, leafCount);

                    for (int fold = firstFoldIdx; fold < features[fid].Folds; fold += 32) {
                        const float val = Histogram[isSecondStatFlag * 4 * histSize + fid * histSize + fold];
                        if (abs(val) > 1e-20f) {
                            if (blockCount > 1) {
                                atomicAdd(dst + fold, val);
                            } else {
                                dst[fold] = val;
                            }
                        }
                    }
                }
            }
        }
    };


    template <int Bits, int BlockSize>
    struct TPointHist2OneByte;

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

        __forceinline__ __device__ void AddPoint(ui32 ci,
                                                 const float s1,
                                                 const float s2) {

            thread_block_tile<8> syncTile = tiled_partition<8>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const float stat1 = flag ? s2 : s1;
            const float stat2 = flag ? s1 : s2;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = ((2 * i + threadIdx.x) & 6);
                const int bin = (ci >> (24 - (f << 2))) & 255;
                const bool pass = bin != 32;
                int offset = f + 32 * (bin & 31);
                const int offset1 = offset + flag;
                const float add1 = pass ? stat1 : 0.0f;
                const int offset2 = offset + !flag;
                const float add2 = pass ? stat2 : 0.0f;

                syncTile.sync();
                Histogram[offset1] += add1;
                syncTile.sync();
                Histogram[offset2] += add2;
            }
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




}
