#pragma once
#include "hist.cuh"

#include "tuning_policy_enums.cuh"
#include "compute_hist_loop_two_stats.cuh"
#include <cooperative_groups.h>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    template <class TImpl, int BlockSize>
    struct TPointHist2OneByteBase {
        float* Histogram;

        static constexpr int GetHistSize() {
            return BlockSize * 32;
        }

        static constexpr int AddPointsBatchSize() {
            return TLoadSizeHist2<LoadSize()>::Size();
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
            return TLoadSizeHist2<LoadSize()>::Size() * BlockSize * Unroll(indexLoadType);
        }

        __forceinline__ __device__ TPointHist2OneByteBase(float* hist) {
            const int histSize = 32 * BlockSize;

            #pragma unroll 8
            for (int i = threadIdx.x; i < histSize; i += BlockSize) {
                hist[i] = 0;
            }

            TImpl* impl = static_cast<TImpl*>(this);
            Histogram = hist + impl->SliceOffset();

            __syncthreads();
        }


        template <int N>
        __forceinline__ __device__ void AddPoints(const ui32* ci, const float* s1, const float* s2) {
            const int NN = AddPointsBatchSize();
            static_assert(N % NN == 0, "Error: incorrect stripe size");
            TImpl* impl = static_cast<TImpl*>(this);

            #pragma unroll
            for (int k = 0; k < N; k += NN) {
                impl->AddPointsImpl<NN>(ci + k, s1 + k, s2 + k);
            }
        }

        __forceinline__ __device__ void AddPoint(const ui32 ci,
                                                 const float s1,
                                                 const float s2) {
            TImpl* impl = static_cast<TImpl*>(this);
            impl->AddPointsImpl<1>(&ci, &s1, &s2);
        }

        __forceinline__ __device__ void ReduceToOneWarp() {
            TImpl* impl = static_cast<TImpl*>(this);
            Histogram -= impl->SliceOffset();
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
                const int isSecondStatFlag = threadIdx.x >= 128;
                const int fid = (threadIdx.x & 127) / 32;
                const int firstFoldIdx = threadIdx.x & 31;
                const int histSize = 1 << TImpl::MaxBits();

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



    template <int Bits, bool IsOdd>
    void ComputeHist2OneByteBits(
                            const TFeatureInBlock* features,
                            const int fCount,
                            const TDataPartition* parts,
                            const ui32* partIds,
                            ui32 partCount,
                            const ui32* bins,
                            ui32 binsLineSize,
                            const float* stats,
                            ui32 numStats,
                            ui32 statLineSize,
                            float* histograms,
                            TCudaStream stream) {

        const int blockSize =  384;
        CB_ENSURE(numStats % 2 == IsOdd);

        dim3 numBlocks;
        numBlocks.z = (numStats - IsOdd) / 2;
        numBlocks.y = partCount;

        const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;
        const int maxActiveBlocks = blocksPerSm * TArchProps::SMCount();

        numBlocks.x = (fCount + 3) / 4;
        numBlocks.x *= CeilDivide(maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        using THist = TPointHist2OneByte<Bits, blockSize>;
        if (partCount) {
            ComputeSplitPropertiesDirectLoadsTwoStastImpl < THist, blockSize, 4, IsOdd ><<<numBlocks, blockSize, 0, stream>>>(
                    features,
                    fCount,
                    bins, binsLineSize,
                    stats,
                    statLineSize,
                    parts,
                    partIds,
                    histograms);
        }
    }


    template <int Bits, bool IsOdd>
    void ComputeHist2OneByteBits(
        const TFeatureInBlock* features,
        const int fCount,
        const TDataPartition* parts,
        const ui32 partId,
        const ui32* bins,
        ui32 binsLineSize,
        const float* stats,
        ui32 numStats,
        ui32 statLineSize,
        float* histograms,
        TCudaStream stream) {

        const int blockSize =  384;
        CB_ENSURE(numStats % 2 == IsOdd);

        dim3 numBlocks;
        numBlocks.z = (numStats - IsOdd) / 2;
        numBlocks.y = 1;

        const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;
        const int maxActiveBlocks = blocksPerSm * TArchProps::SMCount();

        numBlocks.x = (fCount + 3) / 4;
        numBlocks.x *= CeilDivide(maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        using THist = TPointHist2OneByte<Bits, blockSize>;
        {
            ComputeSplitPropertiesDirectLoadsTwoStastImpl < THist, blockSize, 4, IsOdd ><<<numBlocks, blockSize, 0, stream>>>(
                features,
                    fCount,
                    bins, binsLineSize,
                    stats,
                    statLineSize,
                    parts,
                    partId,
                    histograms);
        }
    }


    template <int Bits, bool IsOdd>
    void ComputeHist2OneByteBits(const TFeatureInBlock* features,
                                 const int fCount,
                                 const TDataPartition* parts,
                                 const ui32* partIds,
                                 ui32 partCount,
                                 const ui32* cindex,
                                 const int* indices,
                                 const float* stats,
                                 ui32 numStats,
                                 ui32 statLineSize,
                                 float* histograms,
                                 TCudaStream stream) {
        const int blockSize =  384;
        dim3 numBlocks;
        CB_ENSURE(numStats % 2 == IsOdd);

        numBlocks.z = (numStats - IsOdd) / 2;
        numBlocks.y = partCount;
        const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;
        const int maxActiveBlocks = blocksPerSm * TArchProps::SMCount();
        numBlocks.x = (fCount + 3) / 4;
        numBlocks.x *= CeilDivide(maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));
        if (IsGridEmpty(numBlocks)) {
            return;
        }
        using THist = TPointHist2OneByte<Bits, blockSize>;

        if (partCount) {
            ComputeSplitPropertiesTwoStatsGatherImpl<THist, blockSize, 4, IsOdd><<<numBlocks, blockSize, 0, stream>>>(features,
                        fCount,
                        cindex,
                        indices,
                        stats,
                        statLineSize,
                        parts,
                        partIds,
                        histograms);
        }
    }


    template <int Bits, bool IsOdd>
    void ComputeHist2OneByteBits(const TFeatureInBlock* features,
                                 const int fCount,
                                 const TDataPartition* parts,
                                 const ui32 partId,
                                 const ui32* cindex,
                                 const int* indices,
                                 const float* stats,
                                 ui32 numStats,
                                 ui32 statLineSize,
                                 float* histograms,
                                 TCudaStream stream) {
        const int blockSize =  384;
        dim3 numBlocks;
        CB_ENSURE(numStats % 2 == IsOdd);

        numBlocks.z = (numStats - IsOdd) / 2;
        numBlocks.y = 1;
        const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;
        const int maxActiveBlocks = blocksPerSm * TArchProps::SMCount();
        numBlocks.x = (fCount + 3) / 4;
        numBlocks.x *= CeilDivide(2 * maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));
        if (IsGridEmpty(numBlocks)) {
            return;
        }
        using THist = TPointHist2OneByte<Bits, blockSize>;

        {
            ComputeSplitPropertiesTwoStatsGatherImpl<THist, blockSize, 4, IsOdd><<<numBlocks, blockSize, 0, stream>>>(features,
                fCount,
                cindex,
                indices,
                stats,
                statLineSize,
                parts,
                partId,
                histograms);
        }
    }

    #define DefineHist2Pass_(Bits, IsOdd) \
    template void ComputeHist2OneByteBits<Bits, IsOdd>(const TFeatureInBlock* features, \
                                 const int fCount,\
                                 const TDataPartition* parts,\
                                 const ui32* partIds,\
                                 ui32 partCount,\
                                 const ui32* cindex,\
                                 const int* indices,\
                                 const float* stats,\
                                 ui32 numStats,\
                                 ui32 statLineSize,\
                                 float* histograms,\
                                 TCudaStream stream);\
    template void ComputeHist2OneByteBits<Bits, IsOdd>(const TFeatureInBlock* features,\
        const int fCount,\
        const TDataPartition* parts,\
        const ui32* partIds,\
        ui32 partCount,\
        const ui32* bins,\
        ui32 binsLineSize,\
        const float* stats,\
        ui32 numStats,\
        ui32 statLineSize,\
        float* histograms,\
        TCudaStream stream);\
    template void ComputeHist2OneByteBits<Bits, IsOdd>(const TFeatureInBlock* features, \
                                 const int fCount,\
                                 const TDataPartition* parts,\
                                 const ui32 partId,\
                                 const ui32* cindex,\
                                 const int* indices,\
                                 const float* stats,\
                                 ui32 numStats,\
                                 ui32 statLineSize,\
                                 float* histograms,\
                                 TCudaStream stream);\
    template void ComputeHist2OneByteBits<Bits, IsOdd>(const TFeatureInBlock* features,\
        const int fCount,\
        const TDataPartition* parts,\
        const ui32 partId,\
        const ui32* bins,\
        ui32 binsLineSize,\
        const float* stats,\
        ui32 numStats,\
        ui32 statLineSize,\
        float* histograms,\
        TCudaStream stream);

    #define DefineHist2Pass(Bits)\
    DefineHist2Pass_(Bits, false)\
    DefineHist2Pass_(Bits, true)

    #define DefineExternHist2Pass(Bits, IsOdd) \
    extern template void ComputeHist2OneByteBits<Bits, IsOdd>(const TFeatureInBlock* features, \
                                 const int fCount,\
                                 const TDataPartition* parts,\
                                 const ui32* partIds,\
                                 ui32 partCount,\
                                 const ui32* cindex,\
                                 const int* indices,\
                                 const float* stats,\
                                 ui32 numStats,\
                                 ui32 statLineSize,\
                                 float* histograms,\
                                 TCudaStream stream);\
    extern template void ComputeHist2OneByteBits<Bits, IsOdd>(const TFeatureInBlock* features,\
        const int fCount,\
        const TDataPartition* parts,\
        const ui32* partIds,\
        ui32 partCount,\
        const ui32* bins,\
        ui32 binsLineSize,\
        const float* stats,\
        ui32 numStats,\
        ui32 statLineSize,\
        float* histograms,\
        TCudaStream stream);\
        extern template void ComputeHist2OneByteBits<Bits, IsOdd>(const TFeatureInBlock* features, \
                                 const int fCount,\
                                 const TDataPartition* parts,\
                                 const ui32 partId,\
                                 const ui32* cindex,\
                                 const int* indices,\
                                 const float* stats,\
                                 ui32 numStats,\
                                 ui32 statLineSize,\
                                 float* histograms,\
                                 TCudaStream stream);\
    extern template void ComputeHist2OneByteBits<Bits, IsOdd>(const TFeatureInBlock* features,\
        const int fCount,\
        const TDataPartition* parts,\
        const ui32 partId,\
        const ui32* bins,\
        ui32 binsLineSize,\
        const float* stats,\
        ui32 numStats,\
        ui32 statLineSize,\
        float* histograms,\
        TCudaStream stream);


    DefineExternHist2Pass(5, false)
    DefineExternHist2Pass(5, true)

    DefineExternHist2Pass(6, false)
    DefineExternHist2Pass(6, true)

    DefineExternHist2Pass(7, false)
    DefineExternHist2Pass(7, true)

    #undef DefineExternHist2Pass

}
