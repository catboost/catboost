#include "hist.cuh"
#include "point_hist_half_byte_template.cuh"
#include "compute_hist_loop_one_stat.cuh"

using namespace cooperative_groups;

namespace NKernel {

    template <int BlockSize>
    struct TPointHistHalfByte: public TPointHistHalfByteBase<BlockSize, TPointHistHalfByte<BlockSize>> {
        using TPointHistHalfByteBase<BlockSize, TPointHistHalfByte<BlockSize>>::Histogram;

        __forceinline__ __device__ TPointHistHalfByte(float* buff)
                : TPointHistHalfByteBase<BlockSize,TPointHistHalfByte<BlockSize>>(buff) {

        }


        static constexpr int Unroll(ECIndexLoadType) {
            #if __CUDA_ARCH__ < 500
            return 4;
            #else
            return 1;
            #endif
        }

        __forceinline__ __device__ void AddToGlobalMemory(int statId, int statCount, int blockCount,
                                                          const TFeatureInBlock* features,
                                                          int fCount,
                                                          int leafId, int leafCount,
                                                          float* binSums) {

            const int fid = threadIdx.x >> 4;
            const int fold = threadIdx.x & 15;

            if (fid < fCount && fold < features[fid].Folds) {
                TFeatureInBlock group = features[fid];

                const int deviceOffset = group.GroupOffset * statCount * leafCount;
                const int entriesPerLeaf = statCount * group.GroupSize;

                float* dst = binSums + deviceOffset + leafId * entriesPerLeaf + statId * group.GroupSize + group.FoldOffsetInGroup;
                const float val = Histogram[fid + 8 * fold];

                if (abs(val) > 1e-20f) {
                    if (blockCount > 1) {
                        atomicAdd(dst + fold, val);
                    } else {
                        dst[fold] = val;
                    }
                }
            }
        }
    };

    using THist = TPointHistHalfByte<768>;


    void ComputeHistHalfByte(const TFeatureInBlock* features,
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

        const int blockSize =  768;
        dim3 numBlocks;
        numBlocks.z = numStats;
        numBlocks.y = partCount;

        const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;
        const int maxActiveBlocks = blocksPerSm * TArchProps::SMCount();

        numBlocks.x = (fCount + 7) / 8;
        numBlocks.x *= CeilDivide(maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        ComputeSplitPropertiesDirectLoadsImpl<THist, blockSize, 8><<<numBlocks, blockSize, 0, stream>>>(
                        features,
                        fCount,
                        bins,
                        binsLineSize,
                        stats,
                        statLineSize,
                        parts,
                        partIds,
                        histograms);


    }

    void ComputeHistHalfByte(const TFeatureInBlock* features,
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
        const int blockSize =  768;
        dim3 numBlocks;
        numBlocks.z = numStats;
        numBlocks.y = partCount;

        const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;
        const int maxActiveBlocks = blocksPerSm * TArchProps::SMCount();

        numBlocks.x = (fCount + 7) / 8;
        numBlocks.x *= CeilDivide(maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        ComputeSplitPropertiesGatherImpl<THist, blockSize, 8><<<numBlocks, blockSize, 0, stream>>>(
                        features,
                        fCount,
                        cindex,
                        indices,
                        stats,
                        statLineSize,
                        parts,
                        partIds,
                        histograms);

    }



    /* one part */

    void ComputeHistHalfByte(const TFeatureInBlock* features,
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

        const int blockSize =  768;
        dim3 numBlocks;
        numBlocks.z = numStats;
        numBlocks.y = 1;

        const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;
        const int maxActiveBlocks = blocksPerSm * TArchProps::SMCount();

        numBlocks.x = (fCount + 7) / 8;
        numBlocks.x *= CeilDivide(maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        ComputeSplitPropertiesDirectLoadsImpl<THist, blockSize, 8><<<numBlocks, blockSize, 0, stream>>>(
            features,
                fCount,
                bins,
                binsLineSize,
                stats,
                statLineSize,
                parts,
                partId,
                histograms);


    }

    void ComputeHistHalfByte(const TFeatureInBlock* features,
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
        const int blockSize =  768;
        dim3 numBlocks;
        numBlocks.z = numStats;
        numBlocks.y = 1;

        const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;
        const int maxActiveBlocks = blocksPerSm * TArchProps::SMCount();

        numBlocks.x = (fCount + 7) / 8;
        numBlocks.x *= CeilDivide(maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        ComputeSplitPropertiesGatherImpl<THist, blockSize, 8><<<numBlocks, blockSize, 0, stream>>>(
            features,
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
