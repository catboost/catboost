#include "hist.cuh"
#include "point_hist_half_byte_template.cuh"
#include "compute_hist_loop_one_stat.cuh"

using namespace cooperative_groups;

namespace NKernel {

    template <int BlockSize>
    struct TPointHistBinary: public TPointHistHalfByteBase<BlockSize, TPointHistBinary<BlockSize>> {
        using TPointHistHalfByteBase<BlockSize, TPointHistBinary<BlockSize>>::Histogram;

        __forceinline__ __device__ TPointHistBinary(float* buff)
                : TPointHistHalfByteBase<BlockSize,TPointHistBinary<BlockSize>>(buff) {

        }

        static constexpr int Unroll(ECIndexLoadType) {
            #if __CUDA_ARCH__ < 500
            return 4;
            #elif __CUDA_ARCH__ < 700
            return 1;
            #else
            return 2;
            #endif
        }

        __forceinline__ __device__ void AddToGlobalMemory(int statId, int statCount, int blockCount,
                                                          const TFeatureInBlock* features,
                                                          int fCount,
                                                          int leafId, int leafCount,
                                                          float* binSums) {

            const int fid = threadIdx.x;
            const int fold = 0;

            if (fid < fCount ) {
                TFeatureInBlock group = features[fid];

                if (group.Folds) {
                    const int deviceOffset = group.GroupOffset * statCount * leafCount;
                    const int entriesPerLeaf = statCount * group.GroupSize;

                    float* dst = binSums + deviceOffset + leafId * entriesPerLeaf + statId * group.GroupSize +
                                 group.FoldOffsetInGroup;

                    const int groupId = fid / 4;
                    const int fMask = 1 << (3 - (fid & 3));

                    float val = 0.f;
                    #pragma uroll
                    for (int i = 0; i < 16; i++) {
                        if (!(i & fMask)) {
                            val += Histogram[8 * i + groupId];
                        }
                    }

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
    };

    using THist = TPointHistBinary<768>;


    void ComputeHistBinary(const TFeatureInBlock* features,
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

        numBlocks.x = (fCount + 31) / 32;
        numBlocks.x *= CeilDivide(maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        ComputeSplitPropertiesDirectLoadsImpl<THist, blockSize, 32><<<numBlocks, blockSize, 0, stream>>>(features,
                        fCount,
                        bins,
                        binsLineSize,
                        stats,
                        statLineSize,
                        parts,
                        partIds,
                        histograms);


    }

    void ComputeHistBinary(const TFeatureInBlock* features,
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

        numBlocks.x = (fCount + 31) / 32;
        numBlocks.x *= CeilDivide(maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        ComputeSplitPropertiesGatherImpl<THist, blockSize, 32><<<numBlocks, blockSize, 0, stream>>>(features,
                        fCount,
                        cindex,
                        indices,
                        stats,
                        statLineSize,
                        parts,
                        partIds,
                        histograms);

    }


    /* Single hist */

    void ComputeHistBinary(const TFeatureInBlock* features,
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

        numBlocks.x = (fCount + 31) / 32;
        numBlocks.x *= CeilDivide(maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        ComputeSplitPropertiesDirectLoadsImpl<THist, blockSize, 32><<<numBlocks, blockSize, 0, stream>>>(features,
            fCount,
            bins,
            binsLineSize,
            stats,
            statLineSize,
            parts,
            partId,
            histograms);


    }

    void ComputeHistBinary(const TFeatureInBlock* features,
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

        numBlocks.x = (fCount + 31) / 32;
        numBlocks.x *= CeilDivide(maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        ComputeSplitPropertiesGatherImpl<THist, blockSize, 32><<<numBlocks, blockSize, 0, stream>>>(features,
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
