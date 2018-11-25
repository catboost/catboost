#include "histogram_utils.cuh"
#include <cooperative_groups.h>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <contrib/libs/cub/cub/warp/warp_scan.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    __global__ void CopyHistogramsImpl(const ui32* leftLeaves,
                                       const ui32* rightLeaves,
                                       ui32 numStats,
                                       ui32 binFeaturesInHist,
                                       float* histograms) {

        const ui32 leftLeafId = leftLeaves[blockIdx.y];
        const ui32 rightLeafId = rightLeaves[blockIdx.y];

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        float* srcHist = histograms + leftLeafId * binFeaturesInHist * numStats;
        float* dstHist = histograms + rightLeafId * binFeaturesInHist * numStats;

        const ui32 histSize = binFeaturesInHist * numStats;

        while (i < histSize) {
            dstHist[i] = srcHist[i];
            i += gridDim.x * blockDim.x;
        }
    }

    void CopyHistograms(const ui32* leftLeaves,
                        const ui32* rightLeaves,
                        const ui32 leavesCount,
                        ui32 numStats,
                        ui32 binFeaturesInHist,
                        float* histograms,
                        TCudaStream stream
    ) {

        const ui32 histSize = numStats * binFeaturesInHist;
        ui32 blockSize = 256;
        dim3 numBlocks;
        numBlocks.z = 1;
        numBlocks.y = leavesCount;
        numBlocks.x = CeilDivide(histSize, blockSize);

        if (numBlocks.x) {
            CopyHistogramsImpl<<<numBlocks, blockSize, 0, stream>>>(leftLeaves, rightLeaves,  numStats, binFeaturesInHist, histograms);
        }
    }

    //write histogram block to histograms
    __global__ void WriteReducesHistogramsImpl(int histBlockOffset,
                                               int binFeaturesInBlock,
                                               const ui32* histogramIds,
                                               const float* blockHistogram,
                                               const int binFeatureCount,
                                               float* dstHistogram) {

        const int binFeatureId = blockIdx.x * blockDim.x + threadIdx.x;

        const int leafId = blockIdx.y;
        const int statId = blockIdx.z;
        const size_t statCount = gridDim.z;

        const int dstId = histogramIds[blockIdx.y];


        if (binFeatureId < binFeaturesInBlock) {
            blockHistogram += binFeatureId;
            blockHistogram += binFeaturesInBlock * statId;
            blockHistogram += leafId * binFeaturesInBlock * statCount;
            const float val = __ldg(blockHistogram);


            dstHistogram += dstId * binFeatureCount * statCount;
            dstHistogram += statId * binFeatureCount;
            dstHistogram += histBlockOffset + binFeatureId;

            dstHistogram[0] = val;
        }
    }


    void WriteReducesHistograms(int blockOffset,
                                int histBlockSize,
                                const ui32* histogramIds,
                                ui32 leafCount,
                                ui32 statCount,
                                const float* blockHistogram,
                                const int binFeatureCount,
                                float* dstHistogram,
                                TCudaStream stream) {
        const int blockSize = 128;
        dim3 numBlocks;
        numBlocks.x = CeilDivide(histBlockSize, blockSize);
        numBlocks.y = leafCount;
        numBlocks.z = statCount;

        if (histBlockSize && leafCount && statCount) {
            WriteReducesHistogramsImpl<<<numBlocks, blockSize, 0, stream>>>(blockOffset,
                                                                            histBlockSize,
                                                                            histogramIds,
                                                                            blockHistogram,
                                                                            binFeatureCount,
                                                                            dstHistogram);
        }
    }


    //write histogram block to histograms
    __global__ void ZeroHistogramsImpl(const ui32* histIds,
                                       const int binFeatureCount,
                                       float* dstHistogram) {

        const int binFeatureId = blockIdx.x * blockDim.x + threadIdx.x;
        const int statId = blockIdx.z;
        const size_t statCount = gridDim.z;
        const int dstHist = histIds[blockIdx.y];


        if (binFeatureId < binFeatureCount) {
            dstHistogram += dstHist * binFeatureCount * statCount;
            dstHistogram += statId * binFeatureCount;
            dstHistogram[binFeatureId] = 0;
        }
    }


    void ZeroHistograms(const ui32* histIds,
                        ui32 idsCount,
                        ui32 statCount,
                        const int binFeatureCount,
                        float* dstHistogram,
                        TCudaStream stream) {
        const int blockSize = 256;
        dim3 numBlocks;
        numBlocks.x = CeilDivide(binFeatureCount, blockSize);
        numBlocks.y = idsCount;
        numBlocks.z = statCount;

        if (idsCount && statCount) {
            ZeroHistogramsImpl<<<numBlocks, blockSize, 0, stream>>>(histIds,
                                                                    binFeatureCount,
                                                                    dstHistogram);
        }
    }


    //write histogram block to histograms
    __global__ void SubstractHistogramsImpl(const ui32* fromIds,
                                            const ui32* whatIds,
                                            const int binFeatureCount,
                                            float* histogram) {

        const int binFeatureId = blockIdx.x * blockDim.x + threadIdx.x;
        const int fromId = fromIds[blockIdx.y];
        const int whatId = whatIds[blockIdx.y];
        const int statId = blockIdx.z;
        const size_t statCount = gridDim.z;



        histogram += binFeatureId;
        if (binFeatureId < binFeatureCount) {
            const ui64 fromOffset = fromId * binFeatureCount * statCount + statId * binFeatureCount;
            const ui64 whatOffset = whatId * binFeatureCount * statCount + statId * binFeatureCount;
            float newVal =  histogram[fromOffset] - histogram[whatOffset];
            if (statId == 0) {
                newVal = max(newVal, 0.0f);
            }
            histogram[fromOffset]  = newVal;
        }
    }


    void SubstractHistgorams(const ui32* fromIds,
                             const ui32* whatIds,
                             const int idsCount,
                             const int statCount,
                             const int binFeatureCount,
                             float* dstHistogram,
                             TCudaStream stream) {
        const int blockSize = 256;
        dim3 numBlocks;
        numBlocks.x = CeilDivide(binFeatureCount, blockSize);
        numBlocks.y = idsCount;
        numBlocks.z = statCount;

        if (idsCount && statCount) {
            SubstractHistogramsImpl<<<numBlocks, blockSize, 0, stream>>>(fromIds, whatIds, binFeatureCount, dstHistogram);
        }
    }


    template <int BlockSize>
    __global__ void ScanHistogramsImpl(const TBinarizedFeature* features, int featureCount,
                                       const ui32* histIds,
                                       const int binFeatureCount,
                                       float* histograms) {

        const int featuresPerBlock = BlockSize / 32;
        using WarpScan = cub::WarpScan<double>;
        __shared__ typename WarpScan::TempStorage tempStorage[featuresPerBlock];

        const int warpId = threadIdx.x / 32;
        const int threadIdInWarp = threadIdx.x & 31;
        const int featureId = blockIdx.x * featuresPerBlock + warpId;

        const int histId = histIds[blockIdx.y];
        const int statId = blockIdx.z;
        const ui64 statCount = gridDim.z;


        if (featureId < featureCount) {
            features += featureId;
            const bool skipFeature = features->OneHotFeature || (features->Folds <= 1);

            if (!skipFeature) {
                histograms += histId * binFeatureCount * statCount + statId * binFeatureCount + features->FirstFoldIndex;

                const int folds = features->Folds;

                const int n = ((folds + 31) / 32) * 32;

                double prefixSum = 0;

                for (int binOffset = 0; binOffset < n; binOffset += 32) {
                    const double val = (binOffset + threadIdInWarp) < folds
                                      ? histograms[(binOffset + threadIdInWarp)]
                                      : 0.0f;

                    double sum = 0;
                    __syncwarp();
                    WarpScan(tempStorage[warpId]).InclusiveSum(val, sum);
                    __syncwarp();

                    sum += prefixSum;

                    if ((binOffset + threadIdInWarp) < folds) {
                        histograms[binOffset + threadIdInWarp] = sum;
                    }

                    if ((binOffset + 32) < n) {
                        prefixSum = cub::ShuffleIndex<32, double>(sum, 31, 0xffffffff);
                    }
                }
            }
        }
    };



    void ScanHistgorams(const TBinarizedFeature* features, int fCount,
                        const ui32* ids,
                        const int idsCount,
                        const int statCount,
                        const int binFeatureCount,
                        float* histograms,
                        TCudaStream stream) {
        const int blockSize = 256;
        dim3 numBlocks;
        numBlocks.x = CeilDivide(fCount * 32, blockSize);
        numBlocks.y = idsCount;
        numBlocks.z = statCount;

        if (idsCount && statCount) {
            ScanHistogramsImpl<blockSize><<<numBlocks, blockSize, 0, stream>>>(features, fCount, ids, binFeatureCount, histograms);
        }
    }


}
