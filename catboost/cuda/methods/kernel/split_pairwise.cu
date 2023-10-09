#include "split_pairwise.cuh"
#include "split_properties_helpers.cuh"
#include <cooperative_groups.h>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

#include <cmath>

using namespace cooperative_groups;
namespace NKernel {

    __forceinline__ __device__ void AddToMatrices(int row, int col, float sum,
                                                  float* matrix) {
        const int ind  = col < row ? (row * (row + 1) >> 1) + col : (col * (col + 1) >> 1) + row;
        matrix[ind] += sum;
    }


    template <int BLOCK_SIZE, int PartCount>
    __global__ void MakePairwiseDerivatives(const float* pairwiseHistogram,
                                            int matrixOffset,
                                            int matCount,
                                            int histLineSize /* 4 * totalBinFeatureCount */,
                                            float* linearSystem) {

        const int logicalWarpSize =  PartCount > 32 ? 32 : PartCount;
        const int matricesPerBlock = BLOCK_SIZE / logicalWarpSize;

        int matrixIdx = blockIdx.x * matricesPerBlock + threadIdx.x / logicalWarpSize;
        int localTid = threadIdx.x & (logicalWarpSize - 1);

        if (matrixIdx >= matCount)
            return;

        {
            const size_t rowSize = PartCount * 2;
            const size_t linearSystemSize = (rowSize + rowSize * (rowSize + 1) / 2);
            linearSystem += matrixIdx * linearSystemSize;
        }
        pairwiseHistogram += (matrixOffset + matrixIdx) * 4;


        const int N = PartCount / logicalWarpSize;
        thread_block_tile<logicalWarpSize> groupTile = tiled_partition<logicalWarpSize>(this_thread_block());

        double sum0[N];
        double sum1[N];
        for (int i = 0; i < N; ++i) {
            sum0[i] = 0;
            sum1[i] = 0;
        }


        #pragma unroll 16
        for (int y = 0; y < PartCount; ++y) {
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                const int x = localTid + 32 * i;
                const int partIdx = ConvertBlockToPart(x, y);
                ui64 offset = ((ui64) partIdx * histLineSize * 4ULL);
                float4 hist = __ldg((float4*)(pairwiseHistogram + offset));

                const float w00 = max((x != y ? hist.x : 0.0f), 0.0f);
                const float w01 = max(hist.y, 0.0f);
                const float w10 = max(hist.z, 0.0f);
                const float w11 = max((x != y ? hist.w : 0.0f), 0.0f);

//                sync for row write done in reduce if we need it

                const int nextRow = 2 * y;
                const int nextCol = 2 * x;

                sum0[i] += w00 + w10;
                sum1[i] += w01 + w11;

                if (x == y) {
                    AddToMatrices(nextRow + 1, nextRow, -(w01 + w10), linearSystem);
                } else if (x < y) {
                    AddToMatrices(nextRow, nextCol, -w00, linearSystem);
                    AddToMatrices(nextRow, nextCol + 1, -w01, linearSystem);
                    AddToMatrices(nextRow + 1, nextCol, -w10, linearSystem);
                    AddToMatrices(nextRow + 1, nextCol + 1, -w11, linearSystem);
                }
                groupTile.sync();
            }

            groupTile.sync();
        }

        #pragma unroll 16
        for (int x = 0; x < PartCount; ++x) {

            #pragma unroll
            for (int i = 0; i < N; ++i) {
                const int y = localTid + 32 * i;
                const int partIdx = ConvertBlockToPart(x, y);
                ui64 offset = ((ui64) partIdx * histLineSize * 4ULL);
                float4 hist = __ldg((float4*)(pairwiseHistogram + offset));

                const float w00 = max((x != y ? hist.x : 0.0f), 0.0f);
                const float w01 = max(hist.y, 0.0f);
                const float w10 = max(hist.z, 0.0f);
                const float w11 = max((x != y ? hist.w : 0.0f), 0.0f);

//                sync for row write done in reduce if we need it

                const int nextRow = 2 * y;
                const int nextCol = 2 * x;

                sum0[i] += w01 + w00;
                sum1[i] += w10 + w11;

                if (x > y) {
                    AddToMatrices(nextRow, nextCol, -w00, linearSystem);
                    AddToMatrices(nextRow, nextCol + 1, -w01, linearSystem);
                    AddToMatrices(nextRow + 1, nextCol, -w10, linearSystem);
                    AddToMatrices(nextRow + 1, nextCol + 1, -w11, linearSystem);
                }
                groupTile.sync();
            }

            groupTile.sync();
        }

        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const int x = localTid + 32 * i;
            const int nextRow = 2 * x;
            linearSystem[nextRow * (nextRow + 1) / 2 + nextRow] += sum0[i];
            linearSystem[(nextRow + 1) * (nextRow + 2) / 2 + nextRow + 1] += sum1[i];
        }
    }

    template <int BLOCK_SIZE>
    void RunMakeMatrices(const float* histogram, int partCount, int histLineSize, int firstMatrix, int matricesCount, float* linearSystem, TCudaStream stream) {
        if (matricesCount > 0) {
            const int numBlocks = (((size_t) matricesCount) * min(partCount, 32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
            #define RUN(PartCount)\
            MakePairwiseDerivatives<BLOCK_SIZE, PartCount> << < numBlocks, BLOCK_SIZE, 0, stream >> > (histogram,  firstMatrix, matricesCount, histLineSize, linearSystem);

            if (partCount == 1) {
                RUN(1)
            } else if (partCount == 2) {
                RUN(2)
            } else if (partCount == 4) {
                RUN(4)
            } else if (partCount == 8) {
                RUN(8)
            } else if (partCount == 16) {
                RUN(16)
            } else if (partCount == 32) {
                RUN(32)
            } else if (partCount == 64) {
                RUN(64)
            } else if (partCount == 128) {
                RUN(128)
            } else if (partCount == 256) {
                RUN(256)
            } else {
                Y_ABORT_UNLESS(false);
            }
        }
    }

    void MakePairwiseDerivatives(const float* histogram, int leavesCount, int firstMatrix, int matricesCount, int histLineSize, float* linearSystem,
                                 TCudaStream stream) {
        RunMakeMatrices<256>(histogram, leavesCount, histLineSize, firstMatrix, matricesCount, linearSystem, stream);
    }

    template <int BLOCK_SIZE>
    __global__ void MakePointwiseDerivatives(const float* pointwiseHist, ui64 pointwiseHistSize,
                                             const TPartitionStatistics* partStats,
                                             bool hasPointwiseWeights,
                                             int rowSize,
                                             int firstMatrixIdx,
                                             int matCount,
                                             float* linearSystem) {

        const int lineSize = min(rowSize, 32);
        const int matricesPerBlock = BLOCK_SIZE / lineSize;

        const int matrixIdx = blockIdx.x * matricesPerBlock + threadIdx.x / lineSize;
        pointwiseHist += (firstMatrixIdx + matrixIdx) * (hasPointwiseWeights ? 2 : 1);
        linearSystem += ((size_t)matrixIdx) * (rowSize + rowSize * (rowSize + 1) / 2);

        const int x = threadIdx.x & (lineSize - 1);
        float* targets = linearSystem + rowSize * (rowSize + 1) / 2;

        if (matrixIdx < matCount) {
            for (int col = x; col < rowSize; col += 32) {
                const int i = col / 2;
                ui64 offset = pointwiseHistSize * i;

                if (hasPointwiseWeights) {
                    const float leafWeight = pointwiseHist[offset];
                    const float weight = (col & 1) ? partStats[i].Weight - leafWeight : leafWeight;
                    linearSystem[col * (col + 1) / 2 + col] += max(weight, 0.0f);
                }

                const float leafSum = pointwiseHist[offset + hasPointwiseWeights];
                const float sum = (col & 1) ? partStats[i].Sum - leafSum : leafSum;
                targets[col] = sum;
            }
        }
    }

    template <int BLOCK_SIZE>
    void RunMakePointwiseDerivatives(const float* pointwiseHist, int binFeatureCount,
                                     const TPartitionStatistics* partStats,
                                     bool hasPointwiseWeights,
                                     int rowSize,
                                     int firstMatrixIdx,
                                     int matricesCount,
                                     float* linearSystem,
                                     TCudaStream stream
    ) {
        if (matricesCount > 0) {
            const ui32 pointwiseHistSize = binFeatureCount * (hasPointwiseWeights ? 2 : 1);
            const int lineSize = min(32, rowSize);
            const int numBlocks = (((size_t) matricesCount) * lineSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            MakePointwiseDerivatives<BLOCK_SIZE> << < numBlocks, BLOCK_SIZE, 0, stream >> > (pointwiseHist, pointwiseHistSize, partStats, hasPointwiseWeights, rowSize, firstMatrixIdx, matricesCount,  linearSystem);
        }
    }

    void MakePointwiseDerivatives(const float* pointwiseHist, int pointwiseHistLineSize,
                                  const TPartitionStatistics* partStats,
                                  bool hasPointwiseWeights,
                                  int rowSize,
                                  int firstMatrixIdx,
                                  int matricesCount,
                                  float* linearSystem,
                                  TCudaStream stream) {
       RunMakePointwiseDerivatives<128> (pointwiseHist, pointwiseHistLineSize, partStats, hasPointwiseWeights, rowSize, firstMatrixIdx, matricesCount, linearSystem, stream);
    }


    __global__ void UpdateBinsPairs(TCFeature feature, ui32 binIdx,
                                    const ui32* cindex,
                                    const uint2* pairs,
                                    ui32 pairCount,
                                    ui32 depth,
                                    ui32* bins) {
        ui32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        cindex += feature.Offset;

        const ui32 value = binIdx << feature.Shift;
        const ui32 mask = feature.Mask << feature.Shift;

        while (idx < pairCount) {
            const uint2 p = pairs[idx];
            const ui32 d1 = (cindex[p.x] & mask);
            const ui32 d2 = (cindex[p.y] & mask);
            ui32 bit1 =  feature.OneHotFeature ? d1 == value : d1 > value;
            ui32 bit2 =  feature.OneHotFeature ? d2 == value : d2 > value;
            ui32 bin = bins[idx];
            bin = ((bit1 * 2 + bit2) << (depth * 2)) | bin;
            bins[idx] = bin;
            idx += blockDim.x * gridDim.x;
        }
    }

    void UpdateBinsPairs(TCFeature feature, ui32 bin,
                         const ui32* compressedIndex,
                         const uint2* pairs,
                         ui32 pairCount,
                         ui32 depth,
                         ui32* bins,
                         TCudaStream stream) {
        const ui32 blockSize = 256;
        const ui32 numBlocks = min((pairCount + blockSize - 1) / blockSize,
                                  TArchProps::MaxBlockCount());
        UpdateBinsPairs<<<numBlocks, blockSize, 0, stream>>>(feature, bin, compressedIndex, pairs, pairCount, depth, bins);
    }


    template <int BLOCK_SIZE>
    __global__ void SelectBestSplitImpl(const float* scores,
                                        const TCBinFeature* binFeature, int size,
                                        double scoreBeforeSplit, const float* featureWeights,
                                        int bestIndexBias, TBestSplitPropertiesWithIndex* best) {
        float maxScore = -INFINITY;
        float maxGain = -INFINITY;
        int maxIdx = -1;
        int tid = threadIdx.x;

        #pragma unroll 8
        for (int i = tid; i < size; i += BLOCK_SIZE) {
            float score = scores[i];
            auto featureId = binFeature[i].FeatureId;
            float gain = (score + scoreBeforeSplit) * __ldg(featureWeights + featureId); // scoreBeforeSplit is -score from some previous tree level
            if (gain > maxGain) {
                maxScore = score;
                maxGain = gain;
                maxIdx = i;
            }
        }

        __shared__ float vals[BLOCK_SIZE];
        __shared__ int inds[BLOCK_SIZE];
        __shared__ float gains[BLOCK_SIZE];

        vals[tid] = maxScore;
        inds[tid] = maxIdx;
        gains[tid] = maxGain;
        __syncthreads();

        for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                if ( gains[tid] <  gains[tid + s] || (gains[tid] == gains[tid + s] && inds[tid] > inds[tid + s]) ) {
                    vals[tid] = vals[tid + s];
                    inds[tid] = inds[tid + s];
                    gains[tid] = gains[tid + s];
                }
            }
            __syncthreads();
        }


        if (tid == 0) {
            TCBinFeature bestFeature;
            const int bestIdx = inds[0];
            const float bestScore = vals[0];

            if (bestIdx != -1) {
                bestFeature = binFeature[bestIdx];
            } else {
                bestFeature.BinId = static_cast<ui32>(-1);
                bestFeature.FeatureId = static_cast<ui32>(-1);
            }
            best->Index = bestIndexBias + bestIdx;
            best->Score = -bestScore;
            best->BinId = bestFeature.BinId;
            best->FeatureId = bestFeature.FeatureId;
            best->Gain = -gains[0];
        }
    }

    void SelectBestSplit(const float* scores,
                         const TCBinFeature* binFeature, int size,
                         double scoreBeforeSplit, const float* featureWeights,
                         int bestIndexBias, TBestSplitPropertiesWithIndex* best,
                         TCudaStream stream) {
        const int blockSize = 1024;
        SelectBestSplitImpl<blockSize><<<1, blockSize, 0, stream>>>(
            scores,  binFeature, size,
            scoreBeforeSplit, featureWeights,
            bestIndexBias, best);
    }



    __global__  void ZeroSameLeafBinWeightsImpl(const uint2* pairs,
                                                const ui32* bins,
                                                ui32 pairCount,
                                                float* pairWeights) {
        const ui32 i = blockDim.x * blockIdx.x + threadIdx.x;


        if (i < pairCount) {
            uint2 pair = pairs[i];
            const ui32 binx = bins[pair.x];
            const ui32 biny = bins[pair.y];
            if (binx == biny) {
                pairWeights[i] = 0;
            }
        }
    }

    void ZeroSameLeafBinWeights(const uint2* pairs,
                                const ui32* bins,
                                ui32 pairCount,
                                float* pairWeights,
                                TCudaStream stream
    ) {

        if (pairCount > 0) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (pairCount + blockSize - 1) / blockSize;
            ZeroSameLeafBinWeightsImpl<<<numBlocks, blockSize, 0, stream>>>(pairs, bins, pairCount, pairWeights);
        }
    }


    __global__  void FillPairBinsImpl(const uint2* pairs,
                                      const ui32* bins,
                                      ui32 rowSize,
                                      ui32 pairCount,
                                      ui32* pairBins) {
        const ui32 i = blockDim.x * blockIdx.x + threadIdx.x;


        if (i < pairCount) {
            uint2 pair = pairs[i];
            const ui32 binx = bins[pair.x];
            const ui32 biny = bins[pair.y];
            pairBins[i] = binx * rowSize + biny;
        }
    }


    void FillPairBins(const uint2* pairs,
                      const ui32* bins,
                      ui32 binCount,
                      ui32 pairCount,
                      ui32* pairBins,
                      TCudaStream stream) {
        if (pairCount > 0) {
            const int blockSize = 256;
            const ui32 numBlocks = (pairCount + blockSize - 1) / blockSize;
            FillPairBinsImpl<<<numBlocks, blockSize, 0, stream>>>(pairs, bins, binCount, pairCount, pairBins);
        }
    }




    //for leaves estimation
    __global__ void FillPairDer2OnlyImpl(const float* ders2,
                                         const float* groupDers2,
                                         const ui32* qids,
                                         const uint2* pairs,
                                         ui32 pairCount,
                                         float* pairDer2) {

        const ui32 tid = threadIdx.x;
        const ui32 i = blockIdx.x * blockDim.x + tid;

        if (i < pairCount) {
            uint2 pair = Ldg(pairs + i);

            const float der2x = Ldg(ders2 + pair.x);
            const float der2y = Ldg(ders2 + pair.y);
            const ui32 qid = Ldg(qids + pair.x);
            const float groupDer2 = Ldg(groupDers2 + qid);

            pairDer2[i] = groupDer2 > 1e-20f ? der2x * der2y / (groupDer2 + 1e-20f) : 0;
        }
    }





    void FillPairDer2Only(const float* ders2,
                          const float* groupDers2,
                          const ui32* qids,
                          const uint2* pairs,
                          ui32 pairCount,
                          float* pairDer2,
                          TCudaStream stream
    ) {
        const ui32 blockSize = 256;
        const ui32 numBlocks = (pairCount + blockSize - 1) / blockSize;
        if (numBlocks > 0) {
            FillPairDer2OnlyImpl<<< numBlocks, blockSize, 0, stream >>>(ders2, groupDers2, qids, pairs, pairCount, pairDer2);
        }
    }


}
