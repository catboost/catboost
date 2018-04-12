#include "split_pairwise.cuh"
#include "split_properties_helpers.cuh"
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>


namespace NKernel {

    __forceinline__ __device__ void AddToMatrices(int row, int col, float sum,
                                                  float* matrix) {
        const int ind  = col < row ? (row * (row + 1) >> 1) + col : (col * (col + 1) >> 1) + row;
        matrix[ind] += sum;
    }

    template <int BLOCK_SIZE>
    __global__ void MakePairwiseDerivatives(const float* pairwiseHistogram,
                                            int matrixOffset,
                                            int matCount,
                                            int partCount,
                                            int histLineSize /* 4 * totalBinFeatureCount */,
                                            float* linearSystem) {
        const int matricesPerBlock = BLOCK_SIZE / partCount;

        int matrixIdx = blockIdx.x * matricesPerBlock + threadIdx.x / partCount;
        int x = threadIdx.x & (partCount - 1);
        const int inBlockOffset = threadIdx.x / partCount;

        if (matrixIdx >= matCount)
            return;

        {
            const size_t rowSize = partCount * 2;
            const size_t linearSystemSize = (rowSize + rowSize * (rowSize + 1) / 2);
            linearSystem += matrixIdx * linearSystemSize;
        }
        pairwiseHistogram += (matrixOffset + matrixIdx) * 4;

        __shared__ float lineData[BLOCK_SIZE * 2];

        float* row0 = &lineData[inBlockOffset * partCount];
        float* row1 = &lineData[inBlockOffset * partCount + BLOCK_SIZE];


        float colSum0 = 0.0f;
        float colSum1 = 0.0f;

        for (int y = 0; y < partCount; ++y) {

            const int partIdx = ConvertBlockToPart(x, y);
            ui64 offset = ((ui64)partIdx * histLineSize * 4ULL);
            const float w00 = (x != y ? __ldg(pairwiseHistogram + offset) : 0.0f);
            const float w01 = __ldg(pairwiseHistogram + offset + 1);
            const float w10 = __ldg(pairwiseHistogram + offset + 2);
            const float w11 = (x != y ? __ldg(pairwiseHistogram + offset + 3) : 0.0f);

            row0[x] = w01 + w00;
            row1[x] = w10 + w11;

            //symc for row write done in reduce if we need it
            const float sum0 = FastInBlockReduce(x, row0, partCount);
            const float sum1 = FastInBlockReduce(x, row1, partCount);

            const int nextRow = 2 * y;
            const int nextCol = 2 * x;

            if (x == 0) {
                AddToMatrices(nextRow, nextRow, sum0, linearSystem);
                AddToMatrices(nextRow + 1, nextRow + 1, sum1, linearSystem);
            }

            colSum0 += w00 + w10;
            colSum1 += w01 + w11;

            if (x == y) {
                AddToMatrices(nextRow + 1, nextRow, -(w01 + w10), linearSystem);
            } else {
                AddToMatrices(nextRow, nextCol, -w00, linearSystem);
                AddToMatrices(nextRow , nextCol + 1, -w01, linearSystem);
                AddToMatrices(nextRow + 1, nextCol, -w10, linearSystem);
                AddToMatrices(nextRow + 1, nextCol + 1, -w11, linearSystem);
            }

            __syncthreads();
        }

        const int nextRow = 2 * x;
        linearSystem[nextRow * (nextRow + 1) / 2 + nextRow] += colSum0;
        linearSystem[(nextRow + 1) * (nextRow + 2) / 2 + nextRow + 1] += colSum1;
    }

    template <int BLOCK_SIZE>
    void RunMakeMatrices(const float* histogram, int partCount, int histLineSize, int firstMatrix, int matricesCount, float* linearSystem, TCudaStream stream) {
        if (matricesCount > 0) {
            const int numBlocks = (((size_t) matricesCount) * partCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            MakePairwiseDerivatives<BLOCK_SIZE> << < numBlocks, BLOCK_SIZE, 0, stream >> > (histogram,  firstMatrix, matricesCount,  partCount,  histLineSize, linearSystem);
        }
    }


    void MakePairwiseDerivatives(const float* histogram, int leavesCount, int firstMatrix, int matricesCount, int histLineSize, float* linearSystem,
                                 TCudaStream stream) {
        if (TArchProps::GetMajorVersion() == 2 && (leavesCount <= 64)) {
            RunMakeMatrices<192>(histogram, leavesCount, histLineSize, firstMatrix, matricesCount, linearSystem, stream);
        } else {
            RunMakeMatrices<256>(histogram, leavesCount, histLineSize, firstMatrix, matricesCount, linearSystem, stream);
        }
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
            MakePointwiseDerivatives<BLOCK_SIZE> << < numBlocks, BLOCK_SIZE,0, stream >> > (pointwiseHist, pointwiseHistSize, partStats, hasPointwiseWeights, rowSize, firstMatrixIdx, matricesCount,  linearSystem);
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
        if (TArchProps::GetMajorVersion() == 2) {
            RunMakePointwiseDerivatives<192> (pointwiseHist, pointwiseHistLineSize, partStats, hasPointwiseWeights, rowSize, firstMatrixIdx, matricesCount, linearSystem, stream);
        } else {
            RunMakePointwiseDerivatives<128> (pointwiseHist, pointwiseHistLineSize, partStats, hasPointwiseWeights, rowSize, firstMatrixIdx, matricesCount, linearSystem, stream);
        }
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
        const int blockSize = 256;
        const int numBlocks = min((pairCount + blockSize - 1) / blockSize,
                                  TArchProps::MaxBlockCount());
        UpdateBinsPairs<<<numBlocks, blockSize, 0, stream>>>(feature, bin, compressedIndex, pairs, pairCount, depth, bins);
    }


    template <int BLOCK_SIZE>
    __global__ void SelectBestSplitImpl(const float* scores,
                                        const TCBinFeature* binFeature, int size,
                                        int bestIndexBias, TBestSplitPropertiesWithIndex* best) {
        float maxScore = -5000000.0f;
        int maxIdx = -1;
        int tid = threadIdx.x;

        #pragma unroll 8
        for (int i = tid; i < size; i += BLOCK_SIZE) {
            float score = scores[i];
            if (score > maxScore) {
                maxScore = score;
                maxIdx = i;
            }
        }

        __shared__ float vals[BLOCK_SIZE];
        __shared__ int inds[BLOCK_SIZE];

        vals[tid] = maxScore;
        inds[tid] = maxIdx;
        __syncthreads();

        for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                if ( vals[tid] <  vals[tid + s] || (vals[tid] == vals[tid + s] && inds[tid] > inds[tid + s]) ) {
                    vals[tid] = vals[tid + s];
                    inds[tid] = inds[tid + s];
                }
            }
            __syncthreads();
        }


        if (tid == 0) {
            TCBinFeature bestFeature;
            if (maxIdx != -1) {
                bestFeature = binFeature[maxIdx];
            } else {
                bestFeature.BinId = 0;
                bestFeature.FeatureId = 0;
            }
            best->Index = bestIndexBias + maxIdx;
            best->Score = vals[0];
            best->BinId = bestFeature.BinId;
            best->FeatureId = bestFeature.FeatureId;
        }
    }

    void SelectBestSplit(const float* scores,
                         const TCBinFeature* binFeature, int size,
                         int bestIndexBias, TBestSplitPropertiesWithIndex* best,
                         TCudaStream stream) {
        if (size > 0) {
            const int blockSize = 1024;
            SelectBestSplitImpl<blockSize><<<1, blockSize, 0, stream>>>(scores,  binFeature, size, bestIndexBias, best);
        }
    }


}
