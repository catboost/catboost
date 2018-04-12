#include "pairwise_hist.cuh"
#include "split_properties_helpers.cuh"

#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

namespace NKernel {

    //TODO(noxoomo): fix one hot encoding

    template <bool FULL_PASS>
    __global__ void BuildBinaryFeatureHistograms(const TCFeature* nbFeatures,
                                                 int featureCount,
                                                 const TDataPartition* partition,
                                                 const TPartitionStatistics* partitionStats,
                                                 const ui64 histLineSize,
                                                 float* histogram) {

        if (FULL_PASS) {
            partitionStats += blockIdx.y;
            histogram += blockIdx.y * histLineSize * 4;
        } else {
            const int depth = (int)log2((float)gridDim.y);
            int partId = GetPairwisePartIdToCalculate(partition);
            partitionStats += partId;
            histogram += (((blockIdx.z + 1) << depth) | blockIdx.y) * histLineSize * 4;
        }

        const int featuresPerBlock = blockDim.x / 32;
        const int featureId = blockIdx.x * featuresPerBlock + threadIdx.x / 32;
        nbFeatures += featureId;
        const float partWeight = partitionStats->Weight;

        if (featureId >= featureCount || partitionStats->Weight == 0) {
            return;
        }

        const int x = threadIdx.x & 31;
        const ui32 featureFolds = nbFeatures->Folds;
        const ui32 featureOffset = nbFeatures->FirstFoldIndex;

        for (ui32 fold = x; fold < featureFolds; fold += 32) {
            const ui32 offset = featureOffset + fold;
            const float hist0 = histogram[4 * offset];
            const float hist1 = histogram[4 * offset + 1];
            const float hist2 = histogram[4 * offset + 2];
            const float hist3 = histogram[4 * offset + 3];

            const float w00 = max(hist1 + hist2, 0.0f);
            const float w01 = max(hist0 - hist1, 0.0f);
            const float w10 = max(hist3 - hist2, 0.0f);
            const float w11 = max(partWeight - hist0 - hist3, 0.0f);

            histogram[4 * offset] = w00;
            histogram[4 * offset + 1] = w01;
            histogram[4 * offset + 2] = w10;
            histogram[4 * offset + 3] = w11;
        }
    }


    void BuildBinaryFeatureHistograms(const TCFeature* features, ui32 featureCount,
                                      const TDataPartition* partition,
                                      const TPartitionStatistics* partitionStats,
                                      ui32 partCount,
                                      const ui64 histLineSize,
                                      bool fullPass,
                                      float* histogram,
                                      TCudaStream stream) {

        const int buildHistogramBlockSize = 256;

        dim3 numBlocks;
        numBlocks.x = (featureCount * 32 + buildHistogramBlockSize - 1) / buildHistogramBlockSize;
        numBlocks.y = fullPass ? partCount : partCount / 4;
        numBlocks.z = fullPass ? 1 : 3;

        if (fullPass) {
            BuildBinaryFeatureHistograms<true><< <numBlocks, buildHistogramBlockSize, 0, stream >> > (features, featureCount, partition, partitionStats, histLineSize, histogram);
        } else {
            BuildBinaryFeatureHistograms<false><< <numBlocks, buildHistogramBlockSize, 0, stream >> > (features, featureCount, partition, partitionStats, histLineSize, histogram);
        }
    }

    __global__ void UpdatePairwiseHistogramsImpl(ui32 firstFeatureId, ui32 featureCount,
                                                 const TDataPartition* parts,
                                                 const ui64 histLineSize,
                                                 float* histogram) {
        const int histCount = 4;

        const int depth = (int)log2((float)gridDim.y);
        int partIds[4];
        {
            int partSizes[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int partId = (i << depth) | blockIdx.y;
                partIds[i] = partId;
                partSizes[i] = parts[partId].Size;
            }//

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = i + 1; j < 4; ++j) {
                    if (partSizes[j] > partSizes[i]) {
                        const int tmpSize = partSizes[j];
                        const int tmpId = partIds[j];

                        partSizes[j] = partSizes[i];
                        partIds[j] = partIds[i];

                        partSizes[i] = tmpSize;
                        partIds[i] = tmpId;
                    }
                }
            }
        }

        const ui32 binFeature = firstFeatureId + blockIdx.x * blockDim.x + threadIdx.x;

        if (binFeature < (firstFeatureId + featureCount)) {

            float hists[histCount * 4];
            #pragma unroll
            for (int part = 0; part < 4; ++part) {
                const size_t srcPartIdx = (part << depth) | blockIdx.y;

                #pragma unroll
                for (int i = 0; i < histCount; ++i) {
                    hists[part * 4 + i] = histogram[histCount * (srcPartIdx * histLineSize + binFeature) + i];
                }
            }
            #pragma unroll
            for (int part = 1; part < 4; ++part) {
                #pragma unroll
                for (int i = 0; i < histCount; ++i) {
                    hists[i] -= hists[4 * part + i];
                }
            }

            #pragma unroll
            for (int part = 0; part < 4; ++part) {
                const size_t destPartIdx = partIds[part];
                #pragma unroll
                for (int i = 0; i < histCount; ++i) {
                    histogram[histCount * (destPartIdx * histLineSize + binFeature) + i] = max(hists[part * 4 + i], 0.0f);
                }
            }
        }
    }

    void UpdatePairwiseHistograms(const ui32 firstFeatureId, const ui32 featureCount,
                                  const TDataPartition* dataParts, ui32 partCount,
                                  ui32 histLineSize,
                                  float* histograms,
                                  TCudaStream stream
    ) {
        const int blockSize = 256;
        dim3 numBlocks;
        numBlocks.x = (featureCount + blockSize - 1) / blockSize;
        numBlocks.y = partCount / 4;
        numBlocks.z = 1;
        UpdatePairwiseHistogramsImpl<< <numBlocks, blockSize, 0, stream>>>(firstFeatureId, featureCount, dataParts, histLineSize, histograms);
    }





    void ScanPairwiseHistograms(const TCFeature* features,
                                int featureCount, int partCount,
                                int histLineSize, bool fullPass,
                                float* binSums,
                                TCudaStream stream) {
        const size_t histOffset = fullPass ? 0 : (partCount / 4) * ((ui64) histLineSize * 4);

        const int scanBlockSize = 256;
        dim3 scanBlocks;

        scanBlocks.x = (featureCount * 32 + scanBlockSize - 1) / scanBlockSize;
        scanBlocks.y = fullPass ? 1 : partCount * 3 / 4;
        scanBlocks.z = 1;

        ScanHistogramsImpl<scanBlockSize, 4> << < scanBlocks, scanBlockSize, 0, stream >> > (features, featureCount, histLineSize, binSums + histOffset);
    }


    //shared-memory histograms
    //we store histogram via blocks.
    // every block is 32 bins x 4 features
    // Every binary features has no more, than 8 bits.
    // first 5 bits is index in block
    // next INNER_HIST_BIT is block in warp. For pairwise hists we have no more, than 2 blocks per warp
    // next OUTER_HIST_BITS sets number of warps, that'll be used, to store other part of hist
    // 1 << OUTER_HIST_BITS is number of warps, that will be "at the same time" compute 32 sequential points
    // this logic allows us to reuse l1-cache and make stripped-reads in one pass, instead of (binarization >> 5) passes;
    template<int OUTER_HIST_BITS_COUNT,
             int INNER_HIST_BITS_COUNT,
             int BLOCK_SIZE>
    struct TPairHistOneByte {
        volatile float* Slice;
        uchar BlockId;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 1024 * (threadIdx.x / 32);
            //2 blocks if INNER_HIST_BITS_COUNT = 0, else 1
            const int blocks = 2  >> INNER_HIST_BITS_COUNT;
            //we store 4 histograms per block
            // x4 feature and x4 histograms, though histStart = blockIdx * 16
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << (INNER_HIST_BITS_COUNT + 4)));
            return warpOffset + innerHistStart;
        }


        __forceinline__  __device__ TPairHistOneByte(float* buff) {
            Slice = buff;
            for (int i = threadIdx.x; i < BLOCK_SIZE * 32; i += BLOCK_SIZE) {
                Slice[i] = 0;
            }
            Slice += SliceOffset();
            const int warpId = threadIdx.x / 32;
            BlockId = (uchar) (warpId & ((1 << OUTER_HIST_BITS_COUNT) - 1));
            __syncthreads();
        }

        __forceinline__ __device__ void AddPair(const ui32 ci1,
                                                const ui32 ci2,
                                                const float w) {
            const int binMask = ((1 << (5 + INNER_HIST_BITS_COUNT)) - 1);
            const uchar shift = (threadIdx.x >> 2) & 3;

            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                const uchar f = 4 * ((shift + i) & 3);

                ui32 bin1 = bfe(ci1, 24 - 2 * f, 8);
                ui32 bin2 = bfe(ci2, 24 - 2 * f, 8);

                uchar mults = (((bin1 >> (5 + INNER_HIST_BITS_COUNT)) == BlockId)
                               | (((bin2 >> (5 + INNER_HIST_BITS_COUNT)) == BlockId) << 1));

                mults <<=  bin1 < bin2 ? 0 : 2;

                bin1 &= binMask;
                bin2 &= binMask;

                bin1 *= 1 << (5 - INNER_HIST_BITS_COUNT);
                bin2 *= 1 << (5 - INNER_HIST_BITS_COUNT);

                bin1 += f;
                bin2 += f + 1;

                #pragma  unroll
                for (int currentHist = 0; currentHist < 4; ++currentHist) {
                    const uchar histId = ((threadIdx.x + currentHist) & 3);
                    const int histOffset = histId < 2 ? 0  : 2;
                    const ui32 offset = ((histId & 1) ? bin2 : bin1) + histOffset;
                    const float toAdd = ((mults >> histId) & 1) ? w : 0;

                    //strange, but nvcc can't make this himself
                    if (INNER_HIST_BITS_COUNT != 0) {
                        #pragma unroll
                        for (int k = 0; k < (1 << INNER_HIST_BITS_COUNT); ++k) {
                            if (((threadIdx.x >> 4) & ((1 << INNER_HIST_BITS_COUNT) - 1)) == k) {
                                Slice[offset] += toAdd;
                            }
                        }
                    } else {
                        Slice[offset] += toAdd;
                    }
                }
            }
        }

        __forceinline__ __device__  void Reduce() {
            __syncthreads();
            Slice -= SliceOffset();

            const int outerHistCount =  1 << (OUTER_HIST_BITS_COUNT);
            const int totalLinesPerWarp = 32;

            const int warpIdx = (threadIdx.x / 32);
            const int warpCount = BLOCK_SIZE / 32; // 12
            const int x = (threadIdx.x & 31); // binIdx
            {
                #pragma unroll
                for (int outerBits = 0; outerBits < outerHistCount; ++outerBits) {
                    for (int line = warpIdx; line < totalLinesPerWarp; line += warpCount) {
                        int offset = 32 * line + x;
                        float sum = 0.0f;
                        #pragma unroll
                        for (int i = outerBits; i < warpCount; i += outerHistCount) {
                            sum += Slice[offset + i * 1024];
                        }

                        Slice[offset + outerBits * 1024] = sum;
                    }
                }
                __syncthreads();


                #pragma unroll
                for (int outerBits = 0; outerBits < outerHistCount; ++outerBits) {
                    if ((INNER_HIST_BITS_COUNT == 0))
                    {
                        #pragma unroll
                        for (int line = warpIdx; line < totalLinesPerWarp; line += warpCount)
                        {
                            //we use, that there can be only 1 or 2 hists in warp
                            if ((x < 16)) {
                                int offset = 32 * line + x + outerBits * 1024;
                                Slice[offset] += Slice[offset + 16];
                            }
                        }
                    }
                }
            }

            __syncthreads();
        }
    };



    template<int BLOCK_SIZE>
    struct TPairHistOneByte <0, 0, BLOCK_SIZE> {
        volatile float* Slice;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 1024 * (threadIdx.x / 32);
            //2 blocks if INNER_HIST_BITS_COUNT = 0, else 1
            const int blocks = 2;
            //we store 4 histograms per block
            // x4 feature and x4 histograms, though histStart = blockIdx * 16
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << 4));
            return warpOffset + innerHistStart;
        }


        __forceinline__  __device__ TPairHistOneByte(float* buff) {
            Slice = buff;
            for (int i = threadIdx.x; i < BLOCK_SIZE * 32; i += BLOCK_SIZE) {
                Slice[i] = 0;
            }
            Slice += SliceOffset();
            __syncthreads();
        }

        __forceinline__ __device__ void AddPair(const ui32 ci1,
                                                const ui32 ci2,
                                                const float w) {
            const int binMask = 31;
            const uchar shift = (threadIdx.x >> 2) & 3;

            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                const uchar f = 4 * ((shift + i) & 3);

                ui32 bin1 = bfe(ci1, 24 - 2 * f, 8);
                ui32 bin2 = bfe(ci2, 24 - 2 * f, 8);

                uchar mults = ((bin1 < 32) | ((bin2 < 32) << 1));
                mults <<=  bin1 < bin2 ? 0 : 2;

                bin1 &= binMask;
                bin2 &= binMask;

                bin1 *= 32;
                bin2 *= 32;

                bin1 += f;
                bin2 += f + 1;

                #pragma  unroll
                for (int currentHist = 0; currentHist < 4; ++currentHist) {
                    const uchar histId = ((threadIdx.x + currentHist) & 3);
                    const int histOffset = histId < 2 ? 0  : 2;
                    const ui32 offset = ((histId & 1) ? bin2 : bin1) + histOffset;
                    const float toAdd = ((mults >> histId) & 1) ? w : 0;
                    Slice[offset] += toAdd;
                }
            }
        }

        __forceinline__ __device__  void Reduce() {
            __syncthreads();
            Slice -= SliceOffset();

            const int outerHistCount =  1;
            const int totalLinesPerWarp = 32;

            const int warpIdx = (threadIdx.x / 32);
            const int warpCount = BLOCK_SIZE / 32; // 12
            const int x = (threadIdx.x & 31); // binIdx
            {
                #pragma unroll
                for (int outerBits = 0; outerBits < outerHistCount; ++outerBits) {
                    for (int line = warpIdx; line < totalLinesPerWarp; line += warpCount) {
                        int offset = 32 * line + x;
                        float sum = 0.0f;
                        #pragma unroll
                        for (int i = outerBits; i < warpCount; i += outerHistCount) {
                            sum += Slice[offset + i * 1024];
                        }

                        Slice[offset + outerBits * 1024] = sum;
                    }
                }
                __syncthreads();


                #pragma unroll
                for (int outerBits = 0; outerBits < outerHistCount; ++outerBits) {
                    #pragma unroll
                    for (int line = warpIdx; line < totalLinesPerWarp; line += warpCount)
                    {
                        //we use, that there can be only 1 or 2 hists in warp
                        if ((x < 16)) {
                            int offset = 32 * line + x + outerBits * 1024;
                            Slice[offset] += Slice[offset + 16];
                        }
                    }
                }
            }

            __syncthreads();
        }
    };


    //768
    template<int BLOCK_SIZE>
    struct TPairHistHalfByte {
        volatile float* Slice;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 512 * (threadIdx.x / 32);
            //we store 4 histograms per block
            // x8 feature and x4 histograms, though histStart = blockIdx * 16
            return warpOffset;
        }

        __forceinline__ __device__ int HistSize() {
            return 16 * BLOCK_SIZE;
        }

        __forceinline__  __device__ TPairHistHalfByte(float* buff) {
            Slice = buff;
            for (int i = threadIdx.x; i < BLOCK_SIZE * 16; i += BLOCK_SIZE) {
                Slice[i] = 0;
            }
            Slice += SliceOffset();
            __syncthreads();
        }

        __forceinline__ __device__ void AddPair(const ui32 ci1,
                                                const ui32 ci2,
                                                const float w)
        {
            const uchar shift = (threadIdx.x >> 2) & 7;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const uchar f = 4 * ((shift + i) & 7);

                ui32 bin1 = bfe(ci1, 28 - f, 4);
                ui32 bin2 = bfe(ci2, 28 - f, 4);
                const bool isLeq = bin1 < bin2;

                bin1 <<= 5;
                bin2 <<= 5;

                bin1 += f;
                bin2 += f + 1;


                #pragma  unroll
                for (int currentHist = 0; currentHist < 4; ++currentHist) {

                    const uchar histId = ((threadIdx.x + currentHist) & 3);
                    const bool addToLeqHist = histId < 2;
                    const ui32 offset = ((histId & 1) ? bin2 : bin1) + (addToLeqHist ? 0 : 2);
                    const float toAdd = (isLeq == addToLeqHist) ? w : 0;

                    //offset = 32 * bin + 4 * feature + histId
                    //feature from 0 to 7, histId from 0 to 3
                    //hist0 and hist2 use bin1
                    //host 1 and hist 3 use bin2
                    Slice[offset] += toAdd;
                }
            }
        }

        __forceinline__ __device__  void Reduce() {

            __syncthreads();
            Slice -= SliceOffset();

            float sum = 0.f;

            if (threadIdx.x < 512) {
                const int warpCount = BLOCK_SIZE / 32;
                int binId = threadIdx.x / 32;
                const int x = threadIdx.x & 31;
                Slice += 32 * binId + x;

                {
                    #pragma unroll
                    for (int warpId = 0; warpId < warpCount; ++warpId) {
                        sum += Slice[warpId * 512];
                    }
                }
            }
            __syncthreads();
            //bin0: f0: hist0 hist1 hist2 hist3 f1: hist0 hist1 hist2 hist3 …
            if (threadIdx.x < 512) {
                Slice[0] = sum;
            }
            __syncthreads();
        }
    };

    template <int BLOCK_SIZE>
    struct TPairBinaryHist {
        float* Slice;

        __forceinline__ __device__ int HistSize() {
            return BLOCK_SIZE * 16;
        }

        __forceinline__ __device__ int SliceOffset() {
            return 512 * (threadIdx.x >> 5);

        }

        __forceinline__ __device__ TPairBinaryHist(float* buff) {
            Slice = buff;
            for (int i = threadIdx.x; i < HistSize(); i += BLOCK_SIZE) {
                Slice[i] = 0;
            }
            Slice += SliceOffset();
            __syncthreads();
        }

        __forceinline__ __device__ void AddPair(const ui32 ci1, const ui32 ci2, const float w) {

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int f = (((threadIdx.x >> 2) + i) & 7) << 2;

                const int bin1 = bfe(ci1, 28 - f, 4);
                const int bin2 = bfe(ci2, 28 - f, 4);

                const int invBin1 = (~bin1) & 15;
                const int invBin2 = (~bin2) & 15;

                //00 01 10 11
                const ui32 bins = (invBin1 & invBin2) | ((invBin1 & bin2) << 8) | ((bin1 & invBin2) << 16) | ((bin1 & bin2) << 24);

                #pragma  unroll
                for (int currentHist = 0; currentHist < 4; ++currentHist) {
                    const uchar histOffset = (threadIdx.x + currentHist) & 3;
                    const short bin = (bins >> (histOffset << 3)) & 15;
                    // 32 * bin + 4 * featureId + histId
                    //512 floats per warp
                    Slice[f + (bin << 5) + histOffset] +=  w;
                }
            }
        }

        __forceinline__ __device__  void Reduce() {

            __syncthreads();
            Slice -= SliceOffset();

            float sum = 0.f;

            if (threadIdx.x < 512) {
                const int warpCount = BLOCK_SIZE / 32;
                int binId = threadIdx.x / 32;
                const int x = threadIdx.x & 31;
                Slice += 32 * binId + x;

                {
                    #pragma unroll
                    for (int warpId = 0; warpId < warpCount; ++warpId) {
                        sum += Slice[warpId * 512];
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x < 512) {
                Slice[0] = sum;
            }
            __syncthreads();
        }
    };


    template <int STRIPE_SIZE, int HIST_BLOCK_COUNT, int N, int OUTER_UNROLL, int BLOCKS_PER_FEATURE, typename THist>
    __forceinline__  __device__ void ComputePairHistogram(ui32 offset,
                                                          const ui32* __restrict cindex,
                                                          int dsSize,
                                                          const uint2* __restrict pairs,
                                                          const float* __restrict weight,
                                                          float* __restrict histogram)
    {

        THist hist(histogram);

        offset += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;
        dsSize = max(dsSize - (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE, 0);
        const int stripe = STRIPE_SIZE * BLOCKS_PER_FEATURE;

        int i = (threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32;
        const int iteration_count = (dsSize - i + stripe - 1)  / stripe;
        const int blocked_iteration_count = ((dsSize - (i | 31) + stripe - 1) / stripe) / N;

        if (dsSize) {
            i += offset;
            pairs += i;
            weight += i;

            #pragma unroll OUTER_UNROLL
            for (int j = 0; j < blocked_iteration_count; ++j) {
                ui32 local_ci[N * 2];
                float local_w[N];

                #pragma unroll
                for (int k = 0; k < N; k++) {
                    uint2 p = __ldg(pairs + stripe * k);
                    local_ci[k] = __ldg(cindex + p.x);
                    local_ci[k + N] = __ldg(cindex + p.y);
                    local_w[k] = __ldg(weight + stripe * k);
                }

                #pragma unroll
                for (int k = 0; k < N; k++) {
                    hist.AddPair(local_ci[k], local_ci[k + N], local_w[k]);
                }

                i += stripe * N;
                pairs += stripe * N;
                weight += stripe * N;
            }

            for (int k = blocked_iteration_count * N; k < iteration_count; ++k)
            {
                const uint2 p = __ldg(pairs);
                const ui32 ci1 = __ldg(cindex + p.x);
                const ui32 ci2 = __ldg(cindex + p.y);
                const float w = __ldg(weight);
                hist.AddPair(ci1, ci2, w);

                i += stripe;
                pairs += stripe;
                weight += stripe;
            }

            hist.Reduce();
        }
        __syncthreads();
    }

    //TODO: trait class for unroll constanst for different architectures
    template<int BLOCK_SIZE,
             int OUTER_BITS, int INNER_BITS,
             int N, int OUTER_UNROLL,
             int BLOCKS_PER_FEATURE>
    __forceinline__ __device__ void ComputeSplitPropertiesOneBytePass(const TCFeature* feature, int fCount,
                                                                       const ui32* __restrict cindex,
                                                                       const uint2* __restrict pairs,
                                                                       const float* __restrict  weight,
                                                                       const TDataPartition* partition,
                                                                       float* __restrict histogram,
                                                                       float* __restrict smem) {

        using THist = TPairHistOneByte<OUTER_BITS, INNER_BITS, BLOCK_SIZE>;
        constexpr int stripeSize = BLOCK_SIZE >>  OUTER_BITS;
        constexpr int histBlockCount = 1 << OUTER_BITS;
        ComputePairHistogram< stripeSize, histBlockCount, N, OUTER_UNROLL, BLOCKS_PER_FEATURE, THist>(partition->Offset, cindex, partition->Size, pairs, weight, smem);

        if (threadIdx.x < 256) {
            const int histId = threadIdx.x & 3;
            const int binId = (threadIdx.x >> 2) & 15;
            const int fid = (threadIdx.x >> 6) & 3;
            const int binMask = ((1 << (5 + INNER_BITS)) - 1);

            if (fid < fCount) {
                const ui32 bfStart = feature[fid].FirstFoldIndex;
                histogram += 4 * bfStart;

                for (int fold = binId; fold < feature[fid].Folds; fold += 16) {
                    const int outerBits = fold >> (5 + INNER_BITS);
                    const int readBinIdx = (fold & binMask) << (5 - INNER_BITS);
                    const int readOffset = 1024 * outerBits + readBinIdx + (4 * fid +  histId);
                    if (BLOCKS_PER_FEATURE > 1) {
                        atomicAdd(histogram + 4 * fold + histId, smem[readOffset]);
                    } else {
                        histogram[4 * fold + histId] += smem[readOffset];
                    }
                }
            }
        }
    }


    template<int BLOCK_SIZE, int N, int OUTER_UNROLL, int BLOCKS_PER_FEATURE>
    __forceinline__ __device__ void ComputeSplitPropertiesHalfBytePass(const TCFeature* feature, int fCount,
                                                                        const uint* __restrict cindex,
                                                                        const uint2* __restrict pairs, const float* __restrict  weight,
                                                                        const TDataPartition* partition,
                                                                        float* __restrict histogram,
                                                                        float* __restrict smem) {
        using THist = TPairHistHalfByte<BLOCK_SIZE>;
        ComputePairHistogram<BLOCK_SIZE, 1, N, OUTER_UNROLL, BLOCKS_PER_FEATURE,  THist >(partition->Offset, cindex, partition->Size, pairs, weight, smem);


        if (threadIdx.x < 512) {
            const int histId = threadIdx.x & 3;
            const int fold = (threadIdx.x >> 2) & 15;
            const int fid = (threadIdx.x >> 6) & 7;

            if (fid < fCount) {
                const ui32 bfStart = feature[fid].FirstFoldIndex;
                histogram += 4 * bfStart;

                if (fold < feature[fid].Folds) {
                    const int readOffset = 32 * fold + 4 * fid + histId;

                    if (BLOCKS_PER_FEATURE > 1) {
                        atomicAdd(histogram + 4 * fold + histId, smem[readOffset]);
                    } else {
                        histogram[4 * fold + histId] += smem[readOffset];
                    }
                }
            }
        }
    }


    template<int BLOCK_SIZE, int INNER_UNROLL, int OUTER_UNROLL, int BLOCKS_PER_FEATURE>
    __forceinline__ __device__ void ComputeSplitPropertiesBinaryPass(const TCFeature* feature, int fCount,
                                                                      const ui32* __restrict cindex,
                                                                      const uint2* __restrict pairs,
                                                                      const float* __restrict  weight,
                                                                      const TDataPartition* partition,
                                                                      float* __restrict histogram,
                                                                      float* __restrict smem) {
        using THist = TPairBinaryHist<BLOCK_SIZE>;
        ComputePairHistogram<BLOCK_SIZE, 1, INNER_UNROLL, OUTER_UNROLL, BLOCKS_PER_FEATURE, THist >(partition->Offset, cindex, partition->Size, pairs, weight, smem);

        const int histId = threadIdx.x & 3;
        const int fid = (threadIdx.x >> 2);

        __syncthreads();

        if (fid < fCount) {
            float sum = 0;
            const int groupId = fid / 4;
            const int fixedBitId = 3 - fid % 4;
            const int activeMask = (1 << fixedBitId);

            //fix i'th bit and iterate through others
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (i & activeMask) {
                    sum += smem[32 * i + 4 * groupId + histId];
                }
            }

            if (BLOCKS_PER_FEATURE > 1) {
                atomicAdd(histogram + feature[fid].FirstFoldIndex * 4 + histId, sum);
            } else {
                histogram[feature[fid].FirstFoldIndex * 4 + histId] += sum;
            }
        }
        __syncthreads();
    }


    #define DECLARE_PASS_ONE_BYTE(O, I, N, OUTER_UNROLL, M) \
        ComputeSplitPropertiesOneBytePass<BLOCK_SIZE, O, I, N, OUTER_UNROLL, M>(feature, fCount, cindex, pairs, weight, partition, histogram, &localHist[0]);


    #define DECLARE_PASS_HALF_BYTE(N, OUTER_UNROLL, M) \
        ComputeSplitPropertiesHalfBytePass<BLOCK_SIZE, N, OUTER_UNROLL, M>(feature, fCount, cindex, pairs, weight, partition, histogram, &localHist[0]);


    #define DECLARE_PASS_BINARY(N, OUTER_UNROLL, M) \
        ComputeSplitPropertiesBinaryPass<BLOCK_SIZE, N, OUTER_UNROLL, M>(feature, fCount, cindex, pairs, weight, partition, histogram, &localHist[0]);


    template<int BLOCK_SIZE, bool FULL_PASS, int M>
    #if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
    #else
    __launch_bounds__(BLOCK_SIZE, 1)
    #endif
    __global__ void ComputeSplitPropertiesNonBinaryPairs(const TCFeature* feature, int fCount, const ui32* cindex,
                                                         const uint2* pairs, const float* weight,
                                                         const TDataPartition* partition,
                                                         int histLineSize,
                                                         float* histogram) {

        const int featureOffset = (blockIdx.x / M) * 4;
        feature += featureOffset;
        cindex += feature->Offset;
        fCount = min(fCount - featureOffset, 4);

        if (FULL_PASS) {
            partition += blockIdx.y;
            histogram += blockIdx.y * histLineSize * 4;
        } else {
            const int depth = (int)log2((float)gridDim.y);
            int partId = GetPairwisePartIdToCalculate(partition);
            partition += partId;
            histogram += (((blockIdx.z + 1) << depth) | blockIdx.y) * histLineSize * 4ULL;
        }

        if (partition->Size == 0) {
            return;
        }

        __shared__ float localHist[32 * BLOCK_SIZE];

        const int maxBinCount = GetMaxBinCount(feature, fCount, (int*) &localHist[0]);
        __syncthreads();


        #if __CUDA__ARCH <= 350
        const int INNER_UNROLL = 4;
        const int OUTER_UNROLL = 2;
        #else
        //TODO(noxoomo): tune it on maxwell+
        const int INNER_UNROLL = 1;
        const int OUTER_UNROLL = 2;
        #endif

        if (maxBinCount <= 32) {
            DECLARE_PASS_ONE_BYTE(0, 0, INNER_UNROLL, OUTER_UNROLL, M)
        } else if (maxBinCount <= 64) {
            DECLARE_PASS_ONE_BYTE(1, 0, INNER_UNROLL, OUTER_UNROLL, M)
        } else if (maxBinCount <= 128) {
            DECLARE_PASS_ONE_BYTE(2, 0, INNER_UNROLL, OUTER_UNROLL, M)
        } else {
            DECLARE_PASS_ONE_BYTE(2, 1, INNER_UNROLL, OUTER_UNROLL, M)
        }
    }


    template<int BLOCK_SIZE, bool FULL_PASS, int M>
    #if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
    #else
    __launch_bounds__(BLOCK_SIZE, 1)
    #endif
    __global__ void ComputeSplitPropertiesHalfBytePairs(const TCFeature* feature, int fCount, const ui32* cindex,
                                                        const uint2* pairs, const float* weight,
                                                        const TDataPartition* partition,
                                                        int histLineSize,
                                                        float* histogram) {
        //histogram line size - size of one part hist.
        const int featureOffset = (blockIdx.x / M) * 8;
        feature += featureOffset;
        cindex += feature->Offset;
        fCount = min(fCount - featureOffset, 8);

        if (FULL_PASS) {
            partition += blockIdx.y;
            histogram += blockIdx.y * histLineSize * 4;
        } else {
            const int depth = (int)log2((float)gridDim.y);
            int partId = GetPairwisePartIdToCalculate(partition);
            partition += partId;
            histogram += (((blockIdx.z + 1) << depth) | blockIdx.y) * histLineSize * 4ULL;
        }

        if (partition->Size == 0) {
            return;
        }

        __shared__ float localHist[16 * BLOCK_SIZE];
        __syncthreads();

        DECLARE_PASS_HALF_BYTE(1, 2, M)
    }


    template<int BLOCK_SIZE, bool FULL_PASS, int M>
    #if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
    #else
    __launch_bounds__(BLOCK_SIZE, 1)
    #endif
    __global__ void ComputeSplitPropertiesBinaryPairs(const TCFeature* feature, int fCount, const ui32* cindex,
                                                      const uint2* pairs, const float* weight,
                                                      const TDataPartition* partition,
                                                      int histLineSize,
                                                      float* histogram) {
        {
            const int featureOffset =  (blockIdx.x / M) * 32;
            feature += featureOffset;
            cindex += feature->Offset;
            fCount = min(fCount - featureOffset, 32);
        }
        if (FULL_PASS) {
            partition += blockIdx.y;
            histogram += blockIdx.y * ((ui64)histLineSize * 4ULL);
        } else {
            const int depth = (int)log2((float)gridDim.y);
            int partId = GetPairwisePartIdToCalculate(partition);
            partition += partId;
            histogram += (((blockIdx.z + 1) << depth) | blockIdx.y) * ((ui64)histLineSize) * 4ULL;
        }

        __shared__ float localHist[16 * BLOCK_SIZE];

        if (partition->Size == 0) {
            return;
        }

        DECLARE_PASS_BINARY(1, 2, M);
    }


    void ComputePairwiseHistogramOneByte(const TCFeature* features,
                                         const ui32 featureCount,
                                         const ui32* compressedIndex,
                                         const uint2* pairs, ui32 pairCount,
                                         const float* weight,
                                         const TDataPartition* partition,
                                         ui32 partCount,
                                         ui32 histLineSize,
                                         bool fullPass,
                                         float* histogram,
                                         TCudaStream stream) {

        if (featureCount > 0) {
            const int blockSize = 384;
            dim3 numBlocks;
            numBlocks.x = (featureCount + 3) / 4;
            numBlocks.y = fullPass ? 1 : partCount / 4;
            numBlocks.z = fullPass ? 1 : 3;

            const ui32 blockPerFeatureMultiplier = EstimateBlockPerFeatureMultiplier(numBlocks, pairCount, 32);
            numBlocks.x *= blockPerFeatureMultiplier;


            #define NB_HIST(IS_FULL, BLOCKS_PER_FEATURE)   \
            ComputeSplitPropertiesNonBinaryPairs < blockSize, IS_FULL, BLOCKS_PER_FEATURE > << <numBlocks, blockSize, 0, stream>>>(\
                                                  features, featureCount, compressedIndex,  pairs,\
                                                  weight, partition,  histLineSize, histogram);

            #define DISPATCH(BLOCKS_PER_FEATURE)  \
            if (fullPass) {                       \
                NB_HIST(true, BLOCKS_PER_FEATURE) \
            } else {                              \
                NB_HIST(false, BLOCKS_PER_FEATURE)\
            }


            if (blockPerFeatureMultiplier == 1) {
                DISPATCH(1);
            } else if (blockPerFeatureMultiplier == 2) {
                DISPATCH(2);
            } else if (blockPerFeatureMultiplier == 4) {
                DISPATCH(4);
            } else if (blockPerFeatureMultiplier == 8) {
                DISPATCH(8);
            } else if (blockPerFeatureMultiplier == 16) {
                DISPATCH(16);
            } else if (blockPerFeatureMultiplier == 32) {
                DISPATCH(32);
            } else {
                exit(0);
            }
            #undef NB_HIST
            #undef DISPATCH
        }
    }


    void ComputePairwiseHistogramHalfByte(const TCFeature* features,
                                          const ui32 featureCount,
                                          const ui32* compressedIndex,
                                          const uint2* pairs, ui32 pairCount,
                                          const float* weight,
                                          const TDataPartition* partition,
                                          ui32 partCount,
                                          ui32 histLineSize,
                                          bool fullPass,
                                          float* histogram,
                                          TCudaStream stream) {

        if (featureCount > 0) {
            const int blockSize = 768;
            dim3 numBlocks;
            numBlocks.x = (featureCount + 7) / 8;
            numBlocks.y = fullPass ? 1 : partCount / 4;
            numBlocks.z = fullPass ? 1 : 3;

            const ui32 blockPerFeatureMultiplier = EstimateBlockPerFeatureMultiplier(numBlocks, pairCount, 32);
            numBlocks.x *= blockPerFeatureMultiplier;


            #define NB_HIST(IS_FULL, BLOCKS_PER_FEATURE)   \
            ComputeSplitPropertiesHalfBytePairs < blockSize, IS_FULL, BLOCKS_PER_FEATURE > << <numBlocks, blockSize, 0, stream>>>(\
                                                  features, featureCount, compressedIndex,  pairs,\
                                                  weight, partition, histLineSize, histogram);

            #define DISPATCH(BLOCKS_PER_FEATURE)  \
            if (fullPass) {                       \
                NB_HIST(true, BLOCKS_PER_FEATURE) \
            } else {                              \
                NB_HIST(false, BLOCKS_PER_FEATURE)\
            }


            if (blockPerFeatureMultiplier == 1) {
                DISPATCH(1);
            } else if (blockPerFeatureMultiplier == 2) {
                DISPATCH(2);
            } else if (blockPerFeatureMultiplier == 4) {
                DISPATCH(4);
            } else if (blockPerFeatureMultiplier == 8) {
                DISPATCH(8);
            } else if (blockPerFeatureMultiplier == 16) {
                DISPATCH(16);
            } else if (blockPerFeatureMultiplier == 32) {
                DISPATCH(32);
            } else {
                exit(0);
            }
            #undef NB_HIST
            #undef DISPATCH
        }
    }


    void ComputePairwiseHistogramBinary(const TCFeature* features,
                                          const ui32 featureCount,
                                          const ui32* compressedIndex,
                                          const uint2* pairs, ui32 pairCount,
                                          const float* weight,
                                          const TDataPartition* partition,
                                          ui32 partCount,
                                          ui32 histLineSize,
                                          bool fullPass,
                                          float* histogram,
                                          TCudaStream stream) {

        if (featureCount > 0) {
            const int blockSize = 768;
            dim3 numBlocks;
            numBlocks.x = (featureCount + 31) / 32;
            numBlocks.y = fullPass ? 1 : partCount / 4;
            numBlocks.z = fullPass ? 1 : 3;

            const ui32 blockPerFeatureMultiplier = EstimateBlockPerFeatureMultiplier(numBlocks, pairCount, 32);
            numBlocks.x *= blockPerFeatureMultiplier;


            #define NB_HIST(IS_FULL, BLOCKS_PER_FEATURE)   \
            ComputeSplitPropertiesBinaryPairs < blockSize, IS_FULL, BLOCKS_PER_FEATURE > << <numBlocks, blockSize, 0, stream>>>(\
                                                  features, featureCount, compressedIndex,  pairs,\
                                                  weight, partition, histLineSize,  histogram);

            #define DISPATCH(BLOCKS_PER_FEATURE)  \
            if (fullPass) {                       \
                NB_HIST(true, BLOCKS_PER_FEATURE) \
            } else {                              \
                NB_HIST(false, BLOCKS_PER_FEATURE)\
            }


            if (blockPerFeatureMultiplier == 1) {
                DISPATCH(1);
            } else if (blockPerFeatureMultiplier == 2) {
                DISPATCH(2);
            } else if (blockPerFeatureMultiplier == 4) {
                DISPATCH(4);
            } else if (blockPerFeatureMultiplier == 8) {
                DISPATCH(8);
            } else if (blockPerFeatureMultiplier == 16) {
                DISPATCH(16);
            } else if (blockPerFeatureMultiplier == 32) {
                DISPATCH(32);
            } else {
                exit(0);
            }
            #undef NB_HIST
            #undef DISPATCH
        }
    }





}
