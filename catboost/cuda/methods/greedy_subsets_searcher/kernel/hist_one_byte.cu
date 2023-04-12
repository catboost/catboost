#include "hist.cuh"
#include "hist_2_one_byte_base.cuh"
#include "tuning_policy_enums.cuh"
#include "compute_hist_loop_one_stat.cuh"

#include <cooperative_groups.h>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    template <int Bits,
             int BlockSize>
    struct TPointHistOneByte {
        const int InnerHistBitsCount = Bits - 5;
        float* Histogram;

        static constexpr int GetHistSize() {
            return BlockSize * 32;
        }

        static constexpr int AddPointsBatchSize() {
            return TLoadSize<LoadSize()>::Size();
        }

        static constexpr int Unroll(ECIndexLoadType) {
            #if __CUDA_ARCH__ < 700
            const int NN = 2;
            #else
            const int NN = 4;
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
//            return ELoadSize::TwoElements;
            #endif
        }

        static constexpr int BlockLoadSize(ECIndexLoadType indexLoadType) {
            return TLoadSize<LoadSize()>::Size() * BlockSize * Unroll(indexLoadType);
        }

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 1024 * (threadIdx.x / 32);
            const int blocks = 8 >> InnerHistBitsCount;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << (InnerHistBitsCount + 2)));
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHistOneByte(float* hist) {
            static_assert(Bits >= 5, "Error: this hist is for 5-8 bits");
            const int histSize = 32 * BlockSize;

            #pragma unroll 8
            for (int i = threadIdx.x; i < histSize; i += BlockSize) {
                hist[i] = 0;
            }

            Histogram = hist + SliceOffset();

            __syncthreads();
        }

        __forceinline__ __device__ void AddPoint(ui32 ci, const float t) {
            auto syncTile = tiled_partition<32>(this_thread_block());

#pragma unroll
            for (int i = 0; i < 4; i++) {
                int f = (threadIdx.x + i) & 3;
                int bin = (ci >> (24 - 8 * f)) & 255;
//                int bin = bfe(ci, 24 - 8 * f, 8);

                const float statToAdd =  (bin >> Bits) == 0 ? t : 0;

                const int mask = (1 << InnerHistBitsCount) - 1;
                const int higherBin = (bin >> 5) & mask;

                int offset = 4 * higherBin + f + ((bin & 31) << 5);

                if (InnerHistBitsCount > 0) {
#pragma unroll
                    for (int k = 0; k < (1 << InnerHistBitsCount); ++k) {
                        const int pass = ((threadIdx.x >> 2) + k) & mask;
                        syncTile.sync();
                        if (pass == higherBin) {
                            Histogram[offset] += statToAdd;
                        }
                    }
                } else {
                    syncTile.sync();
                    Histogram[offset] += statToAdd;
                }
            }
        }


        template <int N>
        __forceinline__ __device__ void AddPointsImpl(const ui32* ci, const float* t) {
            auto syncTile = tiled_partition<32>(this_thread_block());

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = (threadIdx.x + i) & 3;

                int bins[N];
                float stats[N];

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    bins[k] = (ci[k] >> (24 - 8 * f)) & 255;
//                    bins[k] = bfe(ci[k], 24 - 8 * f, 8);
                    stats[k] = (bins[k] >> Bits) == 0 ? t[k] : 0.0f;
                }

                int offsets[N];
                int higherBin[N];

                const int mask = (1 << InnerHistBitsCount) - 1;

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    higherBin[k] = (bins[k] >> 5) & mask;
                    offsets[k] = 4 * higherBin[k] + f + ((bins[k] & 31) << 5);
                }

                if (InnerHistBitsCount > 0) {

                    #pragma unroll
                    for (int k = 0; k < (1 << InnerHistBitsCount); ++k) {
                        const int pass = ((threadIdx.x >> 2) + k) & mask;

                        syncTile.sync();

                        #pragma unroll
                        for (int j = 0; j < N; ++j) {
                            if (pass == higherBin[j]) {
                                Histogram[offsets[j]] += stats[j];
                            }
                        }
                    }

                } else {
                    syncTile.sync();
                    #pragma unroll
                    for (int j = 0; j < N; ++j) {
                        Histogram[offsets[j]] += stats[j];
                    }
                }
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
                const int warpHistSize = 1024;
                for (int start = threadIdx.x; start < warpHistSize; start += BlockSize) {
                    float sum = 0;
                    //12 iterations
                    #pragma unroll 12
                    for (int i = start; i < 32 * BlockSize; i += warpHistSize) {
                        sum += Histogram[i];
                    }
                    Histogram[warpHistSize + start] = sum;
                }
            }
            __syncthreads();

            //now we have only 1024 entries hist
            const int warpHistBlockCount = 8 >> InnerHistBitsCount;
            const int fold = threadIdx.x;
            const int histSize = 1 << (5 + InnerHistBitsCount);

            float sum[4];

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sum[i] = 0.0f;
            }

            if (fold < histSize) {
                const int warpHistSize = 1024;
                const int lowerBitsOffset = (fold & 31) << 5;
                const int higherBin = (fold >> 5) & ((1 << InnerHistBitsCount) - 1);
                const int blockSize = 4 * (1 << InnerHistBitsCount);

                const volatile float* src = Histogram + warpHistSize + lowerBitsOffset + 4 * higherBin;
                #pragma unroll
                for (int block = 0; block < warpHistBlockCount; ++block) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sum[i] += src[i + block * blockSize];
                    }
                }
            }

            __syncthreads();

            if (fold < histSize) {
                for (int i = 0; i < 4; ++i) {
                    Histogram[histSize * i + fold] = sum[i];
                }
            }
            __syncthreads();
        }

        __forceinline__ __device__  void AddToGlobalMemory(int statId, int statCount, int blockCount,
                                                           const TFeatureInBlock* features,
                                                           int fCount,
                                                           int leafId, int leafCount,
                                                           float* binSums) {
            const int fold = threadIdx.x;
            const int histSize = 1 << (5 + InnerHistBitsCount);

            #pragma unroll 4
            for (int fid = 0; fid < fCount; ++fid) {
                TFeatureInBlock group = features[fid];


                const int deviceOffset = group.GroupOffset * statCount * leafCount;
                const int entriesPerLeaf = statCount * group.GroupSize;

                float* dst = binSums + deviceOffset + leafId * entriesPerLeaf + statId * group.GroupSize + group.FoldOffsetInGroup;

                if (fold < features[fid].Folds) {
                    const float val = Histogram[fid * histSize + fold];
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




    void ComputeHistOneByte(int maxBins,
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


            #define PASS(Bits, NumStats)\
            const int blockSize =  384;\
            dim3 numBlocks;\
            numBlocks.z = NumStats;\
            numBlocks.y = partCount;\
            const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;\
            const int maxActiveBlocks = blocksPerSm * TArchProps::SMCount();\
            numBlocks.x = (fCount + 3) / 4;\
            numBlocks.x *= CeilDivide(maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));\
            if (IsGridEmpty(numBlocks)) {\
                return;\
            }\
            using THist = TPointHistOneByte<Bits, blockSize>;\
            ComputeSplitPropertiesDirectLoadsImpl<THist, blockSize, 4><<<numBlocks, blockSize, 0, stream>>>(\
                            features,\
                            fCount,\
                            bins, binsLineSize,\
                            stats, numStats, \
                            statLineSize,\
                            parts,\
                            partIds,\
                            histograms);

            #define HIST2_PASS(Bits)\
            if (numStats % 2 != 0) {\
                PASS(Bits, 1)\
                ComputeHist2OneByteBits<Bits, true>(features, fCount, parts, partIds, partCount, bins, binsLineSize, stats, numStats, statLineSize, histograms, stream);\
            } else {\
                ComputeHist2OneByteBits<Bits, false>(features, fCount, parts, partIds, partCount, bins, binsLineSize, stats, numStats, statLineSize, histograms, stream);\
            }

            if (partCount) {
                if (maxBins <= 32) {
                    HIST2_PASS(5)
                } else if (maxBins <= 64) {
                    HIST2_PASS(6)
//                    PASS(6, numStats)
                } else if (maxBins <= 128) {
                    HIST2_PASS(7)
//                    PASS(7, numStats)
                } else if (maxBins <= 255) {
                    PASS(8, numStats)
                } else {
                    CB_ENSURE(false, "Unsupported bits count " << maxBins);
                }
            }
            #undef PASS
            #undef HIST2_PASS
    }

    void ComputeHistOneByte(int maxBins,
                            const TFeatureInBlock* features,
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

        #define PASS(Bits, NumStats)\
        const int blockSize =  384;\
        dim3 numBlocks;\
        numBlocks.z = NumStats;\
        numBlocks.y = partCount;\
        const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;\
        const int maxActiveBlocks = blocksPerSm * TArchProps::SMCount();\
        const int groupCount = (fCount + 3) / 4;\
        numBlocks.x = groupCount;\
        numBlocks.x *= CeilDivide(2 * maxActiveBlocks, (int)(numBlocks.y * numBlocks.z * numBlocks.x));\
        if (IsGridEmpty(numBlocks)) {\
            return;\
        }\
        using THist = TPointHistOneByte<Bits, blockSize>;\
        ComputeSplitPropertiesGatherImpl<THist, blockSize, 4><<<numBlocks, blockSize, 0, stream>>>(\
                        features,\
                        fCount,\
                        cindex,\
                        indices,\
                        stats, numStats, \
                        statLineSize,\
                        parts,\
                        partIds,\
                        histograms);

        #define HIST2_PASS(Bits)\
            if (numStats % 2 != 0) {\
                PASS(Bits, 1)\
                ComputeHist2OneByteBits<Bits, true>(features, fCount, parts, partIds, partCount, cindex, indices, stats, numStats, statLineSize, histograms, stream);\
            } else {\
                ComputeHist2OneByteBits<Bits, false>(features, fCount, parts, partIds, partCount, cindex, indices, stats, numStats, statLineSize, histograms, stream);\
            }


        if (partCount) {
            if (maxBins <= 32) {
                HIST2_PASS(5)
            } else if (maxBins <= 64) {
                HIST2_PASS(6)
//                PASS(6, numStats)
            } else if (maxBins <= 128) {
                HIST2_PASS(7)
//                PASS(7, numStats)
            } else if (maxBins <= 255) {
                PASS(8, numStats)
            } else {
                CB_ENSURE(false, "Unsupported bins count " << maxBins);
            }
        }
        #undef PASS
        #undef HIST2_PASS
    }



    /*
     * Single part
     */

    void ComputeHistOneByte(int maxBins,
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


        #define PASS(Bits, NumStats)\
            const int blockSize =  384;\
            dim3 numBlocks;\
            numBlocks.z = NumStats;\
            numBlocks.y = 1;\
            const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;\
            const int maxActiveBlocks = blocksPerSm * TArchProps::SMCount();\
            numBlocks.x = (fCount + 3) / 4;\
            numBlocks.x *= CeilDivide(2 * maxActiveBlocks, (int)(numBlocks.x * numBlocks.y * numBlocks.z));\
            if (IsGridEmpty(numBlocks)) {\
                return;\
            }\
            using THist = TPointHistOneByte<Bits, blockSize>;\
            ComputeSplitPropertiesDirectLoadsImpl<THist, blockSize, 4><<<numBlocks, blockSize, 0, stream>>>(\
                            features,\
                            fCount,\
                            bins, binsLineSize,\
                            stats, numStats, \
                            statLineSize,\
                            parts,\
                            partId,\
                            histograms);

        #define HIST2_PASS(Bits)\
            if (numStats % 2 != 0) {\
                PASS(Bits, 1)\
                ComputeHist2OneByteBits<Bits, true>(features, fCount, parts, partId, bins, binsLineSize, stats, numStats, statLineSize, histograms, stream);\
            } else {\
                ComputeHist2OneByteBits<Bits, false>(features, fCount, parts, partId, bins, binsLineSize, stats, numStats, statLineSize, histograms, stream);\
            }

        if (maxBins <= 32) {
            HIST2_PASS(5)
        } else if (maxBins <= 64) {
            HIST2_PASS(6)
//                    PASS(6, numStats)
        } else if (maxBins <= 128) {
            HIST2_PASS(7)
//                    PASS(7, numStats)
        } else if (maxBins <= 255) {
            PASS(8, numStats)
        } else {
            CB_ENSURE(false, "Unsupported bits count " << maxBins);
        }
        #undef PASS
        #undef HIST2_PASS
    }

    void ComputeHistOneByte(int maxBins,
                            const TFeatureInBlock* features,
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

        #define PASS(Bits, NumStats)\
        const int blockSize =  384;\
        dim3 numBlocks;\
        numBlocks.z = NumStats;\
        numBlocks.y = 1;\
        const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;\
        const int maxActiveBlocks = blocksPerSm * TArchProps::SMCount();\
        const int groupCount = (fCount + 3) / 4;\
        numBlocks.x = groupCount;\
        numBlocks.x *= CeilDivide(2 * maxActiveBlocks, (int)(numBlocks.y * numBlocks.z * numBlocks.x));\
        if (IsGridEmpty(numBlocks)) {\
            return;\
        }\
        using THist = TPointHistOneByte<Bits, blockSize>;\
        ComputeSplitPropertiesGatherImpl<THist, blockSize, 4><<<numBlocks, blockSize, 0, stream>>>(\
                        features,\
                        fCount,\
                        cindex,\
                        indices,\
                        stats, numStats, \
                        statLineSize,\
                        parts,\
                        partId,\
                        histograms);

        #define HIST2_PASS(Bits)\
            if (numStats % 2 != 0) {\
                PASS(Bits, 1)\
                ComputeHist2OneByteBits<Bits, true>(features, fCount, parts, partId, cindex, indices, stats, numStats, statLineSize, histograms, stream);\
            } else {\
                ComputeHist2OneByteBits<Bits, false>(features, fCount, parts, partId, cindex, indices, stats, numStats, statLineSize, histograms, stream);\
            }


        if (maxBins <= 32) {
            HIST2_PASS(5)
        } else if (maxBins <= 64) {
            HIST2_PASS(6)
//                PASS(6, numStats)
        } else if (maxBins <= 128) {
            HIST2_PASS(7)
//                PASS(7, numStats)
        } else if (maxBins <= 255) {
            PASS(8, numStats)
        } else {
            CB_ENSURE(false, "Unsupported bins count " << maxBins);
        }
        #undef PASS
        #undef HIST2_PASS
    }

}
