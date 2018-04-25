#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/gpu_data/grid_policy.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/methods/kernel/split_pairwise.cuh>
#include <catboost/cuda/methods/kernel/pairwise_hist.cuh>
#include <catboost/cuda/methods/kernel/linear_solver.cuh>
#include <catboost/libs/options/enums.h>
#include <catboost/cuda/utils/compression_helpers.h>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>

namespace NKernelHost {
    inline ui32 GetRowSizeFromLinearSystemSize(ui32 systemSize) {
        return static_cast<ui32>((-3 + sqrt(9 + 8 * systemSize)) / 2);
    }

    class TMakeLinearSystemKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> PointwiseHistogram;
        TCudaBufferPtr<const TPartitionStatistics> PartStats;
        TCudaBufferPtr<const float> PairwiseHistogram;
        ui64 HistogramLineSize; /* number of binary features */
        TSlice BlockFeaturesSlice;
        TCudaBufferPtr<float> LinearSystem;

    public:
        TMakeLinearSystemKernel() = default;

        TMakeLinearSystemKernel(TCudaBufferPtr<const float> pointwiseHistogram,
                                TCudaBufferPtr<const TPartitionStatistics> partStats,
                                TCudaBufferPtr<const float> pairwiseHistogram,
                                ui64 histogramLineSize,
                                TSlice blockFeaturesSlice,
                                TCudaBufferPtr<float> linearSystem)
            : PointwiseHistogram(pointwiseHistogram)
            , PartStats(partStats)
            , PairwiseHistogram(pairwiseHistogram)
            , HistogramLineSize(histogramLineSize)
            , BlockFeaturesSlice(blockFeaturesSlice)
            , LinearSystem(linearSystem)
        {
        }

        Y_SAVELOAD_DEFINE(PointwiseHistogram, PartStats, PairwiseHistogram, HistogramLineSize, BlockFeaturesSlice, LinearSystem);

        void Run(const TCudaStream& stream) const {
            const ui32 systemSize = LinearSystem.ObjectSize();
            const ui32 rowSize = GetRowSizeFromLinearSystemSize(systemSize);
            CB_ENSURE(rowSize > 1, systemSize);
            const ui32 leavesCount = rowSize / 2;

            CB_ENSURE(systemSize == (rowSize + rowSize * (rowSize + 1) / 2));
            CB_ENSURE(BlockFeaturesSlice.Size() <= HistogramLineSize);
            CB_ENSURE(BlockFeaturesSlice.Size() == LinearSystem.ObjectCount());
            const bool useWeights = PointwiseHistogram.ObjectSize() == 2;

            NKernel::MakePointwiseDerivatives(PointwiseHistogram.Get(), HistogramLineSize,
                                              PartStats.Get(),
                                              useWeights,
                                              rowSize,
                                              BlockFeaturesSlice.Left, BlockFeaturesSlice.Size(),
                                              LinearSystem.Get(), stream.GetStream());

            NKernel::MakePairwiseDerivatives(PairwiseHistogram.Get(),
                                             leavesCount,
                                             BlockFeaturesSlice.Left, BlockFeaturesSlice.Size(),
                                             HistogramLineSize,
                                             LinearSystem.Get(),
                                             stream.GetStream());
        }
    };

    class TUpdateBinsPairsKernel: public TStatelessKernel {
    private:
        TCFeature Feature;
        ui32 Bin;
        TCudaBufferPtr<const ui32> CompressedIndex;
        TCudaBufferPtr<const uint2> Pairs;
        ui32 Depth;
        TCudaBufferPtr<ui32> Bins;

    public:
        TUpdateBinsPairsKernel() = default;

        TUpdateBinsPairsKernel(TCFeature feature, ui32 bin,
                               TCudaBufferPtr<const ui32> compressedIndex,
                               TCudaBufferPtr<const uint2> pairs,
                               ui32 depth,
                               TCudaBufferPtr<ui32> bins)
            : Feature(feature)
            , Bin(bin)
            , CompressedIndex(compressedIndex)
            , Pairs(pairs)
            , Depth(depth)
            , Bins(bins)
        {
        }

        Y_SAVELOAD_DEFINE(Feature, Bin, CompressedIndex, Pairs, Depth, Bins);

        void Run(const TCudaStream& stream) const {
            NKernel::UpdateBinsPairs(Feature, Bin, CompressedIndex.Get(), Pairs.Get(), Pairs.Size(), Depth, Bins.Get(), stream.GetStream());
        }
    };

    class TExtractMatricesAndTargetsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> LinearSystem;
        TCudaBufferPtr<float> LowTriangleMatrices;
        TCudaBufferPtr<float> Solutions;
        TSlice SolutionsSlice;

    public:
        TExtractMatricesAndTargetsKernel() = default;

        TExtractMatricesAndTargetsKernel(TCudaBufferPtr<const float> linearSystem,
                                         TCudaBufferPtr<float> matrices,
                                         TCudaBufferPtr<float> solutions,
                                         TSlice solutionsSlice)
            : LinearSystem(linearSystem)
            , LowTriangleMatrices(matrices)
            , Solutions(solutions)
            , SolutionsSlice(solutionsSlice)
        {
        }

        Y_SAVELOAD_DEFINE(LinearSystem,
                          LowTriangleMatrices,
                          Solutions,
                          SolutionsSlice);

        void Run(const TCudaStream& stream) const {
            const ui32 rowSize = GetRowSizeFromLinearSystemSize(LinearSystem.ObjectSize());
            CB_ENSURE(Solutions.ObjectSize() == rowSize);
            const ui32 matricesCount = LinearSystem.ObjectCount();
            CB_ENSURE(matricesCount == SolutionsSlice.Size());
            float* firstSolution = Solutions.GetForObject(SolutionsSlice.Left);
            NKernel::ExtractMatricesAndTargets(LinearSystem.Get(),
                                               matricesCount,
                                               rowSize,
                                               LowTriangleMatrices.Get(),
                                               firstSolution,
                                               stream.GetStream());
        }
    };

    class TZeroMeanKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<float> Solutions;
        TSlice SolutionsSlice;

    public:
        TZeroMeanKernel() = default;

        TZeroMeanKernel(TCudaBufferPtr<float> solutions,
                        TSlice slice)
            : Solutions(solutions)
            , SolutionsSlice(slice)
        {
        }

        Y_SAVELOAD_DEFINE(Solutions, SolutionsSlice);

        void Run(const TCudaStream& stream) const {
            const ui32 rowSize = Solutions.ObjectSize();
            NKernel::ZeroMean(Solutions.GetForObject(SolutionsSlice.Left), rowSize, SolutionsSlice.Size(), stream.GetStream());
        }
    };

    class TRegularizeKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<float> Matrices;
        double LambdaNonDiag;
        double LambdaDiag;

    public:
        TRegularizeKernel() = default;

        TRegularizeKernel(TCudaBufferPtr<float> matrices, double lambdaNonDiag, double lambdaDiag)
            : Matrices(matrices)
            , LambdaNonDiag(lambdaNonDiag)
            , LambdaDiag(lambdaDiag)
        {
        }

        Y_SAVELOAD_DEFINE(Matrices, LambdaNonDiag, LambdaDiag);

        void Run(const TCudaStream& stream) const {
            const ui32 x = Matrices.ObjectSize();
            const ui32 rowSize = (-1 + sqrt(1 + 8 * x)) / 2;
            CB_ENSURE(rowSize * (rowSize + 1) / 2 == x);

            NKernel::Regularize(Matrices.Get(),
                                rowSize,
                                Matrices.ObjectCount(),
                                LambdaNonDiag,
                                LambdaDiag,
                                stream.GetStream());
        }
    };

    class TCholeskySolverKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<float> Matrices;
        TCudaBufferPtr<float> Solutions;
        TSlice SolutionsSlice;
        bool RemoveLast;

    public:
        TCholeskySolverKernel() = default;

        TCholeskySolverKernel(TCudaBufferPtr<float> matrices, TCudaBufferPtr<float> solutions, TSlice solutionsSlice, bool removeLast)
            : Matrices(matrices)
            , Solutions(solutions)
            , SolutionsSlice(solutionsSlice)
            , RemoveLast(removeLast)
        {
        }

        Y_SAVELOAD_DEFINE(Matrices, Solutions, SolutionsSlice, RemoveLast);

        void Run(const TCudaStream& stream) const {
            const ui32 rowSize = Solutions.ObjectSize();
            CB_ENSURE(rowSize * (rowSize + 1) / 2 == Matrices.ObjectSize());
            CB_ENSURE(Matrices.ObjectCount() == SolutionsSlice.Size());

            NKernel::CholeskySolver(Matrices.Get(),
                                    Solutions.GetForObject(SolutionsSlice.Left),
                                    rowSize,
                                    static_cast<int>(SolutionsSlice.Size()),
                                    RemoveLast,
                                    stream.GetStream());

            if (RemoveLast) {
                NKernel::ZeroMean(Solutions.GetForObject(SolutionsSlice.Left),
                                  rowSize, SolutionsSlice.Size(),
                                  stream.GetStream());
            }
        }
    };

    class TCalcScoresKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCBinFeature> BinFeaturesSource;
        TSlice BinFeaturesSlice;
        TCudaBufferPtr<const float> LinearSystem;
        TCudaBufferPtr<const float> Solutions;
        TSlice SolutionsSlice;
        TCudaBufferPtr<TCBinFeature> BinFeaturesResult;
        TCudaBufferPtr<float> Scores;

    public:
        TCalcScoresKernel() = default;

        TCalcScoresKernel(TCudaBufferPtr<const TCBinFeature> binFeaturesSource,
                          TSlice binFeaturesSlice,
                          TCudaBufferPtr<const float> linearSystem,
                          TCudaBufferPtr<const float> solutions,
                          TSlice solutionsSlice,
                          TCudaBufferPtr<TCBinFeature> binFeaturesResult,
                          TCudaBufferPtr<float> scores)
            : BinFeaturesSource(binFeaturesSource)
            , BinFeaturesSlice(binFeaturesSlice)
            , LinearSystem(linearSystem)
            , Solutions(solutions)
            , SolutionsSlice(solutionsSlice)
            , BinFeaturesResult(binFeaturesResult)
            , Scores(scores)
        {
        }

        Y_SAVELOAD_DEFINE(BinFeaturesSource, BinFeaturesSlice, LinearSystem, Solutions, SolutionsSlice, Scores, BinFeaturesResult);

        void Run(const TCudaStream& stream) const {
            const ui32 matrixCount = LinearSystem.ObjectCount();
            const ui32 rowSize = Solutions.ObjectSize();
            CB_ENSURE(BinFeaturesSlice.Size() == SolutionsSlice.Size());
            CB_ENSURE(matrixCount == SolutionsSlice.Size());

            NCudaLib::CopyMemoryAsync(BinFeaturesSource.GetForObject(BinFeaturesSlice.Left),
                                      BinFeaturesResult.GetForObject(SolutionsSlice.Left),
                                      SolutionsSlice.Size(),
                                      stream);

            NKernel::CalcScores(LinearSystem.Get(),
                                Solutions.GetForObject(SolutionsSlice.Left),
                                Scores.GetForObject(SolutionsSlice.Left),
                                rowSize,
                                matrixCount,
                                stream.GetStream());
        }
    };

    class TCopyReducedTempResultKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Source;
        TCudaBufferPtr<float> Dest;
        TSlice DestSlice;

    public:
        TCopyReducedTempResultKernel() = default;

        TCopyReducedTempResultKernel(TCudaBufferPtr<const float> source,
                                     TCudaBufferPtr<float> dest,
                                     TSlice destSlice)
            : Source(source)
            , Dest(dest)
            , DestSlice(destSlice)
        {
        }

        Y_SAVELOAD_DEFINE(Source, Dest, DestSlice);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Source.ObjectCount() == DestSlice.Size());
            CB_ENSURE(Source.ObjectSize() == Dest.ObjectSize());

            const ui64 singleObjectSize = Dest.ObjectSize();
            NCudaLib::CopyMemoryAsync(Source.Get(), Dest.GetForObject(DestSlice.Left), singleObjectSize * DestSlice.Size(), stream);
        }
    };

    class TSelectBestSplitKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Scores;
        TCudaBufferPtr<const TCBinFeature> BinFeature;
        ui32 BestIndexBias;
        TCudaBufferPtr<TBestSplitPropertiesWithIndex> Best;

    public:
        TSelectBestSplitKernel() = default;

        TSelectBestSplitKernel(TCudaBufferPtr<const float> scores,
                               TCudaBufferPtr<const TCBinFeature> binFeature,
                               ui32 bestIndexBias,
                               TCudaBufferPtr<TBestSplitPropertiesWithIndex> best)
            : Scores(scores)
            , BinFeature(binFeature)
            , BestIndexBias(bestIndexBias)
            , Best(best)
        {
        }

        Y_SAVELOAD_DEFINE(Scores, BinFeature, BestIndexBias, Best);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(BinFeature.Size() == Scores.Size());
            NKernel::SelectBestSplit(Scores.Get(), BinFeature.Get(), BinFeature.Size(), BestIndexBias, Best.Get(), stream.GetStream());
        }
    };

    class TComputePairwiseHistogramKernel: public TStatelessKernel {
    private:
        NCatboostCuda::EFeaturesGroupingPolicy Policy;
        TCudaBufferPtr<const TCFeature> Features;
        TSlice BinFeaturesSlice;
        TCudaBufferPtr<const ui32> CompressedIndex;
        TCudaBufferPtr<const uint2> Pairs;
        TCudaBufferPtr<const float> Weight;
        TCudaBufferPtr<const TDataPartition> Partition;
        TCudaBufferPtr<const TPartitionStatistics> PartitionStats;
        ui32 Depth;
        ui32 HistLineSize;
        bool FullPass;
        TCudaBufferPtr<float> Histogram;

    public:
        TComputePairwiseHistogramKernel() = default;

        TComputePairwiseHistogramKernel(NCatboostCuda::EFeaturesGroupingPolicy policy,
                                        TCudaBufferPtr<const TCFeature> features,
                                        TSlice binFeaturesSlice,
                                        TCudaBufferPtr<const ui32> compressedIndex,
                                        TCudaBufferPtr<const uint2> pairs,
                                        TCudaBufferPtr<const float> weight,
                                        TCudaBufferPtr<const TDataPartition> partition,
                                        TCudaBufferPtr<const TPartitionStatistics> partitionStats,
                                        ui32 depth,
                                        ui32 histLineSize,
                                        bool fullPass,
                                        TCudaBufferPtr<float> histogram)
            : Policy(policy)
            , Features(features)
            , BinFeaturesSlice(binFeaturesSlice)
            , CompressedIndex(compressedIndex)
            , Pairs(pairs)
            , Weight(weight)
            , Partition(partition)
            , PartitionStats(partitionStats)
            , Depth(depth)
            , HistLineSize(histLineSize)
            , FullPass(fullPass)
            , Histogram(histogram)
        {
        }

        Y_SAVELOAD_DEFINE(Policy, Features,
                          BinFeaturesSlice,
                          CompressedIndex,
                          Pairs,
                          Weight,
                          Partition,
                          PartitionStats,
                          Depth,
                          HistLineSize,
                          FullPass,
                          Histogram);

        void Run(const TCudaStream& stream) const {
            if (Depth == 0) {
                Y_VERIFY(FullPass);
            }
            const auto leavesCount = static_cast<ui32>(1u << Depth);
            const ui32 partCount = leavesCount * leavesCount;

#define DISPATCH(KernelName)                               \
    NKernel::KernelName(Features.Get(),                    \
                        static_cast<int>(Features.Size()), \
                        CompressedIndex.Get(),             \
                        Pairs.Get(), Pairs.Size(),         \
                        Weight.Get(),                      \
                        Partition.Get(),                   \
                        partCount,                         \
                        HistLineSize,                      \
                        FullPass,                          \
                        Histogram.Get(),                   \
                        stream.GetStream());

            {
                switch (Policy) {
                    case NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures: {
                        DISPATCH(ComputePairwiseHistogramBinary)
                        break;
                    }
                    case NCatboostCuda::EFeaturesGroupingPolicy::HalfByteFeatures: {
                        DISPATCH(ComputePairwiseHistogramHalfByte)
                        break;
                    }
                    case NCatboostCuda::EFeaturesGroupingPolicy::OneByteFeatures: {
                        DISPATCH(ComputePairwiseHistogramOneByte)
                        break;
                    }
                    default: {
                        CB_ENSURE(false);
                    }
                }
#undef DISPATCH
            }
            if (Policy != NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures) {
                NKernel::ScanPairwiseHistograms(Features.Get(),
                                                Features.Size(),
                                                partCount,
                                                HistLineSize,
                                                FullPass,
                                                Histogram.Get(),
                                                stream.GetStream());

                NKernel::BuildBinaryFeatureHistograms(Features.Get(),
                                                      Features.Size(),
                                                      Partition.Get(),
                                                      PartitionStats.Get(),
                                                      partCount,
                                                      HistLineSize,
                                                      FullPass,
                                                      Histogram.Get(),
                                                      stream.GetStream());
            }

            if (!FullPass) {
                NKernel::UpdatePairwiseHistograms(BinFeaturesSlice.Left, BinFeaturesSlice.Size(),
                                                  Partition.Get(), partCount,
                                                  HistLineSize,
                                                  Histogram.Get(),
                                                  stream.GetStream());
            }
        }
    };
}

inline void MakeLinearSystem(const TCudaBuffer<float, NCudaLib::TStripeMapping>& pointwiseHist,
                             const TCudaBuffer<TPartitionStatistics, NCudaLib::TStripeMapping>& pointStats,
                             const TCudaBuffer<float, NCudaLib::TStripeMapping>& pairwiseHist,
                             const ui64 totalBinFeatureCounts,
                             const TSlice& workingSlice,
                             TCudaBuffer<float, NCudaLib::TStripeMapping>& systems,
                             ui32 streamId = 0) {
    using TKernel = NKernelHost::TMakeLinearSystemKernel;
    LaunchKernels<TKernel>(systems.NonEmptyDevices(), streamId,
                           pointwiseHist, pointStats,
                           pairwiseHist,
                           totalBinFeatureCounts,
                           workingSlice,
                           systems);
}

inline void PrepareSystemForCholesky(const TCudaBuffer<float, NCudaLib::TStripeMapping>& linearSystem,
                                     TCudaBuffer<float, NCudaLib::TStripeMapping>& sqrtMatrices,
                                     const NCudaLib::TDistributedObject<TSlice>& solutionsSlice,
                                     TCudaBuffer<float, NCudaLib::TStripeMapping>& solutions,
                                     ui32 streamId = 0) {
    using TKernel = NKernelHost::TExtractMatricesAndTargetsKernel;
    LaunchKernels<TKernel>(linearSystem.NonEmptyDevices(), streamId, linearSystem, sqrtMatrices, solutions, solutionsSlice);
}

inline void CholeskySolver(TCudaBuffer<float, NCudaLib::TStripeMapping>& sqrtMatrices,
                           NCudaLib::TDistributedObject<TSlice>& solutionsSlice,
                           TCudaBuffer<float, NCudaLib::TStripeMapping>& solutions,
                           bool removeLastRow,
                           ui32 streamId = 0) {
    using TKernel = NKernelHost::TCholeskySolverKernel;
    LaunchKernels<TKernel>(sqrtMatrices.NonEmptyDevices(), streamId, sqrtMatrices, solutions, solutionsSlice, removeLastRow);
}

inline void Regularize(TCudaBuffer<float, NCudaLib::TStripeMapping>& sqrtMatrices,
                       double LambdaNonDiag,
                       double LambdaDiag,
                       ui32 streamId = 0) {
    using TKernel = NKernelHost::TRegularizeKernel;
    LaunchKernels<TKernel>(sqrtMatrices.NonEmptyDevices(), streamId, sqrtMatrices, LambdaNonDiag, LambdaDiag);
}

inline void ZeroMean(const NCudaLib::TDistributedObject<TSlice>& solutionsSlice,
                     TCudaBuffer<float, NCudaLib::TStripeMapping>& solutions,
                     ui32 streamId = 0) {
    using TKernel = NKernelHost::TZeroMeanKernel;
    LaunchKernels<TKernel>(solutions.NonEmptyDevices(), streamId, solutions, solutionsSlice);
}

inline void ComputeScores(const TMirrorBuffer<const TCBinFeature>& binFeatures,
                          const NCudaLib::TDistributedObject<TSlice>& binFeaturesAccessorSlice,
                          const TCudaBuffer<float, NCudaLib::TStripeMapping>& linearSystem,
                          const NCudaLib::TDistributedObject<TSlice>& solutionsSlice,
                          const TCudaBuffer<float, NCudaLib::TStripeMapping>& solutions,
                          TCudaBuffer<TCBinFeature, NCudaLib::TStripeMapping>& binFeaturesResult,
                          TCudaBuffer<float, NCudaLib::TStripeMapping>& scores,
                          ui32 streamId = 0) {
    using TKernel = NKernelHost::TCalcScoresKernel;
    LaunchKernels<TKernel>(solutions.NonEmptyDevices(), streamId,
                           binFeatures, binFeaturesAccessorSlice, linearSystem, solutions, solutionsSlice, binFeaturesResult, scores);
}

inline void CopyReducedTempResult(const TStripeBuffer<const float>& source,
                                  const NCudaLib::TDistributedObject<TSlice>& destSlice,
                                  TCudaBuffer<float, NCudaLib::TStripeMapping>& dest,
                                  ui32 streamId = 0) {
    using TKernel = NKernelHost::TCopyReducedTempResultKernel;
    LaunchKernels<TKernel>(source.NonEmptyDevices(), streamId,
                           source, dest, destSlice);
}

template <class TUi32>
inline void UpdatePairwiseBins(const NCudaLib::TCudaBuffer<TUi32, NCudaLib::TStripeMapping>& cindex,
                               const NCudaLib::TDistributedObject<TCFeature>& feature, ui32 bin,
                               const ui32 depth,
                               const TCudaBuffer<uint2, NCudaLib::TStripeMapping>& pairs,
                               NCudaLib::TCudaBuffer<ui32, NCudaLib::TStripeMapping>& pairBins,
                               ui32 stream = 0) {
    using TKernel = NKernelHost::TUpdateBinsPairsKernel;
    LaunchKernels<TKernel>(pairBins.NonEmptyDevices(), stream, feature, bin, cindex, pairs, depth, pairBins);
}

inline void SelectOptimalSplit(const TCudaBuffer<float, NCudaLib::TStripeMapping>& scores,
                               const TCudaBuffer<TCBinFeature, NCudaLib::TStripeMapping>& features,
                               TCudaBuffer<TBestSplitPropertiesWithIndex, NCudaLib::TStripeMapping>& result,
                               ui32 stream = 0) {
    NCudaLib::TDistributedObject<ui32> offsets = CreateDistributedObject<ui32>(0u);

    for (ui32 i = 0; i < offsets.DeviceCount(); ++i) {
        offsets.Set(i, features.GetMapping().DeviceSlice(i).Left);
    }

    using TKernel = NKernelHost::TSelectBestSplitKernel;
    LaunchKernels<TKernel>(features.NonEmptyDevices(), stream, scores, features, offsets, result);
}

inline void ComputeBlockPairwiseHist2(NCatboostCuda::EFeaturesGroupingPolicy policy,
                                      const TCudaBuffer<const TCFeature, NCudaLib::TStripeMapping>& gridBlock,
                                      const TSlice& binFeaturesSlice,
                                      const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& compressedIndex,
                                      const TCudaBuffer<float, NCudaLib::TStripeMapping>& pairWeight,
                                      const TCudaBuffer<uint2, NCudaLib::TStripeMapping>& pairs,
                                      const TCudaBuffer<TDataPartition, NCudaLib::TStripeMapping>& dataPartitions,
                                      const TCudaBuffer<TPartitionStatistics, NCudaLib::TStripeMapping>& partStats,
                                      ui32 depth,
                                      ui32 histogramLineSize,
                                      bool fullPass,
                                      TCudaBuffer<float, NCudaLib::TStripeMapping>& histograms,
                                      ui32 stream) {
    using TKernel = NKernelHost::TComputePairwiseHistogramKernel;

    LaunchKernels<TKernel>(gridBlock.NonEmptyDevices(),
                           stream,
                           policy,
                           gridBlock,
                           binFeaturesSlice,
                           compressedIndex,
                           pairs,
                           pairWeight,
                           dataPartitions,
                           partStats,
                           depth,
                           histogramLineSize,
                           fullPass,
                           histograms);
}
