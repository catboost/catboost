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
#include <catboost/private/libs/options/enums.h>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>
#include <catboost/cuda/gpu_data/folds_histogram.h>

#include <util/generic/cast.h>

#include <cmath>

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

        void Run(const TCudaStream& stream) const;
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
            NKernel::UpdateBinsPairs(Feature, Bin, CompressedIndex.Get(), Pairs.Get(), SafeIntegerCast<ui32>(Pairs.Size()), Depth, Bins.Get(), stream.GetStream());
        }
    };

    class TExtractMatricesAndTargetsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> LinearSystem;
        TCudaBufferPtr<float> LowTriangleMatrices;
        TCudaBufferPtr<float> Solutions;
        TCudaBufferPtr<float> MatrixDiag;
        TSlice SolutionsSlice;

    public:
        TExtractMatricesAndTargetsKernel() = default;

        TExtractMatricesAndTargetsKernel(TCudaBufferPtr<const float> linearSystem,
                                         TCudaBufferPtr<float> matrices,
                                         TCudaBufferPtr<float> solutions,
                                         TCudaBufferPtr<float> matrixDiag,
                                         TSlice solutionsSlice)
            : LinearSystem(linearSystem)
            , LowTriangleMatrices(matrices)
            , Solutions(solutions)
            , MatrixDiag(matrixDiag)
            , SolutionsSlice(solutionsSlice)
        {
        }

        Y_SAVELOAD_DEFINE(LinearSystem,
                          LowTriangleMatrices,
                          Solutions,
                          MatrixDiag,
                          SolutionsSlice);

        void Run(const TCudaStream& stream) const;
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

        void Run(const TCudaStream& stream) const;
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

    class TCholeskySolverKernel: public TKernelBase<NKernel::TCholeskySolverContext> {
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

        using TKernelContext = NKernel::TCholeskySolverContext;
        THolder<TKernelContext> PrepareContext(IMemoryManager& manager) const;

        Y_SAVELOAD_DEFINE(Matrices, Solutions, SolutionsSlice, RemoveLast);

        void Run(const TCudaStream& stream, TKernelContext&) const;
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

        void Run(const TCudaStream& stream) const;
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
        double ScoreBeforeSplit;
        TCudaBufferPtr<const float> FeatureWeights;
        ui32 BestIndexBias;
        TCudaBufferPtr<TBestSplitPropertiesWithIndex> Best;

    public:
        TSelectBestSplitKernel() = default;

        TSelectBestSplitKernel(TCudaBufferPtr<const float> scores,
                               TCudaBufferPtr<const TCBinFeature> binFeature,
                               double scoreBeforeSplit,
                               TCudaBufferPtr<const float> featureWeights,
                               ui32 bestIndexBias,
                               TCudaBufferPtr<TBestSplitPropertiesWithIndex> best)
            : Scores(scores)
            , BinFeature(binFeature)
            , ScoreBeforeSplit(scoreBeforeSplit)
            , FeatureWeights(featureWeights)
            , BestIndexBias(bestIndexBias)
            , Best(best)
        {
        }

        Y_SAVELOAD_DEFINE(Scores, BinFeature, ScoreBeforeSplit, FeatureWeights, BestIndexBias, Best);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(BinFeature.Size() == Scores.Size());

            NKernel::SelectBestSplit(
                Scores.Get(), BinFeature.Get(), BinFeature.Size(),
                ScoreBeforeSplit, FeatureWeights.Get(),
                BestIndexBias, Best.Get(), stream.GetStream());
        }
    };

    class TComputePairwiseHistogramKernel: public TStatelessKernel {
    private:
        NCatboostCuda::EFeaturesGroupingPolicy Policy;
        TCudaBufferPtr<const TCFeature> Features;
        TCudaHostBufferPtr<const TCFeature> FeaturesCpu;
        NCatboostCuda::TFoldsHistogram FoldsHist;
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
        int ParallelStreamsCount;

    public:
        TComputePairwiseHistogramKernel() = default;

        TComputePairwiseHistogramKernel(NCatboostCuda::EFeaturesGroupingPolicy policy,
                                        TCudaBufferPtr<const TCFeature> features,
                                        TCudaHostBufferPtr<const TCFeature> featuresCpu,
                                        NCatboostCuda::TFoldsHistogram foldsHist,
                                        TSlice binFeaturesSlice,
                                        TCudaBufferPtr<const ui32> compressedIndex,
                                        TCudaBufferPtr<const uint2> pairs,
                                        TCudaBufferPtr<const float> weight,
                                        TCudaBufferPtr<const TDataPartition> partition,
                                        TCudaBufferPtr<const TPartitionStatistics> partitionStats,
                                        ui32 depth,
                                        ui32 histLineSize,
                                        bool fullPass,
                                        TCudaBufferPtr<float> histogram,
                                        int parallelStreams)
            : Policy(policy)
            , Features(features)
            , FeaturesCpu(featuresCpu)
            , FoldsHist(foldsHist)
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
            , ParallelStreamsCount(parallelStreams)
        {
        }

        Y_SAVELOAD_DEFINE(Policy,
                          Features,
                          FeaturesCpu,
                          BinFeaturesSlice,
                          FoldsHist,
                          CompressedIndex,
                          Pairs,
                          Weight,
                          Partition,
                          PartitionStats,
                          Depth,
                          HistLineSize,
                          FullPass,
                          Histogram,
                          ParallelStreamsCount);

        void Run(const TCudaStream& stream) const;
    };

    class TFillPairDer2OnlyKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Ders2;
        TCudaBufferPtr<const float> GroupDers2;
        TCudaBufferPtr<const ui32> Qids;
        TCudaBufferPtr<const uint2> Pairs;
        TCudaBufferPtr<float> PairDer2;

    public:
        TFillPairDer2OnlyKernel() = default;

        TFillPairDer2OnlyKernel(TCudaBufferPtr<const float> ders2,
                                TCudaBufferPtr<const float> groupDers2,
                                TCudaBufferPtr<const ui32> qids,
                                TCudaBufferPtr<const uint2> pairs,
                                TCudaBufferPtr<float> pairDer2)
            : Ders2(ders2)
            , GroupDers2(groupDers2)
            , Qids(qids)
            , Pairs(pairs)
            , PairDer2(pairDer2)
        {
        }

        Y_SAVELOAD_DEFINE(Ders2, GroupDers2, Qids, Pairs, PairDer2);

        void Run(const TCudaStream& stream) const {
            NKernel::FillPairDer2Only(Ders2.Get(),
                                      GroupDers2.Get(),
                                      Qids.Get(),
                                      Pairs.Get(),
                                      SafeIntegerCast<ui32>(Pairs.Size()),
                                      PairDer2.Get(),
                                      stream.GetStream());
        }
    };

    class TFillPairBinsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const uint2> Pairs;
        TCudaBufferPtr<const ui32> Bins;
        ui32 BinCount;
        TCudaBufferPtr<ui32> PairBins;

    public:
        TFillPairBinsKernel() = default;

        TFillPairBinsKernel(TCudaBufferPtr<const uint2> pairs, TCudaBufferPtr<const ui32> bins, ui32 binCount, TCudaBufferPtr<ui32> pairBins)
            : Pairs(pairs)
            , Bins(bins)
            , BinCount(binCount)
            , PairBins(pairBins)
        {
        }

        Y_SAVELOAD_DEFINE(Pairs, Bins, BinCount, PairBins);

        void Run(const TCudaStream& stream) const {
            NKernel::FillPairBins(Pairs.Get(), Bins.Get(), BinCount, SafeIntegerCast<ui32>(Pairs.Size()), PairBins.Get(), stream.GetStream());
        }
    };

    class TZeroSameLeafBinWeightsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const uint2> Pairs;
        TCudaBufferPtr<const ui32> Bins;
        TCudaBufferPtr<float> PairWeights;

    public:
        TZeroSameLeafBinWeightsKernel() = default;

        TZeroSameLeafBinWeightsKernel(TCudaBufferPtr<const uint2> pairs, TCudaBufferPtr<const ui32> bins, TCudaBufferPtr<float> pairWeights)
            : Pairs(pairs)
            , Bins(bins)
            , PairWeights(pairWeights)
        {
        }

        Y_SAVELOAD_DEFINE(Pairs, Bins, PairWeights);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Pairs.Size() == PairWeights.Size());
            NKernel::ZeroSameLeafBinWeights(Pairs.Get(), Bins.Get(), SafeIntegerCast<ui32>(Pairs.Size()), PairWeights.Get(), stream.GetStream());
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
                                     TCudaBuffer<float, NCudaLib::TStripeMapping>& matrixDiag,
                                     ui32 streamId = 0) {
    using TKernel = NKernelHost::TExtractMatricesAndTargetsKernel;
    LaunchKernels<TKernel>(linearSystem.NonEmptyDevices(), streamId, linearSystem, sqrtMatrices, solutions, matrixDiag, solutionsSlice);
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
                               double scoreBeforeSplit,
                               const TCudaBuffer<float, NCudaLib::TMirrorMapping>& featureWeights,
                               TCudaBuffer<TBestSplitPropertiesWithIndex, NCudaLib::TStripeMapping>& result,
                               ui32 stream = 0) {
    NCudaLib::TDistributedObject<ui32> offsets = CreateDistributedObject<ui32>(0u);

    for (ui32 i = 0; i < offsets.DeviceCount(); ++i) {
        offsets.Set(i, features.GetMapping().DeviceSlice(i).Left);
    }

    using TKernel = NKernelHost::TSelectBestSplitKernel;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), stream, scores, features, scoreBeforeSplit, featureWeights, offsets, result);
}

inline void ComputeBlockPairwiseHist2(NCatboostCuda::EFeaturesGroupingPolicy policy,
                                      const TCudaBuffer<const TCFeature, NCudaLib::TStripeMapping>& gridBlock,
                                      const TCudaBuffer<const TCFeature, NCudaLib::TStripeMapping, NCudaLib::EPtrType::CudaHost>& gridBlockCpu,
                                      const NCatboostCuda::TFoldsHistogram& foldsHistogram,
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
                                      int parallelStreamsCount,
                                      ui32 stream) {
    using TKernel = NKernelHost::TComputePairwiseHistogramKernel;

    LaunchKernels<TKernel>(pairs.NonEmptyDevices(),
                           stream,
                           policy,
                           gridBlock,
                           gridBlockCpu,
                           foldsHistogram,
                           binFeaturesSlice,
                           compressedIndex,
                           pairs,
                           pairWeight,
                           dataPartitions,
                           partStats,
                           depth,
                           histogramLineSize,
                           fullPass,
                           histograms,
                           parallelStreamsCount);
}

template <class TMapping>
inline void FillGroupwisePairDer2(const TCudaBuffer<float, TMapping>& shiftedDer2,
                                  const TCudaBuffer<float, TMapping>& groupsDer2Sum,
                                  const TCudaBuffer<const ui32, TMapping>& qids,
                                  const TCudaBuffer<uint2, TMapping>& pairs,
                                  TCudaBuffer<float, TMapping>* pairWeights,
                                  ui32 stream = 0) {
    using TKernel = NKernelHost::TFillPairDer2OnlyKernel;
    LaunchKernels<TKernel>(shiftedDer2.NonEmptyDevices(),
                           stream,
                           shiftedDer2,
                           groupsDer2Sum,
                           qids,
                           pairs,
                           pairWeights);
}

template <class TMapping>
inline void FillPairBins(const TCudaBuffer<const ui32, TMapping>& bins,
                         ui32 binCount,
                         const TStripeBuffer<uint2>& pairs,
                         TStripeBuffer<ui32>* pairBins,
                         ui32 stream = 0) {
    using TKernel = NKernelHost::TFillPairBinsKernel;
    LaunchKernels<TKernel>(bins.NonEmptyDevices(),
                           stream,
                           pairs,
                           bins,
                           binCount,
                           pairBins);
}

template <class TMapping>
inline void ZeroSameLeafBinWeights(const TCudaBuffer<const ui32, TMapping>& bins,
                                   const TStripeBuffer<uint2>& pairs,
                                   TStripeBuffer<float>* pairWeights,
                                   ui32 stream = 0) {
    using TKernel = NKernelHost::TZeroSameLeafBinWeightsKernel;
    LaunchKernels<TKernel>(bins.NonEmptyDevices(),
                           stream,
                           pairs,
                           bins,
                           pairWeights);
}
